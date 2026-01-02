import argparse
import gc
import json
import platform
import threading
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import pynvml
import psutil
import statistics
from omegaconf import OmegaConf

from scope.core.pipelines.utils import Quantization
from scope.core.pipelines.registry import PipelineRegistry
from scope.server.download_models import download_models
from scope.server.models_config import models_are_downloaded
from scope.core.config import get_model_file_path, get_models_dir

class HardwareInfo:
    """Collects and stores hardware information."""

    def __init__(self):
        self._info = self._collect_info()

    def _collect_info(self) -> dict[str, Any]:
        return {
            "gpu": self._get_gpu_info(),
            "cpu": self._get_cpu_info(),
            "memory": self._get_memory_info(),
            "platform": self._get_platform_info(),
        }

    def _get_gpu_info(self) -> dict[str, Any]:
        gpu_info = {"available": torch.cuda.is_available(), "count": 0, "devices": []}
        if not torch.cuda.is_available():
            return gpu_info

        gpu_info["count"] = torch.cuda.device_count()
        gpu_info["cuda_version"] = torch.version.cuda

        pynvml.nvmlInit()
        for i in range(gpu_info["count"]):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes): name = name.decode("utf-8")

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes): driver = driver.decode("utf-8")

            gpu_info["devices"].append({
                "index": i,
                "name": name,
                "memory_total_gb": mem.total / (1024**3),
                "driver_version": driver,
            })
        pynvml.nvmlShutdown()

        if not gpu_info["devices"]:
            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "memory_total_gb": props.total_memory / (1024**3),
                })

        return gpu_info

    def _get_cpu_info(self) -> dict[str, Any]:
        return {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "model": platform.processor(),
        }

    def _get_memory_info(self) -> dict[str, Any]:
        mem = psutil.virtual_memory()
        return {"total_gb": mem.total / (1024**3), "available_gb": mem.available / (1024**3)}

    def _get_platform_info(self) -> dict[str, Any]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }

    def to_dict(self) -> dict[str, Any]:
        return self._info

    def get_primary_gpu_vram_gb(self) -> float:
        if not self._info["gpu"]["available"] or not self._info["gpu"]["devices"]:
            return 0.0
        return self._info["gpu"]["devices"][0]["memory_total_gb"]

class ResourceMonitor:
    def __init__(self, interval_ms: int = 100, device_index: int = 0):
        self.interval_ms = interval_ms
        self.device_index = device_index
        self._monitoring = False
        self._thread = None
        self._samples = []
        self._lock = threading.Lock()
        self._process = psutil.Process()
        self._pynvml_initialized = False
        self._gpu_handle = None

        pynvml.nvmlInit()
        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._pynvml_initialized = True

    def start(self):
        if self._monitoring: return
        self._monitoring = True
        self._samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._monitoring: return
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self):
        while self._monitoring:
            sample = self._collect_sample()
            with self._lock:
                self._samples.append(sample)
            time.sleep(self.interval_ms / 1000.0)

    def _collect_sample(self) -> dict[str, Any]:
        sample = {}
        if torch.cuda.is_available():
            sample["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(self.device_index) / (1024**3)
            if self._pynvml_initialized and self._gpu_handle:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                sample["gpu_utilization_percent"] = util.gpu

        sample["system_cpu_percent"] = psutil.cpu_percent()
        return sample

    def get_statistics(self) -> dict[str, float]:
        with self._lock: samples = self._samples.copy()
        if not samples: return {}

        stats = {}
        keys = ["gpu_memory_allocated_gb", "gpu_utilization_percent", "system_cpu_percent"]
        for key in keys:
            values = [s[key] for s in samples if key in s]
            if values:
                stats[f"{key}_avg"] = sum(values) / len(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_std"] = statistics.stdev(values)
        return stats

    def cleanup(self):
        self.stop()
        if self._pynvml_initialized:
            pynvml.nvmlShutdown()


class ConfigurationMatrix:
    STANDARD_RESOLUTIONS = [
        (320, 576),
        (480, 832),
        (512, 512),
        (576, 1024),
        (768, 1344),
    ]

    DEFAULT_PROMPT = "A realistic video of a serene landscape with rolling hills, a clear blue sky, and a gentle stream."

    def __init__(self, pipelines=None, resolutions=None):
        self.selected_pipelines = pipelines
        self.custom_resolutions = resolutions

    def build(self) -> list[dict]:
        all_pipelines = PipelineRegistry.list_pipelines()

        if self.selected_pipelines:
            pipelines = [p for p in all_pipelines if p in self.selected_pipelines]
        else:
            pipelines = [p for p in all_pipelines if p != "passthrough"]

        configurations = []
        for pid in pipelines:
            resolutions = self._get_resolutions(pid)

            for h, w in resolutions:
                config = {
                    "pipeline_id": pid,
                    "height": h,
                    "width": w,
                    "prompt": self.DEFAULT_PROMPT,
                }
                configurations.append(config)

        return configurations

    def _get_resolutions(self, pid: str) -> list[tuple[int, int]]:
        if self.custom_resolutions:
            return self.custom_resolutions

        pipeline_class = PipelineRegistry.get(pid)
        if not pipeline_class: return []
        default_cfg = pipeline_class.get_config_class()()

        res_set = {(default_cfg.height, default_cfg.width)}

        for h, w in self.STANDARD_RESOLUTIONS:
            res_set.add((h, w))

        return sorted(list(res_set))


class BenchmarkRunner:
    def __init__(self, warmup_iterations=5, iterations=30, compile_model=False):
        self.warmup_iterations = warmup_iterations
        self.iterations = iterations
        self.compile_model = compile_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_config(self, config: dict) -> dict:
        pipeline_id = config["pipeline_id"]
        print(f"\n--- Benchmarking {pipeline_id} [{config['height']}x{config['width']}] ---")

        if not models_are_downloaded(pipeline_id):
            print(f"Downloading models for {pipeline_id}...")
            try:
                download_models(pipeline_id)
                print(f"Models downloaded successfully for {pipeline_id}")
            except Exception as e:
                print(f"ERROR: Failed to download models: {e}")
                return {"error": f"Model download failed: {str(e)}"}

        pipeline = None
        try:
            pipeline = self._init_pipeline(config)
            inputs = {"prompts": [{"text": config["prompt"], "weight": 100}]}

            if pipeline_id == "streamdiffusionv2":
                inputs["video"] = torch.randn(
                        1, 3, 4, config["height"], config["width"],
                        device=self.device, dtype=torch.bfloat16
                    )

            print(f"Warmup ({self.warmup_iterations} iterations)...")
            try:
                for _ in range(self.warmup_iterations):
                    pipeline(**inputs)
            except Exception as e:
                raise Exception(f"Warmup failed: {e}")

            print(f"Measuring ({self.iterations} iterations)...")
            monitor = ResourceMonitor()
            latencies = []
            fps_measures = []

            try:
                monitor.start()
                for _ in range(self.iterations):
                    t0 = time.time()
                    output = pipeline(**inputs)
                    latency = time.time() - t0
                    latencies.append(latency)
                    fps_measures.append(output.shape[0] / latency)
                    del output
            finally:
                try:
                    monitor.stop()
                    resource_stats = monitor.get_statistics()
                    monitor.cleanup()
                except Exception:
                    resource_stats = {}

            if not latencies:
                return {"error": "No successful iterations"}

            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

            fps_avg = statistics.mean(fps_measures)
            fps_min = min(fps_measures)
            fps_max = max(fps_measures)

            results = {
                "fps_avg": round(fps_avg, 2),
                "fps_min": round(fps_min, 2),
                "fps_max": round(fps_max, 2),
                "latency_avg_sec": round(avg_latency, 4),
                "latency_min_sec": round(min_latency, 4),
                "latency_max_sec": round(max_latency, 4),
                "jitter_sec": round(jitter, 6),
                **resource_stats
            }

            print(f"-> FPS: {results['fps_avg']} | Latency: {results['latency_avg_sec']}s | Jitter: {results['jitter_sec']}s")
            return results

        except Exception as e:
            print(f"ERROR: {e}")
            return {"error": str(e)}
        finally:
            del pipeline
            self._clear_memory()
            time.sleep(3.0)

    def _init_pipeline(self, config: dict):
        pid = config["pipeline_id"]
        pipeline_class = PipelineRegistry.get(pid)

        model_config = OmegaConf.load(Path(__file__).parent / "src/scope/core/pipelines" / pid / "model.yaml")
        pipeline_config = {
            "model_dir": str(get_models_dir()),
            "model_config": model_config,
            "height": config["height"],
            "width": config["width"],
        }

        def model_path(p): return str(get_model_file_path(p))
        wan_enc = model_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        wan_tok = model_path("Wan2.1-T2V-1.3B/google/umt5-xxl")

        paths = {}
        if pid == "streamdiffusionv2":
            paths = {"generator_path": model_path("StreamDiffusionV2/wan_causal_dmd_v2v/model.pt")}
        elif pid == "longlive":
            paths = {
                "generator_path": model_path("LongLive-1.3B/models/longlive_base.pt"),
                "lora_path": model_path("LongLive-1.3B/models/lora.pt")
            }
        elif pid == "krea_realtime_video":
            paths = {
                "generator_path": model_path("krea-realtime-video/krea-realtime-video-14b.safetensors"),
                "vae_path": model_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
            }
        elif pid == "reward_forcing":
            paths = {"generator_path": model_path("Reward-Forcing-T2V-1.3B/rewardforcing.pt")}

        pipeline_config.update(paths)
        if "text_encoder_path" not in pipeline_config: pipeline_config["text_encoder_path"] = wan_enc
        if "tokenizer_path" not in pipeline_config: pipeline_config["tokenizer_path"] = wan_tok

        quantization = Quantization.FP8_E4M3FN if pid == "krea_realtime_video" else None
        args = {
            "config": OmegaConf.create(pipeline_config),
            "device": self.device,
            "dtype": torch.bfloat16
        }
        if quantization:
            args.update({"quantization": quantization})

        if pid == "krea_realtime_video":
            args["compile"] = self.compile_model
        return pipeline_class(**args)

    def _clear_memory(self):
        """Aggressively clear GPU and system memory."""
        for _ in range(3):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Scope Benchmark")
    parser.add_argument("--pipelines", nargs="+", help="Specific pipelines to test")
    parser.add_argument("--resolutions", nargs="+", help="Resolutions (e.g. 512x512)")
    parser.add_argument("--iterations", type=int, default=30, help="Measurement iterations per config")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per config")
    parser.add_argument("--output", default=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 (enabled by default)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()

    if not args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 Enabled")

    custom_res = []
    if args.resolutions:
        for r in args.resolutions:
            try:
                h, w = map(int, r.split("x"))
                custom_res.append((h, w))
            except ValueError: pass

    hw = HardwareInfo()
    print("\n=== Hardware ===")
    print(f"GPU: {hw._get_gpu_info().get('devices', [{}])[0].get('name', 'None')}")
    print(f"VRAM: {hw.get_primary_gpu_vram_gb():.1f} GB")

    matrix = ConfigurationMatrix(
        pipelines=args.pipelines,
        resolutions=custom_res,
    ).build()

    print(f"\nPlanned Configurations: {len(matrix)}")
    if not matrix: return

    runner = BenchmarkRunner(args.warmup, args.iterations, compile_model=args.compile)
    results = []

    try:
        for i, config in enumerate(matrix, 1):
            print(f"\n[{i}/{len(matrix)}]", end=" ")
            metrics = runner.run_config(config)
            results.append({
                "pipeline": config["pipeline_id"],
                "resolution": f"{config['height']}x{config['width']}",
                "metrics": metrics
            })
    except KeyboardInterrupt:
        print("\nStopped.")

    data = {
        "metadata": {"timestamp": datetime.now().isoformat(), "args": vars(args)},
        "hardware": hw.to_dict(),
        "results": results
    }
    with open(args.output, "w") as f: json.dump(data, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
