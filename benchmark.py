#!/usr/bin/env python3
"""
Simple Benchmarking Script for Scope Pipelines.

Usage:
    uv run benchmark.py [options]
"""

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

import psutil
import torch
from omegaconf import OmegaConf

# Optional dependencies
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# Scope imports
from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.registry import PipelineRegistry
from scope.core.pipelines.utils import Quantization


# =================================================================================================
# HARDWARE INFO
# =================================================================================================

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

        if PYNVML_AVAILABLE:
            try:
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
            except Exception:
                pass

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


# =================================================================================================
# RESOURCE MONITOR
# =================================================================================================

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

        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._pynvml_initialized = True
            except Exception:
                pass

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
            try:
                sample["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(self.device_index) / (1024**3)
                if self._pynvml_initialized and self._gpu_handle:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    sample["gpu_utilization_percent"] = util.gpu
            except Exception:
                pass

        try:
            sample["system_cpu_percent"] = psutil.cpu_percent()
        except Exception:
            pass
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
        return stats

    def cleanup(self):
        self.stop()
        if self._pynvml_initialized:
            try: pynvml.nvmlShutdown()
            except Exception: pass


# =================================================================================================
# CONFIGURATION MATRIX
# =================================================================================================

class ConfigurationMatrix:
    # Default resolutions to test
    STANDARD_RESOLUTIONS = [
        (320, 576),
        (480, 832),
        (512, 512),
        (576, 1024),
        (768, 1344),
    ]

    # Defaults (Single run per resolution)
    DEFAULT_PROMPT = "A realistic video of a serene landscape with rolling hills, a clear blue sky, and a gentle stream."

    PIPELINE_CONSTRAINTS = {
        "krea_realtime_video": {
            "min_vram_gb": 32,
            "high_res_vram_gb": 40,
            "high_res_threshold": (480, 832),
        },
    }

    def __init__(self, hardware_vram_gb: float, pipelines=None, resolutions=None, steps=None):
        self.hardware_vram_gb = hardware_vram_gb
        self.selected_pipelines = pipelines
        self.custom_resolutions = resolutions
        self.steps = steps or [4] # Default to 4 if not specified

    def build(self) -> list[dict]:
        all_pipelines = PipelineRegistry.list_pipelines()

        if self.selected_pipelines:
            pipelines = [p for p in all_pipelines if p in self.selected_pipelines]
        else:
            pipelines = [p for p in all_pipelines if p != "passthrough"]

        configurations = []
        for pid in pipelines:
            if not self._check_constraints(pid):
                print(f"Skipping {pid}: insufficient VRAM ({self.hardware_vram_gb:.1f}GB)")
                continue

            # Determine resolutions
            resolutions = self._get_resolutions(pid)

            for h, w in resolutions:
                config = {
                    "pipeline_id": pid,
                    "height": h,
                    "width": w,
                    "denoising_steps": self.steps,
                    "prompt": self.DEFAULT_PROMPT,
                }
                configurations.append(config)

        return configurations

    def _check_constraints(self, pid: str) -> bool:
        constraints = self.PIPELINE_CONSTRAINTS.get(pid, {})
        return self.hardware_vram_gb >= constraints.get("min_vram_gb", 0)

    def _get_resolutions(self, pid: str) -> list[tuple[int, int]]:
        if self.custom_resolutions:
            return self.custom_resolutions

        # Default config for the pipeline
        pipeline_class = PipelineRegistry.get(pid)
        if not pipeline_class: return []
        default_cfg = pipeline_class.get_config_class()()

        # Start with default resolution
        res_set = {(default_cfg.height, default_cfg.width)}

        # Add standard ones that fit VRAM constraints
        constraints = self.PIPELINE_CONSTRAINTS.get(pid, {})
        high_res_vram = constraints.get("high_res_vram_gb")
        threshold = constraints.get("high_res_threshold")

        for h, w in self.STANDARD_RESOLUTIONS:
            if high_res_vram and threshold:
                th_h, th_w = threshold
                if (h > th_h or w > th_w) and self.hardware_vram_gb < high_res_vram:
                    continue
            res_set.add((h, w))

        return sorted(list(res_set))


# =================================================================================================
# BENCHMARK RUNNER
# =================================================================================================

class BenchmarkRunner:
    def __init__(self, warmup_iterations=2, iterations=5, compile_model=False):
        self.warmup_iterations = warmup_iterations
        self.iterations = iterations
        self.compile_model = compile_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_config(self, config: dict) -> dict:
        pipeline_id = config["pipeline_id"]
        print(f"\n--- Benchmarking {pipeline_id} [{config['height']}x{config['width']}] ---")

        pipeline = None
        try:
            pipeline = self._init_pipeline(config)
            inputs = {"prompts": [{"text": config["prompt"], "weight": 100}]}

            # Warmup Phase
            if self.warmup_iterations > 0:
                print(f"Warmup ({self.warmup_iterations} iterations)...")
                for _ in range(self.warmup_iterations):
                    pipeline(**inputs)
                self._clear_memory()

            # Measurement Phase
            print(f"Measuring ({self.iterations} iterations)...")
            monitor = ResourceMonitor()
            latencies = []
            frame_counts = []

            monitor.start()
            for _ in range(self.iterations):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                output = pipeline(**inputs)
                latencies.append(time.time() - t0)

                # Check output for frame count (batch size)
                # Some pipelines return a tensor (T, C, H, W) or (B, T, C, H, W)
                # If it's 4D (T, C, H, W), dim 0 is frames.
                # If it's 5D (B, T, C, H, W), dim 1 is frames * batch size.
                current_frames = 1
                if hasattr(output, "shape") and len(output.shape) >= 1:
                    current_frames = output.shape[0]
                frame_counts.append(current_frames)

            monitor.stop()
            resource_stats = monitor.get_statistics()
            monitor.cleanup()

            # Metrics Calculation
            if not latencies:
                return {"error": "No successful iterations"}

            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

            # Calculate FPS based on frames generated per call
            avg_frames_per_call = statistics.mean(frame_counts) if frame_counts else 1.0

            fps_avg = avg_frames_per_call / avg_latency if avg_latency > 0 else 0
            fps_min = avg_frames_per_call / max_latency if max_latency > 0 else 0
            fps_max = avg_frames_per_call / min_latency if min_latency > 0 else 0

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

    def _init_pipeline(self, config: dict):
        pid = config["pipeline_id"]
        pipeline_class = PipelineRegistry.get(pid)
        if not pipeline_class: raise ValueError(f"Unknown pipeline: {pid}")

        # Path Logic
        model_dir = Path("src/scope/core/pipelines") / pid
        if not model_dir.exists(): # Handle running from src vs root
             model_dir = Path(__file__).parent / "src/scope/core/pipelines" / pid

        model_config = OmegaConf.load(model_dir / "model.yaml")
        pipeline_config = {
            "model_dir": str(get_models_dir()),
            "model_config": model_config,
            "height": config["height"],
            "width": config["width"],
            "denoising_steps": config["denoising_steps"],
        }

        # Hardcoded paths matching original test scripts
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

        # Init
        quantization = Quantization.FP8_E4M3FN if pid == "krea_realtime_video" else None
        args = {
            "config": OmegaConf.create(pipeline_config),
            "device": self.device,
            "dtype": torch.bfloat16
        }
        if quantization:
            args.update({"quantization": quantization})

        # Add compile flag if pipeline accepts it (most new ones do)
        # Note: Some pipelines might not have 'compile' arg in __init__, but Krea does.
        # We can inspect or try/except, but for simplicity we assume consistency or pass it conditionally
        if pid == "krea_realtime_video":
            args["compile"] = self.compile_model
        # For others, if they support compile, add logic here.
        # StreamDiffusionV2 might not expose it in __init__?
        # If it inherits from BasePipeline that has it?
        # We'll leave it out for others unless we know they support it to avoid TypeError.

        return pipeline_class(**args)

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


# =================================================================================================
# MAIN
# =================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Scope Benchmark")
    parser.add_argument("--pipelines", nargs="+", help="Specific pipelines to test")
    parser.add_argument("--resolutions", nargs="+", help="Resolutions (e.g. 512x512)")
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps (default: 4)")
    parser.add_argument("--iterations", type=int, default=100, help="Measurement iterations per config")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per config")
    parser.add_argument("--output", default=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 (enabled by default)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()

    # Global Torch Settings
    if not args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 Enabled")

    # Parse resolutions
    custom_res = []
    if args.resolutions:
        for r in args.resolutions:
            try:
                h, w = map(int, r.split("x"))
                custom_res.append((h, w))
            except ValueError: pass

    # Detect Hardware
    hw = HardwareInfo()
    print("\n=== Hardware ===")
    print(f"GPU: {hw._get_gpu_info().get('devices', [{}])[0].get('name', 'None')}")
    print(f"VRAM: {hw.get_primary_gpu_vram_gb():.1f} GB")

    # Build Configurations (1 per resolution)
    matrix = ConfigurationMatrix(
        hw.get_primary_gpu_vram_gb(),
        pipelines=args.pipelines,
        resolutions=custom_res,
        steps=[args.steps]
    ).build()

    print(f"\nPlanned Configurations: {len(matrix)}")
    if not matrix: return

    # Run
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

    # Save
    data = {
        "metadata": {"timestamp": datetime.now().isoformat(), "args": vars(args)},
        "hardware": hw.to_dict(),
        "results": results
    }
    with open(args.output, "w") as f: json.dump(data, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
