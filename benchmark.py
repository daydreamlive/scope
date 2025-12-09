#!/usr/bin/env python3
"""
Script to run test.py for each pipeline and collect average FPS statistics.
Results are stored in a Markdown file.
"""

import subprocess
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

try:
    import torch
except ImportError:
    torch = None

# Define pipelines with their test.py paths
PIPELINES = {
    "longlive": "pipelines/longlive/test.py",
    "krea_realtime_video": "pipelines/krea_realtime_video/test.py",
    "streamdiffusionv2": "pipelines/streamdiffusionv2/test.py",
}

OUTPUT_FILE = "BENCHMARK.md"


def get_gpu_name() -> str:
    """
    Get the GPU name using PyTorch or nvidia-smi.
    Returns 'Unknown' if GPU info cannot be retrieved.
    """
    if torch is not None and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            pass

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    return "Unknown"


def extract_avg_fps(output: str) -> Optional[float]:
    """
    Extract average FPS from test.py output.
    Looks for line like: "FPS - Avg: 12.34, Max: 15.67, Min: 10.12"
    """
    # Pattern to match "FPS - Avg: X.XX"
    pattern = r"FPS - Avg:\s+([\d.]+)"
    match = re.search(pattern, output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def run_test(pipeline_name: str, test_path: str) -> Optional[float]:
    """
    Run test.py for a pipeline and return the average FPS.
    Returns None if the test fails or FPS cannot be extracted.
    """
    print(f"\n{'='*60}")
    print(f"Running test for pipeline: {pipeline_name}")
    print(f"Test file: {test_path}")
    print(f"{'='*60}\n")

    test_file = Path(test_path)
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_path}")
        return None

    try:
        # Run the test.py file as a module to handle relative imports correctly
        # The test files use relative imports (e.g., from .pipeline import ...)
        test_module = test_file.stem  # 'test' without .py

        # Run using python -m to handle imports correctly
        # Need to run from project root so imports work
        result = subprocess.run(
            [sys.executable, "-m", f"pipelines.{pipeline_name}.{test_module}"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        # Print stdout and stderr for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"ERROR: Test failed with return code {result.returncode}")
            return None

        # Extract average FPS from output
        avg_fps = extract_avg_fps(result.stdout)
        if avg_fps is None:
            # Try stderr as well
            avg_fps = extract_avg_fps(result.stderr)

        if avg_fps is not None:
            print(f"\n✓ Successfully extracted average FPS: {avg_fps:.2f}")
        else:
            print(f"\n⚠ Warning: Could not extract average FPS from output")

        return avg_fps

    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after 1 hour")
        return None
    except Exception as e:
        print(f"ERROR: Exception while running test: {e}")
        return None


def write_results_to_markdown(results: Dict[str, Optional[float]], output_file: str, gpu_name: str):
    """
    Write FPS results to a Markdown file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, "w") as f:
        f.write("# Pipeline FPS Benchmark Results\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"**GPU:** {gpu_name}\n\n")
        f.write("---\n\n")
        f.write("## Results\n\n")
        f.write("| Pipeline | Average FPS | Status |\n")
        f.write("|----------|-------------|--------|\n")

        for pipeline_name, avg_fps in sorted(results.items()):
            if avg_fps is not None:
                f.write(f"| {pipeline_name} | {avg_fps:.2f} | ✓ Success |\n")
            else:
                f.write(f"| {pipeline_name} | N/A | ✗ Failed/No Data |\n")

    print(f"\n{'='*60}")
    print(f"Results written to: {output_file}")
    print(f"{'='*60}\n")


def main():
    """
    Main function to run all pipeline tests and collect results.
    """
    print("Pipeline FPS Benchmark Runner")
    print("=" * 60)
    print(f"Testing {len(PIPELINES)} pipelines...")

    # Get GPU name
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    results = {}

    for pipeline_name, test_path in PIPELINES.items():
        avg_fps = run_test(pipeline_name, test_path)
        results[pipeline_name] = avg_fps

    # Write results to Markdown file
    write_results_to_markdown(results, OUTPUT_FILE, gpu_name)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for pipeline_name, avg_fps in sorted(results.items()):
        status = f"{avg_fps:.2f} FPS" if avg_fps is not None else "Failed/No Data"
        print(f"  {pipeline_name:30s}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
