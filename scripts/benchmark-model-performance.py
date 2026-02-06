#!/usr/bin/env python3
"""Benchmark VLA model inference performance.

Measures:
- Inference latency (mean, p50, p95, p99)
- GPU memory usage (allocated, reserved, peak)
- Throughput (samples/second)

Targets:
- Latency: <100ms per sample (batch_size=1)
- Memory: <24GB GPU memory (batch_size=32)

Usage:
    # CPU benchmark
    python scripts/benchmark-model-performance.py --device cpu

    # GPU benchmark
    python scripts/benchmark-model-performance.py --device cuda --batch-size 1

    # Throughput test
    python scripts/benchmark-model-performance.py --device cuda --batch-size 32 --iterations 50

    # Save results
    python scripts/benchmark-model-performance.py --device cuda --output results.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch

from vla.models import VLAConfig, VLAModel
from vla.utils import setup_logger

logger = setup_logger(__name__)


def benchmark_latency(
    model: VLAModel,
    batch_size: int = 1,
    num_iterations: int = 100,
    num_warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure inference latency with statistics.

    Args:
        model: VLA model to benchmark
        batch_size: Batch size for inference
        num_iterations: Number of iterations to measure
        num_warmup: Number of warmup iterations (not measured)
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary with latency statistics (mean_ms, std_ms, p50_ms, p95_ms, p99_ms)
    """
    logger.info(
        f"Benchmarking latency: batch_size={batch_size}, "
        f"iterations={num_iterations}, warmup={num_warmup}"
    )

    model.eval()
    device_obj = torch.device(device)

    # Prepare dummy inputs
    images = torch.rand(batch_size, 3, 224, 224, device=device_obj)
    texts = [f"instruction {i}" for i in range(batch_size)]

    # Warmup runs
    logger.info("Running warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.predict(images, texts)
            if device == "cuda":
                torch.cuda.synchronize()

    # Benchmark runs
    logger.info("Running benchmark iterations...")
    latencies_ms = []

    with torch.no_grad():
        for i in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model.predict(images, texts)

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies_ms.append(latency_ms)

            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_iterations} iterations")

    # Calculate statistics
    latencies_ms.sort()
    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p50_ms = latencies_ms[int(len(latencies_ms) * 0.50)]
    p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99_ms = latencies_ms[int(len(latencies_ms) * 0.99)]

    results = {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
    }

    logger.info(f"Latency Results: mean={mean_ms:.2f}ms, p95={p95_ms:.2f}ms, p99={p99_ms:.2f}ms")
    return results


def benchmark_memory(
    model: VLAModel,
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure GPU memory usage.

    Args:
        model: VLA model to benchmark
        batch_size: Batch size for inference
        device: Device to run on (must be 'cuda')

    Returns:
        Dictionary with memory statistics (allocated_mb, reserved_mb, peak_mb)
    """
    if device != "cuda":
        logger.warning("Memory benchmark only supported on CUDA devices")
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}

    logger.info(f"Benchmarking GPU memory: batch_size={batch_size}")

    model.eval()
    device_obj = torch.device(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure initial memory
    initial_allocated = torch.cuda.memory_allocated() / 1024**2

    # Prepare inputs
    images = torch.rand(batch_size, 3, 224, 224, device=device_obj)
    texts = [f"instruction {i}" for i in range(batch_size)]

    # Forward pass
    with torch.no_grad():
        _ = model.predict(images, texts)
        torch.cuda.synchronize()

    # Measure memory
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    results = {
        "allocated_mb": allocated_mb,
        "reserved_mb": reserved_mb,
        "peak_mb": peak_mb,
        "model_mb": allocated_mb - initial_allocated,
    }

    logger.info(
        f"Memory Results: allocated={allocated_mb:.1f}MB, "
        f"peak={peak_mb:.1f}MB, model={results['model_mb']:.1f}MB"
    )
    return results


def benchmark_throughput(
    model: VLAModel,
    batch_size: int = 32,
    num_iterations: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure inference throughput (samples/second).

    Args:
        model: VLA model to benchmark
        batch_size: Batch size for inference
        num_iterations: Number of iterations to measure
        device: Device to run on

    Returns:
        Dictionary with throughput statistics
    """
    logger.info(f"Benchmarking throughput: batch_size={batch_size}, iterations={num_iterations}")

    model.eval()
    device_obj = torch.device(device)

    # Prepare inputs
    images = torch.rand(batch_size, 3, 224, 224, device=device_obj)
    texts = [f"instruction {i}" for i in range(batch_size)]

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model.predict(images, texts)
            if device == "cuda":
                torch.cuda.synchronize()

    # Measure throughput
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model.predict(images, texts)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_samples = batch_size * num_iterations
    throughput = total_samples / total_time

    results = {
        "throughput_samples_per_sec": throughput,
        "total_samples": total_samples,
        "total_time_sec": total_time,
        "batch_size": batch_size,
    }

    logger.info(f"Throughput: {throughput:.1f} samples/sec")
    return results


def print_benchmark_report(
    latency_results: Dict[str, float],
    memory_results: Dict[str, float],
    throughput_results: Dict[str, float],
    device: str,
    batch_size: int,
):
    """Print formatted benchmark report.

    Args:
        latency_results: Latency benchmark results
        memory_results: Memory benchmark results
        throughput_results: Throughput benchmark results
        device: Device used for benchmarking
        batch_size: Batch size used
    """
    print("\n" + "=" * 70)
    print("VLA Model Performance Benchmark Report")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print()

    print("Latency (ms):")
    print(f"  Mean:    {latency_results['mean_ms']:.2f} ms")
    print(f"  Std Dev: {latency_results['std_ms']:.2f} ms")
    print(f"  P50:     {latency_results['p50_ms']:.2f} ms")
    print(f"  P95:     {latency_results['p95_ms']:.2f} ms")
    print(f"  P99:     {latency_results['p99_ms']:.2f} ms")
    print(f"  Min:     {latency_results['min_ms']:.2f} ms")
    print(f"  Max:     {latency_results['max_ms']:.2f} ms")

    # Check latency target
    target_latency_ms = 100
    if batch_size == 1:
        if latency_results['mean_ms'] < target_latency_ms:
            print(f"  ✓ PASS: Mean latency < {target_latency_ms}ms target")
        else:
            print(f"  ✗ FAIL: Mean latency exceeds {target_latency_ms}ms target")

    print()

    if device == "cuda" and memory_results["peak_mb"] > 0:
        print("GPU Memory (MB):")
        print(f"  Allocated: {memory_results['allocated_mb']:.1f} MB")
        print(f"  Reserved:  {memory_results['reserved_mb']:.1f} MB")
        print(f"  Peak:      {memory_results['peak_mb']:.1f} MB")
        print(f"  Model:     {memory_results['model_mb']:.1f} MB")

        # Check memory target
        target_memory_gb = 24
        if memory_results['peak_mb'] / 1024 < target_memory_gb:
            print(f"  ✓ PASS: Peak memory < {target_memory_gb}GB target")
        else:
            print(f"  ✗ FAIL: Peak memory exceeds {target_memory_gb}GB target")
        print()

    print("Throughput:")
    print(f"  Samples/sec: {throughput_results['throughput_samples_per_sec']:.1f}")
    print(f"  Total time:  {throughput_results['total_time_sec']:.2f} sec")
    print(f"  Total samples: {throughput_results['total_samples']}")
    print()

    print("=" * 70)


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark VLA model performance")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmark on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for latency benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput benchmark",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Create model
    logger.info("Creating VLA model...")
    config = VLAConfig()
    model = VLAModel(config)
    model = model.to(args.device)
    model.eval()

    logger.info(f"Model created on device: {args.device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Run benchmarks
    latency_results = benchmark_latency(
        model,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        num_warmup=args.warmup,
        device=args.device,
    )

    memory_results = benchmark_memory(
        model,
        batch_size=args.batch_size,
        device=args.device,
    )

    if not args.skip_throughput:
        throughput_results = benchmark_throughput(
            model,
            batch_size=max(args.batch_size, 16),  # Use at least batch_size=16
            num_iterations=50,
            device=args.device,
        )
    else:
        throughput_results = {
            "throughput_samples_per_sec": 0,
            "total_samples": 0,
            "total_time_sec": 0,
            "batch_size": 0,
        }

    # Print report
    print_benchmark_report(
        latency_results,
        memory_results,
        throughput_results,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        results = {
            "device": args.device,
            "batch_size": args.batch_size,
            "latency": latency_results,
            "memory": memory_results,
            "throughput": throughput_results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
