"""ConfoRL Benchmarking Suite.

Comprehensive benchmarking framework for evaluating ConfoRL against
state-of-the-art baselines in safe reinforcement learning.

Key Components:
- Standardized safety-critical environments
- Baseline algorithm implementations
- Statistical significance testing
- Publication-ready result analysis

Usage:
    from conforl.benchmarks import run_quick_benchmark, create_comprehensive_benchmark
    
    # Quick evaluation
    results = run_quick_benchmark()
    
    # Full research benchmark
    suite = create_comprehensive_benchmark()
    runner = BenchmarkRunner()
    results = runner.run_benchmark_suite(suite)

Author: ConfoRL Research Team  
License: Apache 2.0
"""

from .framework import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkSuite,
    SafetyEnvironment,
    CartPoleSafety,
    BaselineAlgorithm,
    RandomPolicy,
    ConstrainedPolicy,
    ConfoRLBenchmarkWrapper,
    create_quick_benchmark,
    create_comprehensive_benchmark,
    run_quick_benchmark
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult", 
    "BenchmarkSuite",
    "SafetyEnvironment",
    "CartPoleSafety",
    "BaselineAlgorithm",
    "RandomPolicy",
    "ConstrainedPolicy",
    "ConfoRLBenchmarkWrapper",
    "create_quick_benchmark",
    "create_comprehensive_benchmark",
    "run_quick_benchmark"
]

__version__ = "0.1.0"