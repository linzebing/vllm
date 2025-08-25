# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Micro benchmark comparing sha256, sha256_cbor and Python built-in hash function.
Tests only a tuple containing: (32-byte bytes object, 32-int tuple)
"""

import random
import statistics
import time

from vllm.utils import sha256, sha256_cbor


class HashBenchmark:
    def __init__(self):
        self.test_data = self.generate_test_data()

    def generate_test_data(self) -> tuple[bytes, tuple[int, ...]]:
        """Generate test data: tuple with 32-byte bytes object and 32-int tuple."""
        random.seed(42)  # For reproducible results

        # 32-byte bytes object
        bytes_data = bytes(random.getrandbits(8) for _ in range(32))

        # tuple of 32 integers
        int_tuple = tuple(random.randint(1, 1000000) for _ in range(32))

        return (bytes_data, int_tuple)

    def builtin_hash(self, data: tuple) -> int:
        """Built-in hash function wrapper."""
        return hash(data)

    def benchmark_function(self, func, iterations: int = 10000) -> tuple[float, float]:
        """Benchmark a hash function with test data."""
        times = []

        # Warm-up runs
        for _ in range(100):
            func(self.test_data)

        # Actual benchmark runs
        for _ in range(iterations):
            start = time.perf_counter()
            func(self.test_data)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        return avg_time, std_dev

    def run_benchmark(self, iterations: int = 10000):
        """Run the benchmark comparison."""
        print("=" * 60)
        print("HASH FUNCTION PERFORMANCE BENCHMARK")
        print("=" * 60)
        print("Test data: (32-byte bytes object, 32-int tuple)")
        print(f"Iterations: {iterations:,}")
        print("=" * 60)

        # Benchmark SHA256
        sha256_time, sha256_std = self.benchmark_function(sha256, iterations)

        # Benchmark SHA256
        sha256_cbor_time, sha256_cbor_std = self.benchmark_function(
            sha256_cbor, iterations
        )

        # Benchmark built-in hash
        builtin_time, builtin_std = self.benchmark_function(
            self.builtin_hash, iterations
        )

        # Speed ratio
        sha256_speed_ratio = sha256_time / builtin_time
        sha256_cbor_speed_ratio = sha256_cbor_time / builtin_time

        print("\nResults:")
        print(f"  SHA256: {sha256_time * 1e6:8.2f} ± {sha256_std * 1e6:6.2f} μs")
        print(
            f"  SHA256_CBOR: {sha256_cbor_time * 1e6:8.2f} ± "
            f"{sha256_cbor_std * 1e6:6.2f} μs"
        )
        print(
            f"  Built-in hash: {builtin_time * 1e6:8.2f} ± {builtin_std * 1e6:6.2f} μs"
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"• Built-in hash() is {sha256_speed_ratio:.1f}x faster than SHA256")
        print(
            f"• Built-in hash() is {sha256_cbor_speed_ratio:.1f}x faster than "
            "SHA256_CBOR"
        )
        print(
            f"• SHA256: {sha256_time * 1e6:.1f}μs per hash (cryptographically secure)"
        )
        print(
            f"• SHA256_CBOR: {sha256_cbor_time * 1e6:.1f}μs per hash "
            "(cryptographically secure, cross-language compatible)"
        )
        print(f"• Built-in: {builtin_time * 1e6:.1f}μs per hash (fast, not secure)")


def main():
    """Run the simplified benchmark."""
    benchmark = HashBenchmark()
    benchmark.run_benchmark(iterations=10000)


if __name__ == "__main__":
    main()
