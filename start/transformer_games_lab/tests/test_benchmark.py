from __future__ import annotations

import unittest

from gamebench.systems.benchmark import BenchmarkConfig, run_benchmark


class CounterWorkload:
    name = "counter"

    def __init__(self) -> None:
        self.value = 0

    def run(self) -> None:
        self.value += 1

    def synchronize(self) -> None:
        pass

    def metadata(self) -> dict[str, str]:
        return {"kind": "test"}

    def work_items_per_call(self) -> int:
        return 2

    def estimated_flops_per_call(self) -> float:
        return 4.0

    def estimated_bytes_per_call(self) -> float:
        return 8.0


class BenchmarkTests(unittest.TestCase):
    def test_rejects_invalid_memory_sampling_duration(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(memory_seconds=0)

    def test_result_has_units_and_estimate_labels(self) -> None:
        result = run_benchmark(CounterWorkload(), BenchmarkConfig(
            warmup_calls=1,
            min_samples=3,
            max_samples=4,
            target_sample_ms=0.1,
            min_total_seconds=0.0001,
            target_robust_cv=1.0,
            memory_seconds=0.001,
        ))
        self.assertGreater(result.latency_ms["median"], 0)
        self.assertGreater(result.throughput_items_per_second, 0)
        self.assertEqual(result.estimates["arithmetic_intensity_flops_per_byte"], 0.5)
        self.assertIn("not hardware counters", result.estimates["note"])


if __name__ == "__main__":
    unittest.main()
