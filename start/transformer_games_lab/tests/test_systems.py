from __future__ import annotations

import unittest

from gamebench.systems.hardware import cpu_capabilities
from gamebench.systems.matrix_workload import MatrixMultiplyWorkload, MatrixWorkloadConfig


class SystemsTests(unittest.TestCase):
    def test_hardware_report_has_stable_sections(self) -> None:
        report = cpu_capabilities()
        self.assertIn("architecture", report)
        self.assertIn("simd", report)
        self.assertIn("matrix", report)
        self.assertIn("library_backend", report)

    def test_matrix_workload_accounting_and_execution(self) -> None:
        workload = MatrixMultiplyWorkload(MatrixWorkloadConfig(m=4, k=5, n=6))
        workload.run()
        self.assertEqual(workload.estimated_flops_per_call(), 240.0)
        self.assertEqual(workload.estimated_bytes_per_call(), 296.0)


if __name__ == "__main__":
    unittest.main()
