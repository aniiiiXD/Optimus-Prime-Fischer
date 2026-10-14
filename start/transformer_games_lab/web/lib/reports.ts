import { readFile } from "node:fs/promises";
import path from "node:path";

import type { BenchmarkReport, ReportBundle } from "@/lib/types";

const REPORT_NAMES = [
  "m5_simd_baseline.json",
  "m5_matrix_float32.json",
  "m5_matrix_bfloat16.json",
  "m5_transformer_micro.json",
];

async function readReport(name: string): Promise<BenchmarkReport | null> {
  const reportPath = path.resolve(process.cwd(), "..", "reports", name);
  try {
    const payload = JSON.parse(await readFile(reportPath, "utf8")) as BenchmarkReport;
    return { ...payload, source: name };
  } catch {
    return null;
  }
}

export async function loadReports(): Promise<ReportBundle> {
  const loaded = await Promise.all(REPORT_NAMES.map(readReport));
  return {
    reports: loaded.filter((report): report is BenchmarkReport => report !== null),
    missing: REPORT_NAMES.filter((_, index) => loaded[index] === null),
  };
}
