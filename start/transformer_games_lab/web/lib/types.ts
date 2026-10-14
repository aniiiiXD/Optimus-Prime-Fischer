export type Hardware = {
  architecture?: string;
  cpu_name?: string;
  logical_cpus?: number;
  simd?: Record<string, boolean>;
  matrix?: Record<string, boolean>;
  library_backend?: Record<string, string | number | boolean | null>;
};

export type BenchmarkResult = {
  name: string;
  workload: Record<string, string | number | boolean | null>;
  environment?: Record<string, unknown>;
  latency_ms: Record<string, number>;
  variability?: Record<string, number | boolean>;
  throughput_items_per_second?: number;
  memory?: Record<string, number | string | null>;
  estimates: Record<string, number | string | null>;
};

export type BenchmarkReport = {
  schema_version?: number;
  hardware?: Hardware;
  simd_speedup?: number;
  result_relative_difference?: number;
  maximum_relative_error_vs_float64_reference?: number;
  results: BenchmarkResult[];
  source?: string;
};

export type ReportBundle = {
  reports: BenchmarkReport[];
  missing: string[];
};
