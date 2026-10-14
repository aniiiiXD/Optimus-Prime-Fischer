"use client";

import { ChangeEvent, useMemo, useState } from "react";

import { LineChart, RooflineChart } from "@/components/Charts";
import type { BenchmarkReport, BenchmarkResult, Hardware } from "@/lib/types";

type Section = "overview" | "simd" | "matrix" | "transformer" | "roofline" | "method";
type Insight = "instruction" | "kernel" | "operator" | "model" | "roofline" | "method";

const sections: { id: Section; title: string; detail: string }[] = [
  { id: "overview", title: "Overview", detail: "system map" },
  { id: "simd", title: "SIMD", detail: "scalar ↔ vector" },
  { id: "matrix", title: "Matrix", detail: "shape · dtype" },
  { id: "transformer", title: "Transformer", detail: "forward · train" },
  { id: "roofline", title: "Roofline", detail: "memory ↔ compute" },
  { id: "method", title: "Method", detail: "trust the numbers" },
];

const format = (value: number, digits = 2) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: digits });
const number = (value: unknown) => Number(value) || 0;

function buildData(reports: BenchmarkReport[]) {
  const results = reports.flatMap((report) => report.results || []);
  return {
    hardware: reports.find((report) => report.hardware)?.hardware || null,
    simd: reports.find((report) => report.simd_speedup !== undefined) || null,
    matrix: results.filter((result) => result.name.startsWith("matmul/")),
    transformer: results.filter((result) => result.name.startsWith("tiny_transformer/")),
  };
}

function bestMatrix(rows: BenchmarkResult[]) {
  return rows.reduce<BenchmarkResult | null>((best, row) =>
    number(row.estimates.achieved_gflops_per_second) > number(best?.estimates.achieved_gflops_per_second) ? row : best, null);
}

function referenceForward(rows: BenchmarkResult[]) {
  return rows.filter((row) => row.workload.mode === "forward").sort((a, b) =>
    number(a.workload.batch_size) - number(b.workload.batch_size)
      || number(a.workload.sequence_length) - number(b.workload.sequence_length)
      || number(a.workload.requested_torch_threads) - number(b.workload.requested_torch_threads))[0] || null;
}

function MachineStrip({ hardware }: { hardware: Hardware | null }) {
  const backend = hardware?.library_backend || {};
  const items = [
    ["CPU", hardware?.cpu_name || "unknown"],
    ["ISA", hardware?.architecture?.toUpperCase() || "unknown"],
    ["SIMD", hardware?.simd?.neon_128 ? "NEON 128" : "not reported"],
    ["Matrix", hardware?.matrix?.arm_sme2 ? "SME2" : "not reported"],
    ["BLAS", backend.accelerate ? "Accelerate" : "other"],
    ["PyTorch", String(backend.torch || "unknown")],
  ];
  return <div className="machine-strip" aria-label="Detected machine">{items.map(([label, value]) => <span className="machine-item" key={label}>{label} <b>{value}</b></span>)}</div>;
}

function Inspector({ insight, simdSpeedup, matrixPeak, forwardLatency }: {
  insight: Insight;
  simdSpeedup: number;
  matrixPeak: number;
  forwardLatency: number;
}) {
  const content: Record<Insight, { title: string; body: string; formula: string; result?: string }> = {
    instruction: { title: "NEON executes four float32 lanes together.", body: "A 128-bit register holds four 32-bit numbers. One vector fused multiply-add updates four independent products.", formula: "v0[0:4] += v1[0:4] × v2[0:4]", result: `Measured here: ${format(simdSpeedup)}× scalar-to-vector speedup.` },
    kernel: { title: "GEMM reorganizes work for reuse.", body: "A good matrix kernel keeps tiles in registers and cache, reusing each loaded value across many multiply-adds.", formula: "C[M,N] = A[M,K] × B[K,N]", result: `Best measured FP32 rate: ${format(matrixPeak, 0)} GFLOP/s.` },
    operator: { title: "Attention contains two matrix products.", body: "QKᵀ produces token-to-token scores. Multiplying those probabilities by V mixes information from earlier moves.", formula: "softmax(QKᵀ / √d) × V", result: "Softmax and normalization sit between the matrix kernels." },
    model: { title: "The policy is the complete path.", body: "The player pays for embeddings, attention, MLPs, normalization, masking and the policy head—not an isolated GEMM.", formula: "history tokens → legal-action logits", result: `Reference forward median: ${format(forwardLatency, 3)} ms.` },
    roofline: { title: "The roof is a limit, not a target.", body: "Move right by doing more arithmetic per byte. Move up by executing useful FLOPs faster. A point far below both roofs loses time elsewhere.", formula: "performance ≤ min(bandwidth × intensity, peak)", result: "The plotted bytes are analytical estimates, not traffic counters." },
    method: { title: "Trust distributions, not one reading.", body: "Median represents the common run. p99 exposes slow tails. Robust CV tells us whether repeated samples cluster tightly enough to compare.", formula: "robust CV = 1.4826 × MAD / median", result: "Rerun any result marked noisy before changing the model." },
  };
  const item = content[insight];
  return <aside className="inspector"><p className="inspector-label">Interpretation</p><h2>{item.title}</h2><p>{item.body}</p><code>{item.formula}</code><p className="insight-result">{item.result}</p><div className="inspector-rule" /><p className="source-note">Source: local GameBench JSON. No synthetic values are presented as measurements.</p></aside>;
}

function BarRows({ rows, metric }: { rows: BenchmarkResult[]; metric: "gflops" | "throughput" }) {
  const values = rows.map((row) => metric === "gflops" ? number(row.estimates.achieved_gflops_per_second) : number(row.throughput_items_per_second));
  const maximum = Math.max(...values, 1);
  return <div className="comparison-stage">{rows.map((row, index) => {
    const kernel = String(row.workload.kernel || "");
    const label = kernel || `batch ${row.workload.batch_size}`;
    return <div className={`bar-row ${kernel === "vector" || row.variability?.stable ? "highlight" : ""}`} key={`${row.name}-${row.workload.requested_torch_threads}`}>
      <div className="bar-label"><strong>{label}</strong><small>{kernel ? (kernel === "vector" ? "4 lanes" : "1 lane") : (row.variability?.stable ? "stable" : "noisy")}</small></div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(1, values[index] / maximum * 100)}%` }} /></div>
      <div className="bar-value"><strong>{format(row.latency_ms.median, 3)} ms</strong><small>{metric === "gflops" ? `${format(values[index])} GFLOP/s` : `${format(values[index], 0)} items/s`}</small></div>
    </div>;
  })}</div>;
}

export default function ProfilerWorkbench({ initialReports, missing }: { initialReports: BenchmarkReport[]; missing: string[] }) {
  const [reports, setReports] = useState(initialReports);
  const [section, setSection] = useState<Section>("overview");
  const [overviewInsight, setOverviewInsight] = useState<Insight>("instruction");
  const [matrixDtype, setMatrixDtype] = useState("float32");
  const [matrixThreads, setMatrixThreads] = useState(1);
  const [transformerMode, setTransformerMode] = useState("forward");
  const [transformerThreads, setTransformerThreads] = useState(1);
  const [transformerSequence, setTransformerSequence] = useState(8);
  const data = useMemo(() => buildData(reports), [reports]);
  const peak = bestMatrix(data.matrix);
  const forward = referenceForward(data.transformer);
  const resultCount = data.matrix.length + data.transformer.length + (data.simd?.results.length || 0);

  const insightBySection: Record<Section, Insight> = {
    overview: overviewInsight,
    simd: "instruction",
    matrix: "kernel",
    transformer: "model",
    roofline: "roofline",
    method: "method",
  };

  async function importReports(event: ChangeEvent<HTMLInputElement>) {
    const files = [...(event.target.files || [])];
    const loaded: BenchmarkReport[] = [];
    for (const file of files) {
      try { loaded.push({ ...(JSON.parse(await file.text()) as BenchmarkReport), source: file.name }); } catch { /* Invalid files are ignored. */ }
    }
    if (loaded.length) setReports((current) => [...current, ...loaded]);
  }

  const dtypes = [...new Set(data.matrix.map((row) => String(row.workload.dtype)))].sort((a, b) => ({ float32: 0, bfloat16: 1, float16: 2 }[a as "float32"] ?? 99) - ({ float32: 0, bfloat16: 1, float16: 2 }[b as "float32"] ?? 99));
  const matrixThreadValues = [...new Set(data.matrix.map((row) => number(row.workload.requested_torch_threads)))].sort((a, b) => a - b);
  const activeDtype = dtypes.includes(matrixDtype) ? matrixDtype : dtypes[0];
  const activeMatrixThreads = matrixThreadValues.includes(matrixThreads) ? matrixThreads : matrixThreadValues[0];
  const matrixRows = data.matrix.filter((row) => row.workload.dtype === activeDtype && number(row.workload.requested_torch_threads) === activeMatrixThreads).sort((a, b) => number(a.workload.m) - number(b.workload.m));

  const modes = [...new Set(data.transformer.map((row) => String(row.workload.mode)))];
  const transformerThreadValues = [...new Set(data.transformer.map((row) => number(row.workload.requested_torch_threads)))].sort((a, b) => a - b);
  const sequences = [...new Set(data.transformer.map((row) => number(row.workload.sequence_length)))].sort((a, b) => a - b);
  const activeMode = modes.includes(transformerMode) ? transformerMode : modes[0];
  const activeTransformerThreads = transformerThreadValues.includes(transformerThreads) ? transformerThreads : transformerThreadValues[0];
  const activeSequence = sequences.includes(transformerSequence) ? transformerSequence : sequences[0];
  const transformerRows = data.transformer.filter((row) => row.workload.mode === activeMode && number(row.workload.requested_torch_threads) === activeTransformerThreads && number(row.workload.sequence_length) === activeSequence).sort((a, b) => number(a.workload.batch_size) - number(b.workload.batch_size));

  const vector = data.simd?.results.find((row) => row.workload.kernel === "vector") || null;
  const bandwidth = vector ? number(vector.estimates.bytes_per_call) / (number(vector.latency_ms.median) / 1000) / 1e9 : 32;
  const roofPoints = [
    ...(vector ? [{ result: vector, label: "SIMD dot" }] : []),
    ...data.matrix.filter((row) => row.workload.dtype === "float32" && number(row.workload.requested_torch_threads) === 1).map((result) => ({ result, label: `GEMM ${result.workload.m}` })),
    ...data.transformer.filter((row) => row.workload.mode === "forward" && number(row.workload.requested_torch_threads) === 1).map((result) => ({ result, label: `TF b${result.workload.batch_size}/t${result.workload.sequence_length}` })),
  ];

  return <>
    <MachineStrip hardware={data.hardware} />
    <div className="workspace">
      <aside className="experiment-rail">
        <p className="rail-title">Experiments</p>
        <nav className="section-nav" aria-label="Profiler sections">{sections.map((item) => <button className={`nav-item ${section === item.id ? "active" : ""}`} key={item.id} onClick={() => setSection(item.id)}><span>{item.title}</span><b>{item.detail}</b></button>)}</nav>
        <div className="import-block"><label htmlFor="report-input">Load benchmark JSON</label><input id="report-input" type="file" accept="application/json,.json" multiple onChange={importReports} /><p>New reports join the local baseline without changing Python code.</p></div>
        <div className="report-count"><span>{resultCount}</span><small>measurements loaded</small></div>
      </aside>

      <main className="analysis-canvas">
        {section === "overview" && <section className="panel active">
          <div className="panel-heading"><div><p className="context-line">Machine → kernel → model</p><h1>Follow one number all the way down.</h1></div><p>Start with the hardware instruction, watch it become a matrix kernel, then see where that kernel appears inside a Transformer.</p></div>
          <div className="compute-path">{([[
            "instruction", "Instruction", "NEON", "4 × float32 per vector"],
            ["kernel", "Kernel", "GEMM", "C = A × B"],
            ["operator", "Operator", "QKᵀ", "attention scores"],
            ["model", "Model", "Policy", "next legal move"],
          ] as [Insight, string, string, string][]).map(([id, label, value, detail], index) => <div className="path-fragment" key={id}><button className={`path-node ${overviewInsight === id ? "selected" : ""}`} onClick={() => setOverviewInsight(id)}><span>{label}</span><strong>{value}</strong><small>{detail}</small></button>{index < 3 && <span className="path-link" />}</div>)}</div>
          <div className="headline-ledger">
            <div><span>native SIMD speedup</span><strong className="signal">{format(data.simd?.simd_speedup || 0)}×</strong></div>
            <div><span>best matrix throughput</span><strong>{format(number(peak?.estimates.achieved_gflops_per_second), 0)} GF/s</strong></div>
            <div><span>reference forward</span><strong>{format(number(forward?.latency_ms.median), 3)} ms</strong></div>
            <div><span>measurements</span><strong>{resultCount}</strong></div>
          </div>
          <div className="overview-grid"><article><b>A</b><div><h2>Vector width is not model speed</h2><p>NEON accelerates suitable loops. Transformer latency also includes memory movement, dispatch, normalization and masking.</p></div></article><article><b>B</b><div><h2>Hardware support is not kernel support</h2><p>The M5 reports BF16 and SME, but the installed library still decides which optimized route actually runs.</p></div></article></div>
        </section>}

        {section === "simd" && <section className="panel active">
          <div className="panel-heading compact"><div><p className="context-line">Native dot product</p><h1>One loop, four lanes.</h1></div><p>The inputs and mathematics stay fixed. Only the implementation changes from scalar float32 to explicit NEON.</p></div>
          <div className="formula-band"><code>sum += a[i] × b[i]</code><span>2 FLOPs / 8 input bytes</span><strong>0.25 FLOP/byte</strong></div>
          {data.simd ? <><BarRows rows={data.simd.results} metric="gflops" /><div className="speedup-stamp"><strong>{format(data.simd.simd_speedup || 0)}×</strong><span>measured vector speedup</span></div></> : <div className="empty-state">No SIMD report loaded.</div>}
          <div className="reading-grid"><article><h2>Why approximately four?</h2><p>NEON holds four float32 values in one 128-bit register. Memory bandwidth eventually prevents indefinite scaling.</p></article><article><h2>Why do answers differ slightly?</h2><p>SIMD changes accumulation order. Both kernels are checked against float64 accumulation rather than against each other.</p></article></div>
        </section>}

        {section === "matrix" && <section className="panel active">
          <div className="panel-heading compact"><div><p className="context-line">PyTorch CPU GEMM</p><h1>Shape decides whether hardware wakes up.</h1></div><p>Small shapes measure dispatch. Larger shapes expose sustained matrix throughput.</p></div>
          <div className="control-line"><label>Datatype<select value={activeDtype} onChange={(event) => setMatrixDtype(event.target.value)}>{dtypes.map((value) => <option key={value}>{value}</option>)}</select></label><label>Requested threads<select value={activeMatrixThreads} onChange={(event) => setMatrixThreads(Number(event.target.value))}>{matrixThreadValues.map((value) => <option key={value}>{value}</option>)}</select></label><div className="formula-inline"><span>FLOPs</span><code>2 × M × K × N</code></div></div>
          <div className="plot-frame"><div className="plot-header"><h2>Achieved throughput</h2><span>GFLOP/s · higher is better</span></div><LineChart rows={matrixRows} /></div>
          <table className="data-table"><thead><tr><th>Shape</th><th>Median</th><th>p99</th><th>Intensity</th><th>GFLOP/s</th></tr></thead><tbody>{matrixRows.map((row) => <tr key={row.name}><td>{row.workload.m} × {row.workload.k} × {row.workload.n}</td><td>{format(row.latency_ms.median, 3)} ms</td><td>{format(row.latency_ms.p99, 3)} ms</td><td>{format(number(row.estimates.arithmetic_intensity_flops_per_byte), 1)} F/B</td><td>{format(number(row.estimates.achieved_gflops_per_second), 1)}</td></tr>)}</tbody></table>
          <div className="callout"><b>{activeDtype === "bfloat16" ? "Feature support did not produce a fast kernel here." : "Read this as a scaling curve."}</b> {activeDtype === "bfloat16" ? "The framework path is the bottleneck, despite BF16 hardware capability." : "Rising throughput means larger matrices use the backend more efficiently."}</div>
        </section>}

        {section === "transformer" && <section className="panel active">
          <div className="panel-heading compact"><div><p className="context-line">End-to-end workload</p><h1>The model is more than its matmuls.</h1></div><p>Compare inference, gradient calculation and optimizer updates with the same model.</p></div>
          <div className="control-line"><label>Mode<select value={activeMode} onChange={(event) => setTransformerMode(event.target.value)}>{modes.map((value) => <option key={value} value={value}>{value.replaceAll("_", " + ")}</option>)}</select></label><label>Threads<select value={activeTransformerThreads} onChange={(event) => setTransformerThreads(Number(event.target.value))}>{transformerThreadValues.map((value) => <option key={value}>{value}</option>)}</select></label><label>Sequence<select value={activeSequence} onChange={(event) => setTransformerSequence(Number(event.target.value))}>{sequences.map((value) => <option key={value}>{value}</option>)}</select></label></div>
          <div className="transformer-map">{["tokens", "embedding", "attention", "norm", "MLP", "policy"].map((item) => <div className={item === "attention" || item === "MLP" ? "heavy" : ""} key={item}>{item}</div>)}</div>
          {transformerRows.length ? <BarRows rows={transformerRows} metric="throughput" /> : <div className="empty-state">No Transformer report loaded. Run the benchmark and import its JSON.</div>}
        </section>}

        {section === "roofline" && <section className="panel active">
          <div className="panel-heading compact"><div><p className="context-line">Performance bound</p><h1>Hungry for bytes or math?</h1></div><p>Arithmetic intensity sits on the horizontal axis. Achieved throughput sits on the vertical axis.</p></div>
          <div className="roofline-layout"><div className="plot-frame"><div className="plot-header"><h2>Estimated workload positions</h2><span>logarithmic axes</span></div><RooflineChart points={roofPoints} bandwidth={bandwidth} peak={number(peak?.estimates.achieved_gflops_per_second) || 100} /></div><aside className="roofline-key"><p><b>Rising roof</b>Bandwidth-bound region</p><p><b>Flat roof</b>Compute-bound region</p><p><b>Measured points</b>Analytical FLOPs / measured time</p><small>Bytes are estimated, not measured DRAM traffic.</small></aside></div>
        </section>}

        {section === "method" && <section className="panel active">
          <div className="panel-heading compact"><div><p className="context-line">Measurement protocol</p><h1>A fast number must survive a rerun.</h1></div><p>The harness separates startup, steady-state timing, correctness and memory observation.</p></div>
          <ol className="method-list">{[
            ["Warm the path", "Ignore allocator initialization and first dispatch."],
            ["Calibrate the timer", "Repeat fast kernels inside one sample to reduce clock noise."],
            ["Demand stability", "Continue until count, duration and robust-CV targets are satisfied."],
            ["Separate memory", "Sample RSS in another phase so polling does not distort latency."],
            ["Label estimates", "Keep analytical FLOPs and bytes distinct from measured hardware counters."],
          ].map(([title, body], index) => <li key={title}><span>0{index + 1}</span><div><h2>{title}</h2><p>{body}</p></div></li>)}</ol>
          {missing.length > 0 && <div className="callout"><b>Missing local reports:</b> {missing.join(", ")}</div>}
        </section>}
      </main>

      <Inspector insight={insightBySection[section]} simdSpeedup={data.simd?.simd_speedup || 0} matrixPeak={number(peak?.estimates.achieved_gflops_per_second)} forwardLatency={number(forward?.latency_ms.median)} />
    </div>
  </>;
}
