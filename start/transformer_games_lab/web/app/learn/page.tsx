import Link from "next/link";

const terms = [
  ["SIMD", "One instruction applies the same operation to several numbers. It is the general idea."],
  ["NEON", "Arm’s 128-bit SIMD instruction family. Four float32 values fit in one vector on the M5."],
  ["AVX", "The x86 SIMD family. It is not expected on Apple Silicon, so PyTorch reporting NO AVX is correct."],
  ["SME / SME2", "Arm matrix extensions exposed by the M5. A library may use them, but the feature bit does not prove dispatch."],
  ["GEMM", "General matrix multiplication: C = A × B. Attention and MLP layers spend much of their arithmetic here."],
  ["Arithmetic intensity", "Useful floating-point operations divided by estimated bytes moved. It is measured in FLOP/byte."],
];

export default function LearnPage() {
  return <main className="learn-page">
    <header className="learn-hero">
      <div><p className="context-line">Systems notebook</p><h1>From one CPU instruction to one game move.</h1></div>
      <p>This is the compact theory companion to the profiler. Read top to bottom once, then use the measurements to test each claim.</p>
    </header>

    <section className="lesson-section">
      <div className="lesson-heading"><span>Vocabulary</span><h2>Names that describe different layers</h2></div>
      <div className="term-list">{terms.map(([term, definition]) => <article key={term}><h3>{term}</h3><p>{definition}</p></article>)}</div>
    </section>

    <section className="lesson-section">
      <div className="lesson-heading"><span>Compute path</span><h2>Where the Transformer spends work</h2></div>
      <div className="operator-stack">
        <div><b>Token lookup</b><span>Memory and indexing</span></div>
        <i>↓</i>
        <div className="hot"><b>Q, K, V projections</b><span>Matrix multiplication</span></div>
        <i>↓</i>
        <div className="hot"><b>QKᵀ and attention × V</b><span>Matrix multiplication</span></div>
        <i>↓</i>
        <div><b>Softmax and LayerNorm</b><span>SIMD math and reductions</span></div>
        <i>↓</i>
        <div className="hot"><b>MLP</b><span>Two matrix multiplications</span></div>
        <i>↓</i>
        <div><b>Policy head</b><span>Logits for every legal action</span></div>
      </div>
    </section>

    <section className="lesson-section split-lesson">
      <div><div className="lesson-heading"><span>Vector example</span><h2>Why NEON can approach 4×</h2></div><div className="math-sheet"><code>scalar: s += a[i] × b[i]</code><code>NEON: v[0:4] += a[0:4] × b[0:4]</code><p>Both perform two FLOPs for each pair of values. NEON performs four pairs per vector instruction. Loads, reductions and memory bandwidth stop the speedup from being automatically perfect.</p></div></div>
      <div><div className="lesson-heading"><span>Matrix example</span><h2>Why larger shapes run differently</h2></div><div className="math-sheet"><code>FLOPs ≈ 2MKN</code><code>minimum bytes ≈ (MK + KN + MN) × element size</code><p>Matrix work grows cubically for square shapes while minimum tensor storage grows quadratically. Larger matrices can therefore reuse data and offer more arithmetic per byte.</p></div></div>
    </section>

    <section className="lesson-section">
      <div className="lesson-heading"><span>Roofline</span><h2>Two ceilings and one knee</h2></div>
      <div className="roof-equation"><code>achievable performance ≤ min(memory bandwidth × arithmetic intensity, compute peak)</code></div>
      <div className="three-notes"><article><b>Left of the knee</b><p>The workload needs bytes faster than the memory system can deliver them.</p></article><article><b>At the knee</b><p>Bandwidth and compute ceilings meet. This intensity is the balance point.</p></article><article><b>Right of the knee</b><p>More data reuse cannot exceed the machine’s compute ceiling.</p></article></div>
    </section>

    <section className="lesson-section final-lesson">
      <div><p className="context-line">Use the instrument</p><h2>Now test the explanation against the M5.</h2></div>
      <Link href="/profiler">Open the profiler</Link>
    </section>
  </main>;
}
