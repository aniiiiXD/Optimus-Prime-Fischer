import type { BenchmarkResult } from "@/lib/types";

const format = (value: number, digits = 1) => value.toLocaleString(undefined, { maximumFractionDigits: digits });

export function LineChart({ rows }: { rows: BenchmarkResult[] }) {
  const width = 760;
  const height = 330;
  const margin = { left: 62, right: 22, top: 30, bottom: 48 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const values = rows.map((row) => Number(row.estimates.achieved_gflops_per_second) || 0);
  const maximum = Math.max(...values, 1) * 1.12;
  const step = rows.length > 1 ? innerWidth / (rows.length - 1) : 0;
  const points = rows.map((row, index) => ({
    x: margin.left + (rows.length > 1 ? index * step : innerWidth / 2),
    y: margin.top + innerHeight - (Number(row.estimates.achieved_gflops_per_second) || 0) / maximum * innerHeight,
    row,
  }));

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Matrix throughput by size">
      {Array.from({ length: 5 }, (_, index) => {
        const value = maximum * index / 4;
        const y = margin.top + innerHeight - innerHeight * index / 4;
        return <g key={index}><line className="grid" x1={margin.left} x2={width - margin.right} y1={y} y2={y} /><text x={margin.left - 10} y={y + 4} textAnchor="end">{format(value, 0)}</text></g>;
      })}
      <line className="axis" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
      <line className="axis" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
      {points.length > 1 && <path className="series-line" d={`M ${points.map((point) => `${point.x} ${point.y}`).join(" L ")}`} />}
      {points.map((point) => <g key={point.row.name}>
        <circle className="series-point" cx={point.x} cy={point.y} r="5"><title>{format(Number(point.row.estimates.achieved_gflops_per_second))} GFLOP/s</title></circle>
        <text x={point.x} y={height - 22} textAnchor="middle">{String(point.row.workload.m)}³</text>
      </g>)}
    </svg>
  );
}

function logPosition(value: number, minimum: number, maximum: number, start: number, length: number) {
  const clamped = Math.max(minimum, Math.min(maximum, value));
  return start + (Math.log10(clamped) - Math.log10(minimum)) / (Math.log10(maximum) - Math.log10(minimum)) * length;
}

type RoofPoint = { result: BenchmarkResult; label: string };

export function RooflineChart({ points, bandwidth, peak }: { points: RoofPoint[]; bandwidth: number; peak: number }) {
  const width = 760;
  const height = 440;
  const margin = { left: 62, right: 24, top: 28, bottom: 54 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const minX = 0.1;
  const maxX = 1000;
  const minY = 0.1;
  const maxY = Math.max(4000, peak * 1.5);
  const x = (value: number) => logPosition(value, minX, maxX, margin.left, innerWidth);
  const y = (value: number) => margin.top + innerHeight - logPosition(value, minY, maxY, 0, innerHeight);
  const knee = peak / bandwidth;

  return (
    <svg className="chart roofline-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Estimated roofline plot">
      {[0.1, 1, 10, 100, 1000].map((tick) => <g key={`x-${tick}`}>
        <line className="grid" x1={x(tick)} x2={x(tick)} y1={margin.top} y2={height - margin.bottom} />
        <text x={x(tick)} y={height - 27} textAnchor="middle">{tick}</text>
      </g>)}
      {[0.1, 1, 10, 100, 1000].filter((tick) => tick <= maxY).map((tick) => <g key={`y-${tick}`}>
        <line className="grid" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text x={margin.left - 10} y={y(tick) + 4} textAnchor="end">{tick}</text>
      </g>)}
      <path className="roof-memory" d={`M ${x(minX)} ${y(bandwidth * minX)} L ${x(Math.min(knee, maxX))} ${y(Math.min(peak, bandwidth * maxX))}`} />
      {knee < maxX && <path className="roof-compute" d={`M ${x(Math.max(knee, minX))} ${y(peak)} L ${x(maxX)} ${y(peak)}`} />}
      {points.map(({ result, label }) => {
        const intensity = Number(result.estimates.arithmetic_intensity_flops_per_byte) || 0;
        const rate = Number(result.estimates.achieved_gflops_per_second) || 0;
        if (!intensity || !rate) return null;
        return <circle key={`${result.name}-${label}`} className="roof-point" cx={x(intensity)} cy={y(rate)} r="5"><title>{label}: {format(intensity, 2)} F/B, {format(rate)} GF/s</title></circle>;
      })}
      <text x={margin.left + innerWidth / 2} y={height - 5} textAnchor="middle">arithmetic intensity · FLOP / estimated byte</text>
      <text x="14" y={margin.top + innerHeight / 2} transform={`rotate(-90 14 ${margin.top + innerHeight / 2})`} textAnchor="middle">achieved GFLOP/s</text>
    </svg>
  );
}
