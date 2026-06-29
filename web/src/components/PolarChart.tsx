type Pt = { x: number; y: number }

// Petit graphe SVG : une courbe (sweep en alpha) + le point courant (+ marqueur optimum).
export default function PolarChart({
  points, current, color = '#fff', title, markX,
}: { points: Pt[]; current: Pt; color?: string; title: string; markX?: number }) {
  const W = 320, H = 180, pad = 30
  if (points.length === 0) return <div className="text-white/40 text-xs">…</div>

  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const xmin = Math.min(...xs), xmax = Math.max(...xs)
  let ymin = Math.min(...ys), ymax = Math.max(...ys)
  if (ymin === ymax) { ymin -= 1; ymax += 1 }
  const m = (ymax - ymin) * 0.12; ymin -= m; ymax += m

  const sx = (x: number) => pad + ((x - xmin) / (xmax - xmin)) * (W - 2 * pad)
  const sy = (y: number) => H - pad - ((y - ymin) / (ymax - ymin)) * (H - 2 * pad)
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')

  return (
    <div>
      <div className="text-white/55 text-xs mb-1">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {ymin < 0 && ymax > 0 && (
          <line x1={pad} y1={sy(0)} x2={W - pad} y2={sy(0)} stroke="rgba(255,255,255,0.15)" />
        )}
        <line x1={sx(current.x)} y1={pad} x2={sx(current.x)} y2={H - pad}
          stroke="rgba(255,255,255,0.14)" strokeDasharray="3 3" />
        {markX !== undefined && (
          <line x1={sx(markX)} y1={pad} x2={sx(markX)} y2={H - pad}
            stroke="#4ade80" strokeWidth={1.4} strokeDasharray="2 2" />
        )}
        <path d={d} fill="none" stroke={color} strokeWidth={1.6} />
        <circle cx={sx(current.x)} cy={sy(current.y)} r={4.5} fill={color} />
      </svg>
    </div>
  )
}
