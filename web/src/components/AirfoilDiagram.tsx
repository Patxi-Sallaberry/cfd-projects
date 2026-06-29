// Schema du profil NACA 0012 qui s'incline avec l'angle d'attaque alpha.
// La geometrie vient de l'equation analytique du profil (symetrique, 12% d'epaisseur).

const CHORD = 220
const X0 = 55      // x du bord d'attaque (px)
const YC = 85      // ligne centrale (px)

// Contour NACA 0012 : extrados (LE->TE) puis intrados (TE->LE).
const PATH = (() => {
  const t = 0.12, N = 80
  const yt = (x: number) =>
    5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)
  const px = (x: number) => X0 + x * CHORD
  const py = (y: number) => YC - y * CHORD
  const pts: string[] = []
  for (let i = 0; i <= N; i++) { const x = i / N; pts.push(`${px(x).toFixed(1)},${py(yt(x)).toFixed(1)}`) }
  for (let i = N; i >= 0; i--) { const x = i / N; pts.push(`${px(x).toFixed(1)},${py(-yt(x)).toFixed(1)}`) }
  return 'M' + pts.join(' L') + ' Z'
})()

export default function AirfoilDiagram({ alpha }: { alpha: number }) {
  const cx = X0 + CHORD / 2
  return (
    <svg viewBox="0 0 320 170" className="w-full">
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="rgba(255,255,255,0.5)" />
        </marker>
      </defs>

      {/* Flux amont (horizontal, fixe) */}
      {[-26, 0, 26].map((dy) => (
        <line key={dy} x1="6" y1={YC + dy} x2="46" y2={YC + dy}
          stroke="rgba(255,255,255,0.4)" strokeWidth="1.4" markerEnd="url(#arrow)" />
      ))}
      <text x="6" y={YC - 40} fill="rgba(255,255,255,0.55)" fontSize="11">V∞</text>

      {/* Ligne de reference (direction du flux) */}
      <line x1={X0} y1={YC} x2={X0 + CHORD + 18} y2={YC}
        stroke="rgba(255,255,255,0.18)" strokeWidth="1" strokeDasharray="4 4" />

      {/* Le profil, incline de alpha (bord d'attaque releve vers le flux) */}
      <g transform={`rotate(${-alpha} ${cx} ${YC})`}>
        <path d={PATH} fill="rgba(125,211,252,0.12)" stroke="#7dd3fc" strokeWidth="1.6" />
        {/* corde */}
        <line x1={X0} y1={YC} x2={X0 + CHORD} y2={YC} stroke="rgba(255,255,255,0.35)" strokeWidth="1" />
      </g>

      {/* angle alpha annote au bord d'attaque */}
      <text x={X0 + 6} y={YC + 34} fill="rgba(255,255,255,0.7)" fontSize="12">
        α = {alpha.toFixed(1)}°
      </text>
    </svg>
  )
}
