import { useEffect, useMemo, useState } from 'react'
import { loadModels, predict1d, predict2d } from '../lib/model'
import PolarChart from './PolarChart'

const RE_1D = 4e5
const fmtRe = (re: number) => {
  const e = Math.floor(Math.log10(re))
  return `${(re / 10 ** e).toFixed(1)}·10${['⁰','¹','²','³','⁴','⁵','⁶'][e] ?? `^${e}`}`
}

export default function LiveDemo() {
  const [ready, setReady] = useState(false)
  const [mode, setMode] = useState<'1d' | '2d'>('2d')
  const [alpha, setAlpha] = useState(5)
  const [re, setRe] = useState(4e5)

  useEffect(() => { loadModels().then(() => setReady(true)) }, [])

  const predict = (a: number) => (mode === '2d' ? predict2d(a, re) : predict1d(a))
  const cur = ready ? predict(alpha) : { cl: 0, cd: 0 }
  const ld = cur.cd !== 0 ? cur.cl / cur.cd : 0

  const sweep = useMemo(() => {
    const cl: { x: number; y: number }[] = []
    const cd: { x: number; y: number }[] = []
    if (ready) {
      for (let a = -6; a <= 16.0001; a += 0.5) {
        const p = mode === '2d' ? predict2d(a, re) : predict1d(a)
        cl.push({ x: a, y: p.cl }); cd.push({ x: a, y: p.cd })
      }
    }
    return { cl, cd }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, mode, re])

  const Stat = ({ label, value }: { label: string; value: string }) => (
    <div className="flex-1 text-center">
      <div className="text-white/50 text-xs mb-1">{label}</div>
      <div className="text-white text-2xl sm:text-3xl tabular-nums">{value}</div>
    </div>
  )

  return (
    <section id="live" className="px-4 sm:px-6 lg:px-10 py-12 lg:py-16">
      <div id="models" className="max-w-3xl mx-auto text-center mb-7">
        <h2 className="text-white text-2xl sm:text-3xl tracking-tight">Run the surrogate, live</h2>
        <p className="text-white/55 text-sm mt-2 max-w-xl mx-auto">
          The model runs entirely in your browser — no server. Move the sliders and the lift,
          drag and L/D update instantly.
        </p>
      </div>

      <div className="liquid-glass rounded-2xl max-w-3xl mx-auto p-5 sm:p-7" style={{ background: 'rgba(255,255,255,0.03)' }}>
        {/* Toggle modele */}
        <div className="flex justify-center gap-2 mb-6">
          {(['2d', '1d'] as const).map((mO) => (
            <button
              key={mO}
              onClick={() => setMode(mO)}
              className={`rounded-lg px-4 py-1.5 text-xs sm:text-sm transition-all ${
                mode === mO ? 'liquid-glass text-white' : 'text-white/50 hover:text-white/80'}`}
              style={mode === mO ? { background: 'rgba(255,255,255,0.18)' } : undefined}
            >
              {mO === '2d' ? 'Model 2D · (α, Re)' : 'Model 1D · α only'}
            </button>
          ))}
        </div>

        {/* Sliders */}
        <div className="space-y-5 mb-6">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-white/70">Angle of attack α</span>
              <span className="text-white tabular-nums">{alpha.toFixed(1)}°</span>
            </div>
            <input type="range" min={-6} max={16} step={0.1} value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))} />
          </div>

          {mode === '2d' ? (
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-white/70">Reynolds number Re</span>
                <span className="text-white tabular-nums">{fmtRe(re)}</span>
              </div>
              <input type="range" min={5} max={Math.log10(1.5e6)} step={0.01}
                value={Math.log10(re)} onChange={(e) => setRe(10 ** Number(e.target.value))} />
            </div>
          ) : (
            <div className="text-white/45 text-xs">Reynolds fixed at {fmtRe(RE_1D)} (1D model).</div>
          )}
        </div>

        {/* Sorties */}
        <div className="liquid-glass rounded-xl py-4 px-3 flex gap-2 mb-6">
          <Stat label="Cl (lift)" value={cur.cl.toFixed(3)} />
          <Stat label="Cd (drag)" value={cur.cd.toFixed(4)} />
          <Stat label="L / D" value={ld.toFixed(1)} />
        </div>

        {/* Graphes */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <PolarChart title="Cl vs α" color="#7dd3fc" points={sweep.cl} current={{ x: alpha, y: cur.cl }} />
          <PolarChart title="Cd vs α" color="#fca5a5" points={sweep.cd} current={{ x: alpha, y: cur.cd }} />
        </div>

        {mode === '2d' && (
          <p className="text-white/40 text-xs mt-5 text-center">
            Trained on Re ∈ [10⁵, 1.5·10⁶]. Inside this range it interpolates; outside, predictions
            are extrapolation — not trustworthy.
          </p>
        )}
        {!ready && <p className="text-white/40 text-xs mt-4 text-center">loading model…</p>}
      </div>
    </section>
  )
}
