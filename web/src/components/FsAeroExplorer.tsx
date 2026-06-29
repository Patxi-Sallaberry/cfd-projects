import { useEffect, useMemo, useState } from 'react'
import { Gauge } from 'lucide-react'
import { loadFsModel, predictFs, fsModel } from '../lib/model'
import PolarChart from './PolarChart'

export default function FsAeroExplorer() {
  const [ready, setReady] = useState(false)
  const [alpha, setAlpha] = useState(6)     // wing angle (deg)
  const [V, setV] = useState(20)            // car speed (m/s)

  useEffect(() => { loadFsModel().then(() => setReady(true)) }, [])

  const phys = ready ? fsModel().physics : { chord: 0.3, span: 1.2, rho: 1.225, mu: 1.789e-5 }
  const [aMin, aMax] = ready ? fsModel().alpha_range : [-4, 16]
  const [vMin, vMax] = ready ? fsModel().v_range : [5, 35]

  const S = phys.chord * phys.span
  const Re = (phys.rho * V * phys.chord) / phys.mu
  const q = 0.5 * phys.rho * V * V

  // Finite-wing correction: total drag = 2-D profile drag + induced drag Cl²/(π·AR·e).
  const AR = phys.span / phys.chord         // aspect ratio
  const OSWALD = 0.85
  const cdFull = (cl: number, cd: number) => cd + (cl * cl) / (Math.PI * AR * OSWALD)

  const cur = ready ? predictFs(alpha, Re) : { cl: 0, cd: 0 }
  const cdTot = cdFull(cur.cl, cur.cd)
  const downforce = cur.cl * q * S          // N
  const drag = cdTot * q * S                // N (profile + induced)
  const ld = cdTot ? cur.cl / cdTot : 0

  // sweep angle at the current speed -> downforce & efficiency curves
  const sweep = useMemo(() => {
    const df: { x: number; y: number }[] = []
    const eff: { x: number; y: number }[] = []
    if (ready) {
      for (let a = aMin; a <= aMax + 1e-6; a += 0.25) {
        const p = predictFs(a, Re)
        const cdt = cdFull(p.cl, p.cd)
        df.push({ x: a, y: p.cl * q * S })
        eff.push({ x: a, y: cdt ? p.cl / cdt : 0 })
      }
    }
    return { df, eff }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, Re, q, S])

  const best = (arr: { x: number; y: number }[]) =>
    arr.reduce((b, p) => (p.y > b.y ? p : b), { x: alpha, y: -Infinity })
  const bestDf = best(sweep.df)
  const bestEff = best(sweep.eff)

  const Stat = ({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: boolean }) => (
    <div className="flex-1 text-center">
      <div className="text-white/50 text-xs mb-1">{label}</div>
      <div className={`text-2xl sm:text-3xl tabular-nums ${accent ? 'text-emerald-300' : 'text-white'}`}>{value}</div>
      {sub && <div className="text-white/40 text-xs mt-0.5">{sub}</div>}
    </div>
  )

  return (
    <section id="fs" className="px-4 sm:px-6 lg:px-10 py-12 lg:py-16">
      <div className="max-w-3xl mx-auto text-center mb-7">
        <div className="inline-flex items-center gap-2 text-emerald-300/90 text-xs mb-2">
          <Gauge size={15} /> Formula Student
        </div>
        <h2 className="text-white text-2xl sm:text-3xl tracking-tight">Wing aero explorer</h2>
        <p className="text-white/55 text-sm mt-2 max-w-xl mx-auto">
          A NACA 4412 wing element (inverted, for downforce). Set the angle and car speed — downforce,
          drag and efficiency update instantly, and the tool finds the optimal angle. Runs in your
          browser via a neural surrogate.
        </p>
      </div>

      <div className="liquid-glass rounded-2xl max-w-3xl mx-auto p-5 sm:p-7" style={{ background: 'rgba(255,255,255,0.03)' }}>
        {/* Sliders */}
        <div className="space-y-5 mb-6">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-white/70">Wing angle</span>
              <span className="text-white tabular-nums">{alpha.toFixed(1)}°</span>
            </div>
            <input type="range" min={aMin} max={aMax} step={0.5} value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))} />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-white/70">Car speed</span>
              <span className="text-white tabular-nums">{V.toFixed(0)} m/s · {(V * 3.6).toFixed(0)} km/h</span>
            </div>
            <input type="range" min={vMin} max={vMax} step={1} value={V}
              onChange={(e) => setV(Number(e.target.value))} />
          </div>
        </div>

        {/* Live outputs */}
        <div className="liquid-glass rounded-xl py-4 px-3 flex gap-2 mb-5">
          <Stat label="Downforce" value={`${downforce.toFixed(0)} N`} sub={`≈ ${(downforce / 9.81).toFixed(0)} kg`} accent />
          <Stat label="Drag" value={`${drag.toFixed(0)} N`} />
          <Stat label="Efficiency (L/D)" value={ld.toFixed(1)} />
        </div>

        {/* Optimizer */}
        <div className="flex flex-col sm:flex-row gap-2 mb-6">
          <button
            onClick={() => setAlpha(Math.round(bestDf.x * 2) / 2)}
            className="liquid-glass rounded-lg px-4 py-2 text-white text-sm flex-1"
            style={{ background: 'rgba(74,222,128,0.18)' }}
          >
            ⚙ Max downforce → set angle to {bestDf.x.toFixed(1)}° ({bestDf.y.toFixed(0)} N)
          </button>
          <button
            onClick={() => setAlpha(Math.round(bestEff.x * 2) / 2)}
            className="liquid-glass rounded-lg px-4 py-2 text-white/90 text-sm flex-1 hover:bg-white/8"
          >
            ⚙ Best efficiency → {bestEff.x.toFixed(1)}° (L/D {bestEff.y.toFixed(1)})
          </button>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <PolarChart title="Downforce vs angle (N)" color="#6ee7b7"
            points={sweep.df} current={{ x: alpha, y: downforce }} markX={bestDf.x} />
          <PolarChart title="Efficiency (L/D) vs angle" color="#7dd3fc"
            points={sweep.eff} current={{ x: alpha, y: ld }} markX={bestEff.x} />
        </div>

        <p className="text-white/40 text-xs mt-5 text-center">
          Estimate = a 2-D airfoil surrogate (NeuralFoil/XFOIL-class) + a finite-wing <b>induced-drag</b>
          correction (aspect ratio {AR.toFixed(1)}, Oswald e = {OSWALD}). Great for early trade-offs and
          setup — <b>not</b> a full 3-D CFD (no ground effect / multi-element).
          Wing: chord {phys.chord} m × span {phys.span} m.
        </p>
        {!ready && <p className="text-white/40 text-xs mt-3 text-center">loading model…</p>}
      </div>
    </section>
  )
}
