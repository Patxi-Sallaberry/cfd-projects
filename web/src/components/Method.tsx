import { Wind, ShieldCheck, BrainCircuit } from 'lucide-react'

const STEPS = [
  {
    icon: Wind,
    title: '1 · CFD',
    body: 'NACA 0012 simulated in ANSYS Fluent (k-ω SST). Lift/drag extracted across angles of attack and Reynolds numbers.',
  },
  {
    icon: ShieldCheck,
    title: '2 · Verification & Validation',
    body: 'Each run is compared to a documented reference polar. Non-physical results are caught and diagnosed — not published as truth.',
  },
  {
    icon: BrainCircuit,
    title: '3 · Surrogate (this demo)',
    body: 'A small PyTorch MLP learns (α, Re) → (Cl, Cd) on a clean reference dataset, then predicts instantly — exported to run client-side here.',
  },
]

export default function Method() {
  return (
    <section id="method" className="px-4 sm:px-6 lg:px-10 py-12 lg:py-20">
      <div id="approach" className="max-w-3xl mx-auto text-center mb-10">
        <h2 className="text-white text-2xl sm:text-3xl tracking-tight">From CFD to a real-time model</h2>
        <p className="text-white/55 text-sm mt-2 max-w-xl mx-auto">
          The pipeline behind this demo — physics first, then machine learning, with verification at every step.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl mx-auto">
        {STEPS.map(({ icon: Icon, title, body }) => (
          <div key={title} className="liquid-glass rounded-2xl p-5" style={{ background: 'rgba(255,255,255,0.03)' }}>
            <Icon size={22} className="text-white/80 mb-3" strokeWidth={1.4} />
            <h3 className="text-white text-base mb-2">{title}</h3>
            <p className="text-white/55 text-sm leading-relaxed">{body}</p>
          </div>
        ))}
      </div>

      <div className="text-center mt-10">
        <a
          href="https://github.com/Patxi-Sallaberry/cfd-projects"
          target="_blank" rel="noreferrer"
          className="liquid-glass inline-flex rounded-xl px-6 py-2.5 text-white font-light text-sm"
          style={{ background: 'rgba(255,255,255,0.16)' }}
        >
          Full project & code on GitHub
        </a>
      </div>
    </section>
  )
}
