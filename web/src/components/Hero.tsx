import { ArrowRight, ArrowUpRight } from 'lucide-react'
import SplitText from './SplitText'

export default function Hero() {
  return (
    <div className="flex flex-col items-center text-center px-4 sm:px-6 pt-8 sm:pt-12 lg:pt-16 pb-10 lg:pb-14">
      {/* Badge */}
      <div className="liquid-glass rounded-full px-3 sm:px-4 py-1.5 mb-5 flex items-center gap-2.5">
        <span className="text-white/80 text-xs sm:text-sm">NACA 0012 · neural surrogate</span>
        <span className="text-white/40">|</span>
        <a href="#method" className="text-white/70 hover:text-white text-xs sm:text-sm flex items-center gap-1 transition-colors">
          Method <ArrowRight size={13} />
        </a>
      </div>

      {/* Titre anime */}
      <div
        className="flex flex-col items-center"
        style={{ fontSize: 'clamp(36px, 8vw, 76px)', letterSpacing: '-1.5px', lineHeight: 1.1, fontWeight: 200, textShadow: '0 2px 20px rgba(0,0,0,0.3)' }}
      >
        <SplitText text="CFD ×" />
        <div style={{ marginTop: '-0.15em' }}>
          <SplitText text="Machine Learning" delay={45} />
        </div>
      </div>

      {/* Sous-texte */}
      <p
        className="text-white/65 max-w-md mt-5 mb-7 leading-relaxed px-2 hero-fade-up"
        style={{ fontSize: 'clamp(13px, 1.5vw, 17px)', lineHeight: 1.6, animationDelay: '0.6s' }}
      >
        A neural surrogate trained on NACA 0012 aerodynamics —<br className="hidden sm:block" />
        predicting lift &amp; drag in milliseconds, live in your browser.
      </p>

      {/* CTA */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 sm:gap-4 w-full sm:w-auto max-w-xs sm:max-w-none">
        <a
          href="#live"
          className="liquid-glass rounded-xl px-6 sm:px-7 py-2.5 text-white font-light flex items-center justify-center gap-2.5 transition-all duration-200 group hero-fade-up"
          style={{ background: 'rgba(255,255,255,0.22)', animationDelay: '0.85s', fontSize: 15 }}
        >
          Try the model
          <ArrowUpRight size={18} className="group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
        </a>
        <a
          href="https://github.com/Patxi-Sallaberry/cfd-projects"
          target="_blank" rel="noreferrer"
          className="liquid-glass rounded-xl px-6 sm:px-7 py-2.5 text-white font-light flex items-center justify-center gap-2.5 transition-all duration-200 hover:bg-white/8 hero-fade-up"
          style={{ animationDelay: '1.0s', fontSize: 15 }}
        >
          View on GitHub
        </a>
      </div>
    </div>
  )
}
