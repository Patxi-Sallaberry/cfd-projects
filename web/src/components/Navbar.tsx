import { ArrowUpRight } from 'lucide-react'

const LINKS = ['Formula Student', 'Models', 'Method', 'GitHub']

export default function Navbar() {
  return (
    <nav className="px-4 sm:px-6 lg:px-10 py-4 lg:py-5 flex items-center justify-between">
      {/* Logo + wordmark */}
      <a href="#top" className="flex items-center gap-2.5">
        <svg width="22" height="22" viewBox="0 0 256 256" aria-hidden="true">
          <path
            d="M 256 256 L 128 256 L 0 128 L 128 128 Z M 256 128 L 128 128 L 0 0 L 128 0 Z"
            fill="white"
          />
        </svg>
        <span className="text-white font-light text-lg tracking-tight">Patxi Sallaberry</span>
      </a>

      {/* Liens (caches sous lg) */}
      <div className="hidden lg:flex items-center gap-8">
        {LINKS.map((l) => (
          <a
            key={l}
            href={l === 'GitHub'
              ? 'https://github.com/Patxi-Sallaberry/cfd-projects'
              : l === 'Formula Student' ? '#fs'
              : `#${l.toLowerCase()}`}
            className="text-white/70 hover:text-white text-sm transition-colors"
          >
            {l}
          </a>
        ))}
      </div>

      {/* CTA pilule glass */}
      <a
        href="#fs"
        className="liquid-glass rounded-xl px-4 sm:px-5 py-2 text-white/90 text-xs sm:text-sm font-light flex items-center gap-1.5"
        style={{ background: 'rgba(255,255,255,0.22)' }}
      >
        FS tool
        <ArrowUpRight size={14} />
      </a>
    </nav>
  )
}
