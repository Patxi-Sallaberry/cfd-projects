import { useRef, type CSSProperties } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'

type Props = {
  text: string
  className?: string
  delay?: number      // ms de decalage entre chaque caractere (stagger)
  duration?: number
  ease?: string
  style?: CSSProperties
}

// Decoupe le texte en caracteres et les anime (montee + fondu) avec un stagger GSAP.
export default function SplitText({
  text, className = '', delay = 60, duration = 0.8, ease = 'power3.out', style,
}: Props) {
  const ref = useRef<HTMLHeadingElement>(null)

  useGSAP(() => {
    if (!ref.current) return
    const chars = ref.current.querySelectorAll<HTMLElement>('.st-char')
    gsap.fromTo(
      chars,
      { opacity: 0, y: 40 },
      { opacity: 1, y: 0, duration, ease, stagger: delay / 1000 },
    )
  }, { scope: ref, dependencies: [text] })

  return (
    <h1 ref={ref} className={className} style={style}>
      {Array.from(text).map((ch, i) => (
        <span key={i} className="st-char" style={{ display: 'inline-block', whiteSpace: 'pre' }}>
          {ch}
        </span>
      ))}
    </h1>
  )
}
