import { useEffect, useRef } from 'react'

// Fond CFD anime : un champ de particules s'ecoulant de gauche a droite avec une
// deviation sinusoidale (evoque des lignes de courant). Trainees douces sur fond sombre.
export default function FlowBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    let raf = 0
    let w = 0, h = 0
    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    type P = { x: number; y: number; vx: number; py: number }
    let parts: P[] = []

    function resize() {
      w = canvas.clientWidth; h = canvas.clientHeight
      canvas.width = w * dpr; canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const n = Math.floor((w * h) / 9000)            // densite adaptee a l'ecran
      parts = Array.from({ length: n }, () => spawn())
      ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, w, h)
    }

    function spawn(fromLeft = false): P {
      const y = Math.random() * h
      return { x: fromLeft ? -10 : Math.random() * w, y, py: y, vx: 0.4 + Math.random() * 1.1 }
    }

    function frame(t: number) {
      // voile sombre semi-transparent -> trainees qui s'estompent
      ctx.fillStyle = 'rgba(5,7,10,0.10)'
      ctx.fillRect(0, 0, w, h)
      ctx.lineWidth = 1
      for (const p of parts) {
        p.py = p.y
        // deviation verticale type ligne de courant (depend de x, y et du temps)
        const angle = Math.sin(p.x * 0.004 + p.y * 0.012 + t * 0.0002) * 0.6
        p.x += p.vx
        p.y += Math.sin(angle) * 1.2
        const a = 0.06 + (p.vx / 1.5) * 0.10           // plus rapide = plus visible
        ctx.strokeStyle = `rgba(255,255,255,${a})`
        ctx.beginPath(); ctx.moveTo(p.x - p.vx, p.py); ctx.lineTo(p.x, p.y); ctx.stroke()
        if (p.x > w + 10 || p.y < -10 || p.y > h + 10) Object.assign(p, spawn(true))
      }
      raf = requestAnimationFrame(frame)
    }

    resize()
    raf = requestAnimationFrame(frame)
    window.addEventListener('resize', resize)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full"
      style={{ display: 'block', pointerEvents: 'none' }}
    />
  )
}
