// Inference 100% navigateur des surrogates NACA 0012 (forward "maison" = x@W^T+b + tanh).
// Les poids + constantes de normalisation viennent de public/models.json (exporte depuis PyTorch).

export type Layer = { W: number[][]; b: number[]; act: 'tanh' | 'linear' }
export type Model = {
  inputs: string[]
  alpha_range: [number, number]
  re_range?: [number, number]
  x_mean: number[]; x_std: number[]
  y_mean: number[]; y_std: number[]
  layers: Layer[]
}
type Bundle = { model1d: Model; model2d: Model }

let DATA: Bundle | null = null

export async function loadModels(): Promise<Bundle> {
  if (DATA) return DATA
  const res = await fetch(`${import.meta.env.BASE_URL}models.json`)
  DATA = (await res.json()) as Bundle
  return DATA
}

// Une passe avant : normalise -> couches lineaires (+tanh) -> denormalise.
function forward(m: Model, feat: number[]): { cl: number; cd: number } {
  let x = feat.map((v, i) => (v - m.x_mean[i]) / m.x_std[i])
  for (const L of m.layers) {
    const z = L.W.map((row, i) => row.reduce((s, w, j) => s + w * x[j], L.b[i]))
    x = L.act === 'tanh' ? z.map(Math.tanh) : z
  }
  const out = x.map((v, i) => v * m.y_std[i] + m.y_mean[i])
  return { cl: out[0], cd: out[1] }
}

export function predict1d(alpha: number) {
  return forward(DATA!.model1d, [alpha])
}
export function predict2d(alpha: number, re: number) {
  return forward(DATA!.model2d, [alpha, Math.log10(re)])
}
