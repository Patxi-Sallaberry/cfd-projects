# Live demo — NACA 0012 surrogate (web)

Interactive site that runs the [Phase 1](../piml/phase1_surrogate) surrogates **entirely in the
browser** — no backend, no server cost.

**Live:** https://patxi-sallaberry.github.io/cfd-projects/

## How it works

- The trained PyTorch models' **weights + normalization constants** are exported to
  `public/models.json` by [`../piml/phase1_surrogate/src/export_weights.py`](../piml/phase1_surrogate/src/export_weights.py).
- `src/lib/model.ts` reimplements the **forward pass** (`x·Wᵀ + b` then `tanh`) in ~15 lines of
  JavaScript — so the visitor's browser does the inference. An automated test verifies it matches
  PyTorch exactly.
- Sliders for **α** and **Re** drive live Cl / Cd / L-D values and polar charts; a NACA 0012
  schematic tilts with the angle of attack so the geometry is obvious.

## Stack

Vite · React · TypeScript · Tailwind CSS v4 · GSAP · lucide-react.

## Develop / build / deploy

```bash
npm install
npm run dev                          # local dev server
npm run build                        # production build -> dist/
npx gh-pages -d dist --dotfiles      # deploy to the gh-pages branch
```

`vite.config.ts` uses `base: './'` so the build works both locally and under the GitHub Pages
sub-path (`/cfd-projects/`).
