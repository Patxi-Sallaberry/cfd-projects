# `references/` — PINN & Scientific-ML knowledge base

A curated, **verified** knowledge base on physics-informed neural networks (PINNs) and physics-applied
machine learning. Its purpose is twofold:

1. **Boost Claude Code in this repo.** Claude reads the repository as context, so a high-signal,
   accurate reference corpus here means better ML/PINN help — grounded in real papers and in this
   project's own results, instead of generic recall. (`CLAUDE.md` points Claude here.)
2. **A learning resource for Patxi** — a compact map of the field for the CFD × PIML roadmap.

## Contents
| File | What it is |
|---|---|
| [`bibliography.md`](bibliography.md) | Annotated, grouped bibliography (foundations, reviews, training pathologies, variants, operator learning, CFD applications, software). Every citation checked against the source. |
| [`pinn_playbook.md`](pinn_playbook.md) | Practical guide: the core recipe, a **failure-mode → fix** diagnostic table, formulation choices, V&V discipline, operator learning, CFD notes — cross-linked to this repo's Phase-2 experiments. |
| [`theme_classical_cfd_x_pinn.md`](theme_classical_cfd_x_pinn.md) | Thematic note: **bridging classical CFD/numerics with PINNs** (SIMPLE-PINN, SP-PINN, cPINN/XPINN, VPINN, PPINN, RAR…) — a "classical idea → PINN counterpart" map, and Phase-3 hooks. |

## How it was built
Anchored on the review **Ganga & Uddin, "Exploring PINNs: From Fundamentals to Applications in Complex
Systems"** (a PDF in Patxi's library), whose reference list was mined for primary sources, then key
citations (Raissi 2019; Karniadakis 2021; Krishnapriyan 2021; DeepONet 2021; FNO 2021; the "Expert's
Guide" 2023) were **web-verified**. No fabricated citations.

## Principle
Accuracy over volume. Prefer a *correct, cross-linked* entry to a long unverified list. When new
experiments teach a lesson, extend `pinn_playbook.md` and link the README section that shows it.

## Related in this repo
- [`../phase2_pinns/README.md`](../phase2_pinns/README.md) — the worked PINN progression (ODE → PDE →
  inverse → Navier–Stokes → airfoil → parametric), including the honest limitations these references explain.
- [`../../docs/pytorch_guide.md`](../../docs/pytorch_guide.md) — the French PyTorch/autograd guide (§20 = PINNs).
