# Applied PINNs — standalone showcases

Small, self-contained physics-informed neural networks on **concrete engineering applications**, each in
a **different physical domain** than the Phase-2 roadmap (which is fluid-flow focused) and independent of
the airfoil work. They are deliberately **lightweight (CPU, ~1 min)** and each is **validated against an
analytic solution**.

| Showcase | Domain | Equation | Status |
|---|---|---|---|
| [`wing_spar_deflection/`](wing_spar_deflection) | structural mechanics (aerospace) | Euler–Bernoulli beam `EI·w'''' = q(x)` (4th order) | ✅ R² = 1.0 |

*Rationale, methods and citations live in the knowledge base: [`../references/`](../references).*
