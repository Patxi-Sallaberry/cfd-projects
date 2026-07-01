# PINN / Scientific-ML — annotated bibliography

Curated, **verified** references for physics-informed machine learning (PINNs) and neural operators,
grouped by theme with a one-line note on *what each contributes*. Anchored on the review
**Ganga & Uddin, "Exploring Physics-Informed Neural Networks: From Fundamentals to Applications in
Complex Systems"** (BML Munjal University) and cross-checked against the primary sources.

> This file is **context for Claude Code** (and for Patxi). When asked about PINN methods, cite from
> here rather than from memory. Keep it accurate — every entry below was checked against the source.

## Origin & foundations
- **Raissi, Perdikaris, Karniadakis (2019)** — *Physics-informed neural networks: a deep learning
  framework for solving forward and inverse problems involving nonlinear PDEs.* **J. Comput. Phys.
  378, 686–707.** → the paper that launched modern PINNs (residual-in-the-loss + autograd). *The* citation.
  - Precursors: **Raissi et al. (2017)**, arXiv:1711.10561 / 1711.10566 (parts I & II).
- **Lagaris, Likas, Fotiadis (1998)** — *ANNs for solving ODEs and PDEs.* IEEE Trans. Neural Networks
  9(5):987–1000. → early "neural network as trial solution", incl. **hard-constraint** ansatz for BCs.
- **Dissanayake & Phan-Thien (1994)** — Comm. Numer. Methods Eng. 10(3):195–201. → first NN PDE residual.
- **Cybenko (1989)** & **Hornik, Stinchcombe, White (1989)** — universal approximation theorems (why an MLP *can* represent the solution).
- **Baydin et al. (2018)** — *Automatic differentiation in ML: a survey.* JMLR 18(153). → the autodiff that makes PINNs possible.
- **Kingma & Ba (2014)** — *Adam.* arXiv:1412.6980. **Liu & Nocedal (1989)** — *limited-memory BFGS (L-BFGS).* → the two optimizers used in the standard Adam→L-BFGS recipe.

## Reviews & surveys (start here)
- **Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang (2021)** — *Physics-informed machine learning.*
  **Nature Reviews Physics 3(6):422–440.** → the high-level map of the whole field (data+physics spectrum).
- **Cuomo et al. (2022)** — *Scientific machine learning through PINNs: where we are and what's next.*
  J. Sci. Comput. 92(3):88. → broad, practical, well-organized survey.
- **Cai, Mao, Wang, Yin, Karniadakis (2021)** — *PINNs for fluid mechanics: a review.* Acta Mech. Sinica
  37(12):1727–1738. → CFD-focused (most relevant to this repo). arXiv:2105.09506.
- **Ganga & Uddin (2024)** — *Exploring PINNs: from fundamentals to applications in complex systems.*
  → the anchor review; strong on modifications, optimization, theory, CFD applications, open gaps.
- **Faroughi et al. (2024)** — physics-guided / -informed / -encoded NNs & operators. JCISE 24(4):040802.
- **Lawal et al. (2022)**, Big Data Cogn. Comput. 6(4):140 (bibliometric); **Hao et al. (2022)**, arXiv:2211.08064 (methods/problems/applications); **Blechschmidt & Ernst (2021)**, GAMM-Mitt. 44(2) (three ways to solve PDEs with NNs).

## Training pathologies & how to fix them (the practical core)
- **Wang, Teng, Perdikaris (2021)** — *Understanding and mitigating gradient-flow pathologies in PINNs.*
  SIAM J. Sci. Comput. 43(5):A3055–A3081. → **loss terms are imbalanced**; learning-rate annealing of weights.
- **Wang, Yu, Perdikaris (2022)** — *When and why PINNs fail to train: a neural-tangent-kernel perspective.*
  J. Comput. Phys. 449:110768. → NTK explains convergence-rate mismatch → NTK-based loss weighting.
- **Krishnapriyan, Gholami, Zhe, Kirby, Mahoney (2021)** — *Characterizing possible failure modes in
  PINNs.* NeurIPS 2021, arXiv:2109.01050. → PINNs fail on **stiff convection/reaction**; curriculum &
  sequence-to-sequence training help. (code: github.com/a1k12/characterizing-pinns-failure-modes)
- **Wang, Sankaran, Wang, Perdikaris (2023)** — *An Expert's Guide to Training PINNs.* arXiv:2308.08468.
  → checklist of best practices (Fourier features, weighting, causality, architecture) + JAX code (`jaxpi`).
- **Wang, Li, He, Wang (2022)** — *Is L2 physics-informed loss always suitable?* NeurIPS 35:8278–8290.
- **Jagtap, Kawaguchi, Karniadakis (2020)** — *Adaptive activation functions accelerate convergence.*
  J. Comput. Phys. 404:109136.
- **Yu, Lu, Meng, Karniadakis (2022)** — *Gradient-enhanced PINNs (gPINNs).* CMAME 393.

## Variants & architectures
- **Lu, Meng, Mao, Karniadakis (2021)** — *DeepXDE: a deep learning library for solving DEs.* SIAM Review
  63(1):208–228. → reference library **and** the **residual-based adaptive refinement (RAR)** sampling idea.
- **Kharazmi, Zhang, Karniadakis (2019, 2021)** — *VPINN / hp-VPINNs* (variational form, domain decomposition). CMAME 374:113547.
- **Jagtap et al. (2020)** — *cPINN* (conservative, discrete domains) & *XPINN* (space-time domain decomposition).
- **Meng et al. (2020)** — *PPINN* (parareal, time-parallel). CMAME 370:113250.
- **Pang, Lu, Karniadakis (2019)** — *fPINNs* (fractional PDEs). SIAM J. Sci. Comput. 41(4).
- **Yang, Zhang, Karniadakis (2020)** — *PI-GANs* (stochastic PDEs / UQ). SIAM J. Sci. Comput. 42(1).
- **Gladstone, Nabian, Meidani (2022)** — *FO-PINNs* (first-order formulation → lower-order autograd). arXiv:2210.14320.

## Convergence & error theory
- **Shin, Darbon, Karniadakis (2020)** — convergence for linear 2nd-order elliptic/parabolic PDEs. arXiv:2004.01806.
- **Mishra & Molinaro (2022, 2023)** — generalization-error estimates for PINNs (forward & inverse). IMA J. Numer. Anal.

## Operator learning (learn a *family* of solutions — the "next level" of parametric PINNs)
- **Lu, Jin, Pang, Zhang, Karniadakis (2021)** — *Learning nonlinear operators via DeepONet.* **Nature
  Machine Intelligence 3:218–229.** → branch/trunk nets map input *functions* → output functions.
- **Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar (2021)** — *Fourier Neural
  Operator (FNO) for parametric PDEs.* **ICLR 2021.** → convolution in Fourier space; zero-shot
  super-resolution; first ML method to model turbulent flow as an operator.
  > Relevance to this repo: our §2.6 "parametric" PINN `(x,y,α)→ψ` is a *poor-man's* operator; DeepONet/FNO
  > are the principled way to amortize over a family (here, over angle of attack / geometry / Re).

## Fluid dynamics & heat applications
- **Jin, Cai, Li, Karniadakis (2021)** — *NSFnets: PINNs for the incompressible Navier–Stokes equations.*
  J. Comput. Phys. 426. → velocity–pressure & vorticity–stream-function formulations (cf. our §2.4–2.6).
- **Rao, Sun, Liu (2020)** — PINNs for incompressible laminar flows. Theor. Appl. Mech. Lett. 10(3):207–212.
- **Cai, Wang, Wang, Perdikaris, Karniadakis (2021)** — PINNs for heat-transfer problems. J. Heat Transfer 143(6):060801.
- **Sun, Gao, Pan, Wang (2020)** — surrogate flow modeling with physics constraints, **no simulation data.** CMAME 361.
- **Raissi, Yazdani, Karniadakis (2020)** — *Hidden Fluid Mechanics.* Science 367(6481). → reconstruct
  velocity & pressure fields from flow-visualization data (data assimilation — PINNs' real edge).
- **Ding, Feng, Lou, Fu, Li, Zhang, Ma, Zhang (2025)** — *Prediction of velocity and pressure of
  gas-liquid flow using spectrum-based physics-informed neural networks (SP-PINN).* **Applied Mathematics
  and Mechanics (English Ed.) 46(2), 341–356**, DOI 10.1007/s10483-025-3217-8. → **two-phase (gas-liquid)**
  flow; folds **spectral analysis** + a **pressure-correction module** into the Navier–Stokes residual to
  sharpen predictions at gas-phase boundaries and velocity peaks (error ~1‰ vs plain PINN, inference
  <0.01 s). Ties to the **spectral-bias** remedy (Fourier features) — see the playbook.
- **Wei, Fan, Ooi, Wong, Wang, Chiu (2026)** — *Bridging CFD Algorithm and Physics-Informed Learning:
  SIMPLE-PINN for Incompressible Navier–Stokes Equations.* **arXiv:2603.24013.** → embeds the classical
  **SIMPLE** pressure–velocity-coupling algorithm (the pressure-correction scheme at the heart of most
  incompressible CFD solvers) into the PINN loss: a derived **velocity–pressure coupling correction
  loss** enables precise **data-free** simulation (validated on lid-driven cavity). A clean example of
  *bridging a classical numerical method with a PINN* — see the playbook (formulation choices).
- **Wei, Fan, Wong, Ooi, Wang, Chiu (2026)** — *FFV-PINN: A Fast Physics-Informed Neural Network with
  Simplified Finite Volume Discretization and Residual Correction.* **arXiv:2603.24114.** → tackles a
  core PINN weakness — pointwise autodiff **ignores neighboring points**. FFV-PINN uses a **simplified
  finite-volume (FVM)** discretization of the **convection term** (a main instability source) plus a
  **residual-correction** loss → better dispersion/dissipation, more stable training for complex flows.
  Same group as SIMPLE-PINN; see the "classical CFD × PINN" theme note.

## Software / libraries
- **DeepXDE** (Python; Lu et al. 2021) — the standard PINN library (this repo's Phase-2 roadmap targets it).
- **NVIDIA Modulus** — industrial PINN/operator framework (formerly SimNet).
- **NeuralPDE.jl** (Julia / SciML; Zubov et al. 2021, arXiv:2107.09443) — automated PINNs.
- **jaxpi** (JAX; Wang et al. 2023) — reference implementation of the "Expert's Guide" best practices.

## Local copies (Patxi's machine, not committed — copyrighted)
- `Downloads/Exploring_Physics-Informed_Neural_Networks_From_Fu.pdf` — Ganga & Uddin review (anchor).
- `Downloads/introduction_pinn.pdf` — P. Kestener (CEA), *Introduction to PINNs*, MS-HPC-IA Sophia-Antipolis, 2022 (good beginner slides).
