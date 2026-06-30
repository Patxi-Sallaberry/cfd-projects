# Running the parametric PINN on a GPU (RunPod)

The training script `src/pinn_flow_airfoil_parametric_psi_gpu.py` is **device-agnostic**: it uses a
CUDA GPU when one is available and falls back to CPU otherwise (see README §2.6, *GPU-ready*). This dev
machine has no NVIDIA GPU, so to get the speedup we rent one for a few minutes on **RunPod** (on-demand
NVIDIA GPUs). A full 30 000-epoch run costs a few **cents**.

## Steps

1. **Account & credit** — sign up at runpod.io and add a small amount of credit ($5 is plenty).

2. **Deploy a GPU Pod**
   - *Community Cloud* (cheapest), pick the cheapest GPU available — an **RTX 3090 / A4000 / A5000**
     (~$0.2–0.4/h) is *more* than enough; this PINN is tiny, no need for an A100.
   - Template: **RunPod PyTorch** (it ships CUDA + a CUDA-enabled `torch` already — don't reinstall torch).

3. **Connect** — open **JupyterLab** or the **Web Terminal** from the pod's *Connect* menu.

4. **Check the GPU is visible**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

5. **Get the code and run** (the repo is public → no auth needed)
   ```bash
   git clone https://github.com/Patxi-Sallaberry/cfd-projects
   cd cfd-projects/piml/phase2_pinns
   pip install -r requirements.txt          # torch is already there; this adds numpy/matplotlib
   python src/pinn_flow_airfoil_parametric_psi_gpu.py
   ```
   The first line printed should read `[device] entrainement sur : cuda (NVIDIA RTX 3090)` and the run
   finishes in a few minutes instead of ~50 min on CPU.

6. **Collect the results** — download
   `results/figures/pinn_flow_airfoil_parametric_psi_gpu.png` (right-click → *Download* in JupyterLab)
   and copy the `Cl(α)` table from the console. The physics is identical to the CPU model, so the
   numbers match — this run is about **speed**, not a different result.

7. **⚠️ Terminate the pod** when done — RunPod bills for as long as the pod is *running* (even idle).
   *Stop* then *Terminate* it from the dashboard. Download anything you need first (pod storage is
   ephemeral unless you attach a network volume).

## Notes
- To **commit the GPU-produced figure** back to the repo from the pod, you'd need a GitHub token; it's
  simpler to download the PNG and add it locally.
- Because the GPU makes runs cheap, you *could* scale the model up (wider network / more collocation) by
  editing the constants at the top of the script — but recall (README §2.6) the lift-magnitude gap is an
  intrinsic limit of potential-flow PINNs, not something more compute fixes. The right tool for exact
  inviscid lift stays the panel method (`src/panel_method.py`).
