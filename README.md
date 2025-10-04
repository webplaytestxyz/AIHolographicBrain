AI Holographic Brain
====================

A small interactive toy demonstrating a "holographic" 3D field of activations that evolves over time and can be seeded by user prompts. The program visualizes active nodes as a 3D dot cloud using PyVista, allows clicking to excite nodes, and accepts typed prompts (press `P`) which are deterministically mapped to a seed in the field. The system then evolves the field and chooses a "stable attractor" as the output; a short canned response is shown and the winning attractor node is highlighted.

This repository contains a single main script: `aibrain.py`.

Quick summary
-------------
- Field data structure: a PyTorch tensor with shape (X, Y, Z, 2) where the last channel holds (amplitude, phase).
- Dynamics: a local Laplacian-based coupling + nonlinearity + decay that produces localized bumps (attractors) under the right parameters.
- Prompt injection: typed prompts are hashed deterministically to a (x,y,z) seed which is boosted; the dynamics are run and the final attractor is mapped to a canned string response.
- Visualization: PyVista renders the dot cloud. The attractor is highlighted and a response text is displayed in the window.

Files
-----
- `aibrain.py` — main program (interactive). This file contains the simulation, prompt injection, visualization and UI bindings.

How it maps to the concept
--------------------------
- Inputs: optional prompt text or clicks on the 3D cloud.
- Internal state: 3D field of amplitudes and phases (torch.Tensor).
- Processing: iterative dynamics that can converge to attractor states.
- Output: the attractor coordinate (or converged field pattern) mapped deterministically to a human-readable token/response.

Requirements (recommended)
--------------------------
This project runs best in an Anaconda/Miniconda environment on Windows 10/11. The main libraries are PyTorch (CPU builds are fine), PyVista, PyVistaQt, VTK and a Qt binding (PyQt5 or PyQt6).

Recommended conda commands (Windows PowerShell)
------------------------------------------------
# create and activate a fresh env (recommended)
conda create -n aibrain python=3.10 -y; conda activate aibrain

# install PyVista and Qt (conda-forge is more consistent on Windows)
conda install -c conda-forge pyvista=0.46.3 pyvistaqt vtk pyqt -y

# install PyTorch (CPU build) and related packages (use the CPU wheel if you don't have a matching CUDA setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Other pip installs (if you prefer pip for everything):
# pip install pyvista==0.46.3 pyvistaqt vtk PyQt5

Quick pip alternative (if not using conda)
------------------------------------------
pip install numpy torch pyvista==0.46.3 pyvistaqt vtk PyQt5

Running
-------
From your project folder (PowerShell):

python .\aibrain.py --size 24 --max-iters 300

CLI flags
---------
- `--size`: integer (default 20) — size of the cubic field (X=Y=Z=size).
- `--threshold`: float — amplitude threshold for which nodes are visualized.
- `--max-iters`: integer — max animation iterations.

Interactive controls (in the PyVista window)
--------------------------------------------
- Click nodes (left-click) — excites the nearest node and updates the cloud.
- Press `P` — opens a small text dialog (requires Qt) to type a prompt. The prompt is hashed and injected as a seed. The program evolves and selects an attractor; the response and attractor marker appear in the window.

Behavior and UX
---------------
- The program attempts to be robust across PyVista versions and environment differences. It will try to use the best available picker and timers for animation. If a Qt dialog is not available the script prints a console hint.
- The upper-left of the PyVista window shows the last response text. The lower-left shows a persistent hint: "Press 'P' to prompt the AI". A short response history is shown in the lower-right. The chosen attractor is highlighted with a colored sphere (green if considered stable, red if unstable).

Common issues & troubleshooting
-------------------------------
1. PyVista window doesn't appear or errors about Qt
   - Ensure `pyvistaqt` and a Qt binding (PyQt5/PyQt6) are installed. The conda-forge approach above is the most reliable on Windows.

2. OMP / libiomp5md.dll errors ("multiple copies of the OpenMP runtime")
   - This is a common conflict when mixing packages (PyTorch, NumPy/MKL, VTK) that bring different OpenMP runtimes. Two safe options:
     - Create a clean conda environment and install the packages from conda-forge where possible (minimizes runtime conflicts).
     - As a pragmatic local workaround (not recommended for production) set these environment variables before running:

       # PowerShell (session only):
       $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; python .\aibrain.py

     - Or persist them for your account (Windows):
       setx KMP_DUPLICATE_LIB_OK TRUE
       setx OMP_NUM_THREADS 1
       setx MKL_NUM_THREADS 1

3. Clicking rotates but doesn't pick
   - Ensure a Qt event loop is present (`pyvistaqt`/`PyQt5` installed) and that the Plotter's picker method is compatible. The script tries `use_picker=True` and falls back to other signatures.

4. Colors change or nodes appear purple
   - The color mapping is governed by the amplitude scalar range. The script attempts to set a stable scalar range (0–1) to prevent dramatic colormap jumps. If you customize dynamics heavily, you may want to re-normalize amplitudes before visualization.

Notes on deterministic responses
-------------------------------
- The current demo maps attractor coordinates + the prompt hash to a small set of canned responses. This is intentionally simple and deterministic so the mapping is reproducible.
- For a production mapping you could: store a dictionary of coordinate -> token, compute an embedding of the converged field and run a nearest-neighbor classifier, or train a small MLP that maps attractor patterns → tokens.

Developer notes
---------------
- `aibrain.py` key functions:
  - `build_field(size)` — creates the tensor field
  - `inject_prompt(field, coords)` — boosts amplitude/phase at coords
  - `_laplacian3d(amplitude)` — computes local coupling (via conv3d kernel)
  - `update_step(...)` & `evolve_until_convergence(...)` — field dynamics
  - `extract_attractor(field)` — returns argmax coordinates and amplitude
  - `launch_visualization(field, ...)` — all the PyVista plumbing, pick callbacks and prompt mapping

- Tuning parameters to experiment with:
  - `coupling` (Laplacian strength)
  - `decay` (amplitude damping)
  - `noise_scale` (stochasticity)
  - `threshold` (visualization cutoff)

Suggested next features
-----------------------
- Temporal attractor persistence: require the same attractor to persist for N consecutive frames before accepting as a stable answer.
- Replace canned answers with a small dictionary or a nearest-neighbor mapping from a pooled attractor embedding to tokens.
- Add on-plot GUI sliders to tune `coupling/decay/noise` at runtime.
- Add an option to save snapshots (PNG) or record a short GIF of field evolution for offline inspection.

License & acknowledgements
--------------------------
- This is a demo / research toy. No license file included by default — add one if you plan to publish or share widely.

If you want, I can also:
- Generate an `environment.yml` you can use with `conda env create -f environment.yml` that pins versions known to work together on Windows/Anaconda.
- Add an example `requirements.txt` for pip-only setups.
- Implement a temporal stability criterion and an on-plot prompt button (avoids keyboard dependence).

---

If you'd like the `environment.yml`, tell me whether you prefer conda-forge packages for PyVista/VTK or pip wheels for PyTorch (GPU), and I will prepare it and add it to the repo.