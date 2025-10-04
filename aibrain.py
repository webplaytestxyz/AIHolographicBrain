"""AI Holographic Brain (refactored)

Key changes:
- safer .cpu().numpy() conversions
- simple 3D Laplacian coupling implemented via conv3d to enable attractors
- convergence detection (L2 norm) with early stop
- robust PyVista usage (handles empty point sets and prints helpful errors on GUI issues)
"""

import os
import argparse
import sys
import time
from typing import List, Tuple

# Workaround for mixing OpenMP runtimes (may occur with PyTorch, VTK, MKL). This is an unsafe
# but pragmatic mitigation for local interactive use on Windows/Anaconda. Prefer creating a
# clean conda env with compatible builds if you see instability.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Limit OpenMP threads to reduce concurrency-related issues in mixed-runtime setups
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv


def build_field(field_size: int = 20, device: str = "cpu") -> torch.Tensor:
    """Create a random field tensor with shape (X,Y,Z,2) where last dim is (amplitude, phase)."""
    return torch.rand(field_size, field_size, field_size, 2, device=device)


def inject_prompt(field: torch.Tensor, coords_list: List[Tuple[int, int, int]], amp_boost: float = 1.0,
                  phase_shift: float = np.pi / 4):
    """Add amplitude and phase to coordinates in-place."""
    for (x, y, z) in coords_list:
        if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and 0 <= z < field.shape[2]:
            field[x, y, z, 0] += amp_boost
            field[x, y, z, 1] += phase_shift


def _laplacian3d(amplitude: torch.Tensor) -> torch.Tensor:
    """Compute 3D discrete Laplacian using conv3d (expects amplitude shape D,H,W)."""
    # amplitude: (D,H,W) -> conv3d needs (N,C,D,H,W)
    a = amplitude.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=amplitude.dtype, device=amplitude.device)
    # 6-neighbor Laplacian
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0

    lap = F.conv3d(a, kernel, padding=1)
    return lap.squeeze(0).squeeze(0)


def evolve_until_convergence(field: torch.Tensor,
                             coupling: float = 0.2,
                             decay: float = 0.95,
                             noise_scale: float = 0.01,
                             max_iters: int = 500,
                             eps: float = 1e-4,
                             stable_steps: int = 5) -> Tuple[torch.Tensor, int]:
    """Iterate field dynamics until convergence (L2 change < eps for stable_steps consecutive iterations).

    Returns the converged field and number of iterations run.
    """
    amp = field[..., 0]
    phase = field[..., 1]

    prev = amp.clone()
    stable_count = 0
    for i in range(int(max_iters)):
        # diffusion/coupling via Laplacian
        lap = _laplacian3d(amp)
        amp = amp + coupling * lap

        # pointwise nonlinearity and small noise
        amp = torch.tanh(amp + torch.randn_like(amp) * noise_scale)

        # decay
        amp = amp * decay

        # write back
        field[..., 0] = amp

        # phase can drift slightly
        phase = phase + (torch.randn_like(phase) * (noise_scale * 0.1))
        field[..., 1] = phase

        # convergence test
        delta = torch.norm((amp - prev).view(-1), p=2).item()
        if delta < eps:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_steps:
            return field, i + 1

        prev = amp.clone()

    return field, int(max_iters)


def _get_active_nodes_and_colors(field: torch.Tensor, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Return Nx3 float32 points and N-length colors array for amplitudes > threshold."""
    amp = field[..., 0].cpu().numpy()
    active_idx = np.argwhere(amp > threshold)
    if active_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    colors = amp[active_idx[:, 0], active_idx[:, 1], active_idx[:, 2]].astype(np.float32)
    return active_idx.astype(np.float32), colors


def launch_visualization(field: torch.Tensor,
                         threshold: float = 0.1,
                         interval: int = 50,
                         max_iters: int = 200):
    """Create a PyVista interactive visualization and animate the field until convergence.

    This function is defensive: if PyVista cannot open an interactive window (common on headless servers),
    it will print instructions instead of crashing.
    """
    try:
        # Detect Qt availability early so we can initialize QApplication before creating widgets/timers
        Qt_available = False
        QtWidgets = None
        QtCore = None
        try:
            from PyQt5 import QtWidgets, QtCore  # type: ignore
            Qt_available = True
        except Exception:
            try:
                from PyQt6 import QtWidgets, QtCore  # type: ignore
                Qt_available = True
            except Exception:
                Qt_available = False

        # If Qt is available, ensure a QApplication exists before creating any QWidget/QTimer
        if Qt_available:
            app = QtWidgets.QApplication.instance()
            if app is None:
                _app = QtWidgets.QApplication(sys.argv)

        plotter = pv.Plotter()
        plotter.background_color = "black"

        points, colors = _get_active_nodes_and_colors(field, threshold=threshold)

        # Create initial point cloud; if no points, create empty PolyData
        if points.shape[0] == 0:
            point_cloud = pv.PolyData(np.zeros((1, 3), dtype=np.float32))
            plotter.add_points(point_cloud, color="black", point_size=1)
        else:
            point_cloud = pv.PolyData(points)
            plotter.add_points(point_cloud, scalars=colors, render_points_as_spheres=True, point_size=10,
                               name="brain_cloud")

        def update_point_cloud():
            try:
                pts, cols = _get_active_nodes_and_colors(field, threshold=threshold)
                if pts.shape[0] == 0:
                    point_cloud.points = np.zeros((1, 3), dtype=np.float32)
                    try:
                        point_cloud.point_data.clear()
                    except Exception:
                        pass
                else:
                    point_cloud.points = pts
                    try:
                        point_cloud.point_data['Amplitude'] = cols
                    except Exception:
                        pass

                # Safe render wrapper
                try:
                    plotter.render()
                except Exception as e:
                    print("Warning: plotter.render() failed:", e)
            except Exception as e:
                import traceback
                print("Exception in update_point_cloud:")
                traceback.print_exc()


        def _extract_point_from_args(args):
            # Try to find a 3-element numeric sequence in args or use picker's GetPickPosition
            if not args:
                return None
            for a in args:
                try:
                    # VTK pickers may have GetPickPosition
                    if hasattr(a, 'GetPickPosition'):
                        pos = a.GetPickPosition()
                        return np.array(pos[:3], dtype=float)
                except Exception:
                    pass
                try:
                    arr = np.asarray(a)
                    if arr is not None and arr.size >= 3:
                        return arr[:3].astype(float)
                except Exception:
                    pass
            return None


        def click_callback(*args):
            # Accept flexible callback signatures from different PyVista versions
            try:
                point = _extract_point_from_args(args)
                if point is None:
                    return
                # convert to nearest integer grid coordinate and clip
                coords = np.round(point).astype(int)
                x = int(np.clip(coords[0], 0, field.shape[0] - 1))
                y = int(np.clip(coords[1], 0, field.shape[1] - 1))
                z = int(np.clip(coords[2], 0, field.shape[2] - 1))
                field[x, y, z, 0] += 1.0
                field[x, y, z, 1] += np.pi / 4
                print(f"Node excited at {x},{y},{z}")
                update_point_cloud()
            except Exception as e:
                import traceback
                print("Exception in click_callback:")
                traceback.print_exc()
                return

        # Use use_picker=True to pick anywhere in the scene and show_point to visualize pick
        try:
            plotter.enable_point_picking(callback=click_callback, show_message=False, use_picker=True, show_point=True)
        except TypeError:
            # older/newer pyvista may use different signature
            try:
                plotter.enable_point_picking(callback=click_callback, show_message=False, use_mesh=False, show_point=True)
            except Exception:
                plotter.enable_point_picking(callback=click_callback, show_message=False)

        iteration = 0

        def animation_callback():
            nonlocal iteration
            if iteration >= max_iters:
                return

            # run a few evolution steps per callback to speed convergence
            evolve_until_convergence(field, max_iters=5)

            points, colors = _get_active_nodes_and_colors(field, threshold=threshold)

            if points.shape[0] == 0:
                # hide or set an offscreen point
                try:
                    point_cloud.points = np.zeros((1, 3), dtype=np.float32)
                    point_cloud.point_data.clear()
                except Exception:
                    pass
            else:
                point_cloud.points = points
                point_cloud.point_data['Amplitude'] = colors

            iteration += 1

        # Prompt injection via keyboard: press 'p' to type a prompt which will deterministically map to a seed
        def prompt_callback(_=None):
            # Try Qt input dialog first
            prompt_text = None
            try:
                from PyQt5.QtWidgets import QInputDialog
                parent = getattr(plotter, 'app_window', None)
                text, ok = QInputDialog.getText(parent, "Inject prompt", "Enter prompt text:")
                if ok and text:
                    prompt_text = str(text)
            except Exception:
                try:
                    from PyQt6.QtWidgets import QInputDialog
                    parent = getattr(plotter, 'app_window', None)
                    text, ok = QInputDialog.getText(parent, "Inject prompt", "Enter prompt text:")
                    if ok and text:
                        prompt_text = str(text)
                except Exception:
                    # No Qt available — can't prompt interactively without blocking the GUI
                    print("Interactive prompt unavailable: install PyQt5/PyQt6 and pyvistaqt for GUI prompts.")
                    prompt_text = None

            if not prompt_text:
                print("No prompt provided.")
                return

            # deterministically map prompt text to coordinates using a hash
            import hashlib

            h = hashlib.sha256(prompt_text.encode("utf-8")).digest()
            # Use bytes to produce three integers
            x = h[0] % field.shape[0]
            y = h[1] % field.shape[1]
            z = h[2] % field.shape[2]
            inject_prompt(field, [(x, y, z)], amp_boost=2.0)
            print(f"Injected prompt '{prompt_text}' -> seed at {x},{y},{z}")
            update_point_cloud()

            # Let the field evolve after prompt injection to attempt to settle to an attractor
            field_after, nit = evolve_until_convergence(field, max_iters=100)
            coords_attractor, val = extract_attractor(field_after)

            # Map attractor -> deterministic canned response
            def map_attractor_to_response(coords, prompt_text):
                responses = [
                    "I am an AI holographic brain.",
                    "My name is John.",
                    "I am here to help.",
                    "I process holographic attractors to produce answers.",
                    "I am a toy demo — answers are illustrative."
                ]
                import hashlib
                h = hashlib.sha256(prompt_text.encode('utf-8')).digest()
                seed = (coords[0] * 73856093) ^ (coords[1] * 19349663) ^ (coords[2] * 83492791) ^ int.from_bytes(h[:4], 'little')
                idx = int(seed % len(responses))
                return responses[idx]

            response = map_attractor_to_response(coords_attractor, prompt_text)
            print(f"Attractor: {coords_attractor} value={val:.4f} -> response: {response}")

            # Stability/dominance check: ensure attractor amplitude is dominant among top k
            amps = field_after[..., 0].cpu().numpy().ravel()
            topk = np.sort(amps)[-3:]
            dominant_ratio = float(val) / (float(topk.sum()) + 1e-8)

            # Always show a response in the GUI; mark it as unstable if dominance is low
            if dominant_ratio < 0.5:
                gui_msg = f"Answer (unstable): {response}"
            else:
                gui_msg = f"Answer: {response}"

            # Display GUI feedback: remove prior response/marker if present
            try:
                # Remove help text if present so response text doesn't overlap
                try:
                    if hasattr(plotter, '_help_text') and plotter._help_text is not None:
                        try:
                            plotter.remove_actor(plotter._help_text)
                        except Exception:
                            pass
                        plotter._help_text = None
                except Exception:
                    pass

                if hasattr(plotter, '_response_text') and plotter._response_text is not None:
                    try:
                        plotter.remove_actor(plotter._response_text)
                    except Exception:
                        pass
                # add_text returns the vtk actor; store it and render
                plotter._response_text = plotter.add_text(gui_msg, position='upper_left', color='white', font_size=12)
                try:
                    plotter.render()
                except Exception:
                    pass

                # Maintain a short response history shown in the lower-right
                try:
                    if not hasattr(plotter, '_response_history'):
                        plotter._response_history = []
                    # store newest at top
                    plotter._response_history.insert(0, f"{time.strftime('%H:%M:%S')} - {response}")
                    plotter._response_history = plotter._response_history[:5]
                    # remove previous history actor
                    try:
                        if hasattr(plotter, '_history_text') and plotter._history_text is not None:
                            plotter.remove_actor(plotter._history_text)
                    except Exception:
                        pass
                    history_text = "\n".join(plotter._response_history)
                    plotter._history_text = plotter.add_text(history_text, position='lower_right', color='white', font_size=10)
                    try:
                        plotter.render()
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                # fallback: print to console if GUI text fails
                print(gui_msg)

            # Highlight attractor in the scene
            try:
                # remove previous marker
                if hasattr(plotter, '_attractor_actor') and plotter._attractor_actor is not None:
                    try:
                        plotter.remove_actor(plotter._attractor_actor)
                    except Exception:
                        pass

                ax, ay, az = coords_attractor
                sphere = pv.Sphere(radius=0.6, center=(float(ax), float(ay), float(az)))
                # color by stability
                marker_color = 'green' if dominant_ratio >= 0.5 else 'red'
                plotter._attractor_actor = plotter.add_mesh(sphere, color=marker_color, name='attractor_marker')
                plotter.render()
            except Exception:
                pass

            # Boost the attractor's amplitude to visually emphasize the chosen node in data
            ax, ay, az = coords_attractor
            inject_prompt(field, [(ax, ay, az)], amp_boost=3.0)
            update_point_cloud()

        # register key event if supported
        try:
            plotter.add_key_event('p', prompt_callback)
        except Exception:
            # some plotter builds may not have add_key_event; ignore silently
            pass

        # Store help text actor so we can remove it when showing responses
        try:
            plotter._help_text = plotter.add_text("Click nodes to excite; field evolves automatically", color="white")
        except Exception:
            plotter._help_text = None

        # Persistent hint for prompting
        try:
            plotter._hint_text = plotter.add_text("Press 'P' to prompt the AI", position='lower_left', color='yellow', font_size=10)
        except Exception:
            plotter._hint_text = None

        # PyVista versions differ: some have Plotter.add_callback, older/newer may not.
        if hasattr(plotter, 'add_callback') and callable(getattr(plotter, 'add_callback')):
            plotter.add_callback(animation_callback, interval=interval)
            plotter.show()
        else:
            # Fallback: use a Qt QTimer if pyvistaqt / PyQt5 are available
                try:
                    if not Qt_available:
                        raise RuntimeError("Qt not available")

                    # Ensure QApplication exists
                    app = QtWidgets.QApplication.instance()
                    if app is None:
                        app = QtWidgets.QApplication(sys.argv)

                    timer = QtCore.QTimer()
                    # Keep a reference so it doesn't get GC'd
                    plotter._qt_timer = timer
                    timer.timeout.connect(animation_callback)
                    timer.start(interval)

                    plotter.show()
                except Exception as e:
                    # If Qt isn't available, raise a helpful error
                    raise RuntimeError(
                        "PyVista Plotter does not support add_callback and no Qt timer could be created. "
                        "Install 'pyvistaqt' and 'PyQt5' (or PyQt6) for interactive animation on Windows/Anaconda."
                    ) from e
    except Exception as e:
        print("Could not open an interactive PyVista window.")
        print("If you're on Windows/Anaconda ensure you have 'pyvistaqt' and a Qt package installed (PyQt5/PyQt6).")
        print("Error:", e)


def extract_attractor(field: torch.Tensor) -> Tuple[Tuple[int, int, int], float]:
    """Return the coordinate of the maximum amplitude and its value.

    Uses unravel_index so we get (x,y,z) rather than a flattened index.
    """
    amp = field[..., 0]
    flat_idx = torch.argmax(amp)
    coords = tuple(int(x) for x in torch.unravel_index(flat_idx, amp.shape))
    value = float(amp[coords].item())
    return coords, value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20, help="field size (cube)")
    parser.add_argument("--threshold", type=float, default=0.1, help="visualization amplitude threshold")
    parser.add_argument("--max-iters", type=int, default=200, help="max animation iterations")
    args = parser.parse_args(argv)

    field = build_field(args.size)

    # seed with a small prompt in the center
    center = args.size // 2
    inject_prompt(field, [(center, center, center), (center - 1, center, center + 1)])

    # run a longer convergence to reach an attractor before visualizing
    print("Evolving field to find attractor...")
    field, iters = evolve_until_convergence(field, max_iters=500)
    print(f"Converged (or reached max) after {iters} iterations")

    coords, val = extract_attractor(field)
    print("Stable attractor node:", coords, "value:", val)
    print("Output token:", "AI – My name is John")

    # launch visualization (interactive) — may require pyvistaqt/Qt on Windows
    launch_visualization(field, threshold=args.threshold, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
