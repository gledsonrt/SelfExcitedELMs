"""Shared example helpers; these use only standard Matplotlib facilities."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

COLORS = {"blue": "#1775b5", "red": "#d04a3a", "grey": "#9aa0a6"}

def style():
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 300, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .25, "axes.spines.top": True,
                         "axes.spines.right": True, "legend.frameon": False})

def output_path(name):
    OUT.mkdir(exist_ok=True)
    return OUT / name

def training_data(variant="log", compact=False):
    """Generate a reproducible training dataset for an example variant."""
    from self_excited_elm import generate_flat_plate_dataset
    if variant == "base":
        vrs = np.r_[np.linspace(.5, 20, 20), np.geomspace(.25, 20, 20)]
        amplitudes = np.c_[np.linspace(np.finfo(float).eps, .85, 50), np.linspace(np.finfo(float).eps, .12, 50)]
        return generate_flat_plate_dataset(vrs=vrs, amplitudes=amplitudes, steps_per_cycle=16 if compact else 25)
    vrs = np.geomspace(1, 50, 25 if compact else 100) if variant == "log" else np.linspace(1, 50, 25 if compact else 100)
    amplitudes = np.c_[np.linspace(np.finfo(float).eps, .373, 15 if compact else 50), np.linspace(np.finfo(float).eps, .0302, 15 if compact else 50)]
    return generate_flat_plate_dataset(rho=1.2, u=10., b=31., dt=.01, vrs=vrs, amplitudes=amplitudes, steps_per_cycle=20 if compact else 25)

def train_models(variant="log", n_hidden=490, compact=False, include_reduced_velocity=False):
    """Generate a dataset and fit the four ELMs without reading or writing model files."""
    from self_excited_elm import ExtremeLearningMachine
    data = training_data(variant, compact)
    h_inputs, a_inputs = np.c_[data.h, data.hdot, data.hddot], np.c_[data.a, data.adot, data.addot]
    if include_reduced_velocity:
        h_inputs, a_inputs = np.c_[h_inputs, data.h_vr], np.c_[a_inputs, data.a_vr]
    specifications = ((h_inputs, data.flh), (h_inputs, data.fmh),
                      (a_inputs, data.fla), (a_inputs, data.fma))
    models = [ExtremeLearningMachine(specifications[0][0].shape[1], n_hidden).fit(x, y) for x, y in specifications]
    return models, data.amp_h, data.amp_a, data
