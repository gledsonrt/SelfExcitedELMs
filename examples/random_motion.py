"""Compare ELM predictions for a prescribed random-motion input."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from self_excited_elm import ExtremeLearningMachine, unsteady_time_domain
from common import COLORS, output_path, style, training_data


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
MOTION_FILE = DATA_DIRECTORY / "random_motion.csv"

RHO = 1.2
U = 20.0
B = 31.0
DT = 0.00775
PREDICTION_SAMPLES = 25_000


def load_motion():
    """Load the prescribed motion record used for the force comparison."""
    motion = np.loadtxt(MOTION_FILE, delimiter=",", skiprows=1)
    selected = motion[:PREDICTION_SAMPLES]
    return selected[:, 1], selected[:, 2]


def train_models(rho, u, b, dt):
    """Fit component ELMs from the harmonic flat-plate benchmark dataset."""
    if (rho, u, b, dt) != (1.2, 20.0, 31.0, 0.00775):
        raise ValueError("Random-motion and harmonic-training parameters must match.")

    data = training_data("base")
    heave_inputs = np.c_[data.h, data.hdot, data.hddot]
    pitch_inputs = np.c_[data.a, data.adot, data.addot]

    def fit(inputs, target):
        return ExtremeLearningMachine(inputs.shape[1], 490).fit(inputs, target)

    return (
        fit(heave_inputs, data.flh),
        fit(pitch_inputs, data.fla),
        fit(heave_inputs, data.fmh),
        fit(pitch_inputs, data.fma),
    )


def main():
    style()
    heave, pitch = load_motion()
    models = train_models(RHO, U, B, DT)

    _, _, time, l_h, l_a, _, m_h, m_a, _ = unsteady_time_domain(RHO, U, B, DT, heave, pitch)
    lift_scale = 0.5 * RHO * U**2 * B
    moment_scale = lift_scale * B
    references = (l_h / lift_scale, l_a / lift_scale, m_h / moment_scale, m_a / moment_scale)

    features_h = np.c_[heave, np.gradient(heave, DT), np.gradient(np.gradient(heave, DT), DT)]
    features_a = np.c_[pitch, np.gradient(pitch, DT), np.gradient(np.gradient(pitch, DT), DT)]
    predictions = (
        models[0].predict(features_h),
        models[1].predict(features_a),
        models[2].predict(features_h),
        models[3].predict(features_a),
    )

    reduced_time = time * U / B
    index = np.flatnonzero(reduced_time <= 100.0)[::10]
    panels = (
        (heave, r"$h$ [m]", None, (-0.8, 0.8), np.arange(-0.8, 0.81, 0.4)),
        (np.rad2deg(pitch), r"$\alpha$ [deg]", None, (-0.5, 0.5), np.arange(-0.5, 0.51, 0.25)),
        (references[0], r"$C_{L,h}$ [-]", predictions[0], (-0.08, 0.08), np.arange(-0.08, 0.081, 0.04)),
        (references[1], r"$C_{L,\alpha}$ [-]", predictions[1], (-0.06, 0.06), np.arange(-0.06, 0.061, 0.03)),
        (references[2], r"$C_{M,h}$ [-]", predictions[2], (-0.02, 0.02), np.arange(-0.02, 0.021, 0.01)),
        (references[3], r"$C_{M,\alpha}$ [-]", predictions[3], (-0.02, 0.02), np.arange(-0.02, 0.021, 0.01)),
    )

    figure, axes = plt.subplots(3, 2, figsize=(8, 7), sharex=True, constrained_layout=True)
    for axis, (reference, label, prediction, y_limits, y_ticks) in zip(axes.flat, panels):
        axis.plot(reduced_time[index], reference[index], color=COLORS["blue"], lw=1.2, label="Analytical")
        if prediction is not None:
            axis.plot(reduced_time[index], prediction[index], color=COLORS["red"], lw=1.0, label="ELM")
        axis.set(xlim=(0, 100), xticks=np.arange(0, 101, 20), ylim=y_limits, yticks=y_ticks, ylabel=label)

    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel(r"Reduced time, $tU/B$ [-]")
    axes[-1, 1].set_xlabel(r"Reduced time, $tU/B$ [-]")
    figure.savefig(output_path("flat_plate_random_motion.pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
