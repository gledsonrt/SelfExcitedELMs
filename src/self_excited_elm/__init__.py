"""Flat-plate self-excited-force modelling with extreme learning machines."""

from .elm import ExtremeLearningMachine
from .flat_plate import (
    FlatPlateDataset,
    flat_plate_derivatives,
    generate_flat_plate_dataset,
    get_derivatives,
    harmonic_heave,
    harmonic_pitch,
    unsteady_features,
    unsteady_time_domain,
)

__all__ = [
    "ExtremeLearningMachine", "FlatPlateDataset", "flat_plate_derivatives",
    "generate_flat_plate_dataset", "get_derivatives", "harmonic_heave",
    "harmonic_pitch", "unsteady_features", "unsteady_time_domain",
]
