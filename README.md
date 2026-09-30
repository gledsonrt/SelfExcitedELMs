# Self-Excited ELM: flat-plate benchmark

A standalone Python implementation of an Extreme Learning Machine (ELM)
framework for modelling self-excited aerodynamic loads on a two-dimensional
flat plate. It uses NumPy, SciPy, and Matplotlib only, and is designed to run
from a fresh clone: the examples generate their data, train ELMs, make
predictions, and write publication-quality figures.

## Relation to the paper

This repository accompanies the benchmark presented in
[*Modelling non-linear aeroelastic loads in long-span bridges with extreme
learning machines*](https://www.emerald.com/jbren/article-abstract/doi/10.1680/jbren.25.00003/1393399/Modelling-non-linear-aeroelastic-loads-in-long).
The paper investigates ELMs as efficient surrogate models for non-linear,
motion-induced aerodynamic loads in long-span bridges. It evaluates the
approach using an analytical flat-plate problem and bridge-deck data. A preprint of the paper is available at [*arXiv*](https://arxiv.org/abs/2609.33274).

The code here provides the fully runnable, analytical flat-plate part of that
study. It is intended as a compact and reproducible reference implementation:
it generates harmonic-motion training data, fits the ELM force models,
identifies aerodynamic derivatives, and evaluates the models for harmonic and
prescribed random motion. The codes can be directly applied to train ELMs for other structural models.

## Contents

- `src/self_excited_elm/` - the ELM estimator and flat-plate aerodynamic tools.
- `examples/` - scripts that train models and reproduce the benchmark figures.
- `data/random_motion.csv` - portable prescribed-motion input for the
  random-motion example.

## Reproducing the examples

Run the training-only demonstration:

```bash
python examples/train_flat_plate.py
```

Generate the benchmark figures:

```bash
python examples/plot_aerodynamic_derivatives.py
python examples/plot_flat_plate_performance.py
python examples/random_motion.py
```

Each plotting script creates its training data and fits its ELM models in
memory before making predictions. No pre-trained model or binary data archive
is required. The random-motion example reads its prescribed input from the
portable CSV file in `data/`. Figures are written to `output/`.

## Citation

If you use this code in academic work, please cite the associated paper:

> Tondo, G.R., Chawdhury, S., Castro Giraldo, S.A.. and Morgenthal, G. (2026).
> *Modelling non-linear aeroelastic loads in long-span bridges with extreme
> learning machines*. *Bridge Engineering*.
> [https://doi.org/10.1680/jbren.25.00003](https://doi.org/10.1680/jbren.25.00003)

```bibtex
@article{Tondo2026ELM,
  author  = {Tondo, Gledson and Chawdhury, Samir and Castro Giraldo, Sergio Andres and Morgenthal, Guido},
  title   = {Modelling non-linear aeroelastic loads in long-span bridges with extreme learning machines},
  journal = {Bridge Engineering},
  year    = {2026},
  doi     = {10.1680/jbren.25.00003},
  url     = {https://doi.org/10.1680/jbren.25.00003}
}
```
