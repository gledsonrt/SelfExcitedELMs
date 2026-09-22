"""A deterministic single-hidden-layer extreme learning machine."""

from __future__ import annotations

import numpy as np


class ExtremeLearningMachine:
    """ELM with random, fixed sigmoid hidden neurons and least-squares output.

    Inputs and targets are standardised during fitting.  A fixed seed makes
    model selection and the examples reproducible.
    """

    def __init__(self, n_inputs: int, n_hidden: int = 50, n_outputs: int = 1,
                 regularizer: float = 0.0, random_state: int = 0,
                 activation: str = "sigmoid"):
        self.n_inputs, self.n_hidden, self.n_outputs = n_inputs, n_hidden, n_outputs
        self.regularizer, self.random_state = regularizer, random_state
        if activation not in {"sigmoid", "linear"}:
            raise ValueError("activation must be 'sigmoid' or 'linear'.")
        self.activation = activation
        self.reset_neurons(n_hidden)

    def reset_neurons(self, n_hidden: int) -> None:
        self.n_hidden = int(n_hidden)
        self.weights = np.random.default_rng(self.random_state).standard_normal(
            (self.n_inputs, self.n_hidden)
        )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def _activate(self, x: np.ndarray) -> np.ndarray:
        return x if self.activation == "linear" else self._sigmoid(x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ExtremeLearningMachine":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, self.n_outputs)
        self.x_mean, self.x_scale = x.mean(0), x.std(0)
        self.y_mean, self.y_scale = y.mean(0), y.std(0)
        self.x_scale[self.x_scale == 0] = 1.0
        self.y_scale[self.y_scale == 0] = 1.0
        h = self._activate(((x - self.x_mean) / self.x_scale) @ self.weights)
        if self.regularizer:
            self.beta = np.linalg.solve(h.T @ h + self.regularizer**2 * np.eye(self.n_hidden), h.T @ ((y - self.y_mean) / self.y_scale))
        else:
            self.beta = np.linalg.lstsq(h, (y - self.y_mean) / self.y_scale, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = self._activate(((x - self.x_mean) / self.x_scale) @ self.weights) @ self.beta
        return (y * self.y_scale + self.y_mean).squeeze()
