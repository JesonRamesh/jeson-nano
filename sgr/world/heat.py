"""Heat model (spec §7.4): T(x) = ambient + sum over fires of 380 * exp(-d^2 / (2 * 1.5^2)). No occlusion."""

from __future__ import annotations

import numpy as np

PEAK_C = 380.0
SIGMA_M = 1.5


def temperature_at(x: float, y: float, fires_xy, ambient_c: float = 20.0) -> float:
    if len(fires_xy) == 0:
        return float(ambient_c)
    f = np.asarray(fires_xy, dtype=float).reshape(-1, 2)
    d2 = (f[:, 0] - x) ** 2 + (f[:, 1] - y) ** 2
    return float(ambient_c + PEAK_C * np.exp(-d2 / (2 * SIGMA_M**2)).sum())
