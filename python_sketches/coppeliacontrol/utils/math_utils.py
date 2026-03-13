"""
math_utils.py
=============
Lightweight 3-D vector helpers (plain Python lists, no numpy dependency).
"""


def vec3(lm):
    return [lm.x, lm.y, lm.z]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(3)]


def vec_scale(v, s):
    return [v[i] * s for i in range(3)]


def vec_length(v):
    return sum(x**2 for x in v) ** 0.5


def vec_normalize(v):
    l = vec_length(v)
    return vec_scale(v, 1.0 / l) if l > 1e-6 else None


def remap_axes(v):
    """Convert MediaPipe camera-space axes to CoppeliaSim world axes."""
    return [-v[2], v[0], -v[1]]
