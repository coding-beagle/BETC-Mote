import pytest
import numpy as np
from gdata.utils import (
    vector_between_two_points,
    angle_between_three_points,
    normal_vector_of_plane_on_three_points,
)


def compare_vecs(a, b) -> bool:
    if len(a) != len(b):
        print(f"{len(a)} != {len(b)}")
        return False

    for index, i in enumerate(a):
        if i != b[index]:
            print(f"index {index}: {i} != {b[index]}")
            return False

    return True


vector_point_test_data = [
    [[0, 1, 0], [0, 1, 1], [0, 0, 1], False],
    [[1, 0, 0], [1, 1, 0], [0, 1, 0], False],
]


@pytest.mark.parametrize("vec_a, vec_b, expected, should_fail", vector_point_test_data)
def test_vector_between_points(vec_a, vec_b, expected, should_fail):
    vec_c = vector_between_two_points(vec_a, vec_b)
    assert compare_vecs(vec_c, expected) and not (should_fail)


three_point_test_data = [
    [[0, 1, 0], [0, 0, 0], [1, 0, 0], 90.0, False],
]


@pytest.mark.parametrize("a,b,c,expected,should_fail", three_point_test_data)
def test_get_angle_between_three_points(a, b, c, expected, should_fail):
    angle = angle_between_three_points(a, b, c, True)

    assert not (should_fail) and angle == expected


plane_normal_test_data = [
    [[0, 1, 0], [0, 0, 0], [1, 0, 0], False, [0, 0, 1], False],
    [[0, 10, 0], [0, 0, 0], [10, 0, 0], True, [0, 0, 1], False],
]


@pytest.mark.parametrize("a,b,c,unit,expected,should_fail", plane_normal_test_data)
def test_get_normal_between_two_points(a, b, c, unit, expected, should_fail):
    normal = normal_vector_of_plane_on_three_points(a, b, c)
    assert not (should_fail) and compare_vecs(normal, np.array(expected))
