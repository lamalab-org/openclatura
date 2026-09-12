"""Ring orientation must not depend on the drawn polygon's radius."""

from openclatura.fusion.layout import _center_row_orientation_score, _direction_grid_centers


def test_adjacent_unequal_polygons_have_equal_direction_steps():
    adjacency = frozenset((frozenset((0, 2)), frozenset((1, 2))))
    centers = {0: (7, 4), 1: (3, 2), 2: (9, 2)}
    grid = _direction_grid_centers(centers, adjacency)
    assert grid is not None
    assert 2 * grid[0][0] == grid[1][0] + grid[2][0]
    mirrored = _direction_grid_centers({face: (-x, y) for face, (x, y) in centers.items()}, adjacency)
    assert mirrored is not None
    assert _center_row_orientation_score(grid, [1, 2]) == _center_row_orientation_score(mirrored, [2, 1])


def test_direction_map_requires_consistent_closed_walks():
    adjacency = frozenset(frozenset(edge) for edge in ((0, 1), (1, 2), (2, 0)))
    assert _direction_grid_centers({0: (0, 0), 1: (1, 0), 2: (2, 0)}, adjacency) is None


def test_point_on_both_axes_contributes_one_quarter_to_each_quadrant():
    assert _center_row_orientation_score({0: (0, 0)}, [0]) == (-1, 1, -2)
