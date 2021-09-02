"""
author: Diego Di Benedetto
"""

import numpy as np


def normalize(a):
    # vector normalization
    a = np.array(a)
    return a / np.linalg.norm(a)


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # calculates the point [x, y] where two lines intersect
    v1 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    v2 = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if u == 0:
        u = 1e-7

    m = v1 / u
    n = -v2 / u

    if not (0.0 <= m <= 1.0):
        return None
    if not (0.0 <= n <= 1.0):
        return None

    x = x1 + m * (x2 - x1)
    y = y1 + m * (y2 - y1)

    return [x, y]


def get_point(a, b, c):
    # given a segment by two points a and b, calculates the perpendicular point to c that belongs to the segment
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    x3 = c[0]
    y3 = c[1]
    px = x2 - x1
    py = y2 - y1
    dAB = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
    x = x1 + u * px
    y = y1 + u * py
    return x, y  # perpendicular point to c that lays in ab


def vector_between_points(point0, point1):
    # calculates the vector that goes from point0 to point1
    return [point1[0] - point0[0], point1[1] - point0[1]]


def shift_point_with_vector(point, vector):
    # shifts point from the origin of the vector to its end
    return [point[0] + vector[0], point[1] + vector[1]]