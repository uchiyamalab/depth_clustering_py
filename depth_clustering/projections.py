"""
Copyright (C) 2023  T. Kamatani
Copyright (C) 2020  I. Bogoslavskyi, C. Stachniss

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from numba import deferred_type, float32, int32
from numba.experimental import jitclass


@jitclass(
    [
        ("start_angle", float32),
        ("end_angle", float32),
        ("num_beams", int32),
        ("step", float32),
        ("span", float32),
    ]
)
class SpanParams:
    def __init__(self, start_angle, end_angle, num_beams):

        self.start_angle = start_angle
        self.end_angle = end_angle
        self.num_beams = num_beams
        self.step = (end_angle - start_angle) / num_beams
        self.span = abs(end_angle - start_angle)


SpanParamsType = deferred_type()
SpanParamsType.define(SpanParams.class_type.instance_type)


@jitclass(
    [
        ("h_span_params", SpanParamsType),
        ("v_span_params", SpanParamsType),
        ("col_angles", float32[:]),
        ("row_angles", float32[:]),
        ("row_angles_sines", float32[:]),
        ("row_angles_cosines", float32[:]),
    ]
)
class ProjectionParams:
    def __init__(self, h_span_params, v_span_params):
        """
        FIXME:

        The original implementation accepts an array of span_param.
        Velodyne Lidars don't scan at equally spaced angles to the
        vertical, so we should fix this code to ensure it does.
        Dolphin Lidars are basically implemented as if it scans at
        equally spaced angles to the vertical.
        """
        self.h_span_params = h_span_params
        self.col_angles = self.fill_vector(h_span_params)

        self.v_span_params = v_span_params
        self.row_angles = self.fill_vector(v_span_params)

        self.row_angles_sines = np.sin(self.row_angles)
        self.row_angles_cosines = np.cos(self.row_angles)

    @property
    def size(self):
        return self.rows * self.cols

    @property
    def cols(self):
        return len(self.col_angles)

    @property
    def rows(self):
        return len(self.row_angles)

    @property
    def h_span(self):
        return self.h_span_params.span

    @property
    def v_span(self):
        return self.v_span_params.span

    def angle_from_row(self, r):
        if r < 0:
            r += len(self.row_angles)
        elif r >= len(self.row_angles):
            r -= len(self.row_angles)
        return self.row_angles[r]

    def angle_from_col(self, c):
        if c < 0:
            c += len(self.col_angles)
        elif c >= len(self.col_angles):
            c -= len(self.col_angles)
        return self.col_angles[c]

    @staticmethod
    def fill_vector(span_params):
        result = np.empty(span_params.num_beams, dtype=np.float32)
        rad = span_params.start_angle
        for i in range(span_params.num_beams):
            result[i] = rad
            rad += span_params.step
        return result


ProjectionParamsType = deferred_type()
ProjectionParamsType.define(ProjectionParams.class_type.instance_type)
