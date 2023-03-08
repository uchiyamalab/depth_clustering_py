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

# flake8: noqa F841,E501

import unittest
from math import radians

import numpy as np

from depth_clustering import (
    AngleDiff,
    LinearImageLabeler,
    PixelCoord,
    ProjectionParams,
    SpanParams,
    calculate_segmented_point_clouds,
    compute_labels,
    convert_spherical_to_cartesian,
    filter_clusters,
)


class TestPixelCoord(unittest.TestCase):
    def test_addition(self):
        x = PixelCoord(1, 2)
        y = PixelCoord(-1, 3)

        self.assertEqual(PixelCoord(0, 5), x + y)


class TestProjecctionParams(unittest.TestCase):
    def test_properties(self):
        h_span_params = SpanParams(radians(-45), radians(45), num_beams=328)
        v_span_params = SpanParams(radians(-30), radians(30), num_beams=64)
        params = ProjectionParams(h_span_params, v_span_params)

        self.assertEqual(params.rows, 64)
        self.assertEqual(params.cols, 328)
        self.assertEqual(params.size, 64 * 328)

        targets = [
            (params.angle_from_col, 0, -0.785398),
            (params.angle_from_col, 164, 0),
            (params.angle_from_col, 327, 0.780607),
            (params.angle_from_row, 0, -0.523597),
            (params.angle_from_row, 63, 0.5072361),
        ]
        for func, index, output in targets:
            self.assertAlmostEqual(func(index), output, places=4)


class TestAngleDiff(unittest.TestCase):
    def test_angle_diff(self):
        h_span_params = SpanParams(radians(-45), radians(45), num_beams=328)
        v_span_params = SpanParams(radians(-30), radians(30), num_beams=64)
        params = ProjectionParams(h_span_params, v_span_params)

        input_image = np.random.rand(64, 328).astype("float32")
        angle_diff = AngleDiff(input_image, params)
        mat = angle_diff.visualize()


class TestLinearImageLabeler(unittest.TestCase):
    def test_labeler(self):
        h_span_params = SpanParams(radians(-45), radians(45), num_beams=328)
        v_span_params = SpanParams(radians(-30), radians(30), num_beams=64)
        params = ProjectionParams(h_span_params, v_span_params)

        input_image = np.random.rand(64, 328).astype("float32")
        angle_diff = AngleDiff(input_image, params)

        labeler = LinearImageLabeler(
            rows=64, cols=328, angle_threshold=radians(10.0), diff_helper=angle_diff
        )
        l_mat = labeler.compute_labels(input_image)
        # print(Counter(l_mat.flatten()).most_common(20))

        l_mat = compute_labels(input_image, params, angle_threshold=radians(10.0))
        l_filtered_mat = filter_clusters(l_mat, 5, 3000)
        # print(Counter(l_filtered_mat.flatten()).most_common(20))

        pc_image = convert_spherical_to_cartesian(input_image, params)
        segmented = calculate_segmented_point_clouds(l_filtered_mat, pc_image)
        # print(list(segmented.keys()))


if __name__ == "__main__":
    unittest.main()
