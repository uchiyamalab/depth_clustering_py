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
    ProjectionParams,
    SpanParams,
    DepthGroundRemover,
)


class TestGroundRemover(unittest.TestCase):
    def test_ground_remover(self):
        h_span_params = SpanParams(radians(-180), radians(180), num_beams=870)
        v_span_params = SpanParams(radians(-24), radians(2), num_beams=64)
        params = ProjectionParams(h_span_params, v_span_params)

        remover = DepthGroundRemover(params, window_size=5, ground_remove_angle=radians(5))

        depth_image = np.random.rand(64, 870).astype("float32")
        removed = remover.on_new_object_received(depth_image)

        assert removed.shape == (64, 870)


if __name__ == "__main__":
    unittest.main()
