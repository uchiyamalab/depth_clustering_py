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
from numba import float32, int32, uint16
from numba.experimental import jitclass

from .angle_diff import AngleDiffType


@jitclass(
    [
        ("row", int32),
        ("col", int32),
    ]
)
class PixelCoord:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __add__(self, other):
        return PixelCoord(self.row + other.row, self.col + other.col)

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


@jitclass(
    [
        ("rows", uint16),
        ("cols", uint16),
        ("angle_threshold", float32),
        ("diff_helper", AngleDiffType),
    ]
)
class LinearImageLabeler:
    def __init__(self, rows, cols, angle_threshold, diff_helper):

        self.rows = rows
        self.cols = cols
        self.angle_threshold = angle_threshold
        self.diff_helper = diff_helper

    def compute_labels(self, depth_image):

        label_image = np.zeros((self.rows, self.cols), dtype=np.uint16)

        label = 1
        for row in range(self.rows):
            for col in range(self.cols):
                if label_image[row][col] > 0:
                    continue
                if depth_image[row][col] < 0.005:
                    continue
                self.label_one_component(
                    label_image, depth_image, label, PixelCoord(row, col)
                )
                label += 1

        return label_image

    @staticmethod
    def satisfies_threshold(angle, _radian_threshold):
        return angle > _radian_threshold

    def label_one_component(self, label_image, depth_image, label, start):

        labeling_queue = []
        labeling_queue.append(start)

        while len(labeling_queue) > 0:
            current = labeling_queue.pop()
            current_label = label_image[current.row][current.col]

            if current_label > 0:
                continue

            label_image[current.row][current.col] = label
            current_depth = depth_image[current.row][current.col]
            if current_depth < 0.001:
                continue

            for step in (
                PixelCoord(-1, 0),
                PixelCoord(1, 0),
                PixelCoord(0, -1),
                PixelCoord(0, 1),
            ):

                neighbor = current + step
                if neighbor.row < 0 or neighbor.row >= self.rows:
                    continue

                # WrapCols
                if neighbor.col < 0:
                    # neighbor.col = neighbor.col + neighbor.col
                    neighbor.col = (neighbor.col + self.cols * 10) % self.cols
                if neighbor.col >= self.cols:
                    neighbor.col = neighbor.col % self.cols

                neigh_label = label_image[neighbor.row][neighbor.col]
                if neigh_label > 0:
                    continue

                diff = self.diff_helper.diff_at(current, neighbor)
                if self.satisfies_threshold(diff, self.angle_threshold):
                    labeling_queue = [neighbor] + labeling_queue
