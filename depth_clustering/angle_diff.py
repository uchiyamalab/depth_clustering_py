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

from math import degrees

import numpy as np
from numba import deferred_type, float32
from numba.experimental import jitclass

from .projections import ProjectionParamsType


@jitclass(
    [
        ("depth_image", float32[:, :]),
        ("params", ProjectionParamsType),
        ("_row_alphas", float32[:]),
        ("_col_alphas", float32[:]),
        ("_beta_rows", float32[:, :]),
        ("_beta_cols", float32[:, :]),
    ]
)
class AngleDiff:
    def __init__(self, depth_image, params):

        self.depth_image = depth_image
        self.params = params

        # --- PreComputeAlphaVecs()
        self._row_alphas = np.empty(params.rows, dtype=float32)
        for r in range(params.rows - 1):
            self._row_alphas[r] = np.fabs(
                params.angle_from_row(r + 1) - params.angle_from_row(r)
            )
        self._row_alphas[-1] = 0.0

        self._col_alphas = np.empty(params.cols, dtype=float32)
        for c in range(params.cols - 1):
            self._col_alphas[c] = np.fabs(
                params.angle_from_col(c + 1) - params.angle_from_col(c)
            )
        last_alpha = np.fabs(
            (params.angle_from_col(0) - params.angle_from_col(params.cols - 1))
        )
        last_alpha -= params.h_span
        self._col_alphas[-1] = last_alpha

        # --- PreComputeBetaAngles()
        _beta_rows = np.zeros((params.rows, params.cols), dtype=np.float32)
        _beta_cols = np.zeros((params.rows, params.cols), dtype=np.float32)

        for r in range(params.rows):
            angle_rows = self._row_alphas[r]
            for c in range(params.cols):
                if depth_image[r][c] < 0.001:
                    continue
                angle_cols = self._col_alphas[c]
                curr = depth_image[r][c]

                next_c = (c + 1) % params.cols
                _beta_cols[r][c] = self.get_beta(
                    angle_cols, curr, depth_image[r][next_c]
                )

                next_r = r + 1
                if next_r >= params.rows:
                    continue
                _beta_rows[r][c] = self.get_beta(
                    angle_rows, curr, depth_image[next_r][c]
                )

        self._beta_rows = _beta_rows
        self._beta_cols = _beta_cols

    @staticmethod
    def get_beta(alpha, current_depth, neighbor_depth):
        d1 = max(current_depth, neighbor_depth)
        d2 = min(current_depth, neighbor_depth)
        beta = np.arctan2(d2 * np.sin(alpha), d1 - d2 * np.cos(alpha))
        return abs(beta)

    def compute_alpha(self, current, neighbor):
        if current.col == 0 and neighbor.col == self.params.cols - 1:
            return self._cols_alphas[-1]
        if neighbor.col == 0 and current.col == self.params.cols - 1:
            return self._cols_alphas[-1]

        if current.row < neighbor.row:
            return self._row_alphas[current.row]
        elif current.row > neighbor.row:
            return self._row_alphas[neighbor.row]
        elif current.col < neighbor.col:
            return self._col_alphas[current.col]
        elif current.col > neighbor.col:
            return self._col_alphas[neighbor.col]
        return 0

    def diff_at(self, fr, to):
        """
        Substituting "from" to "fr" since "from" is a reserved word in Python
        """
        assert fr.row != to.row or fr.col != to.col

        row, col = 0, 0

        last_row = self.params.rows - 1
        row_crosses_border = (fr.row == last_row and to.row == 0) or (
            fr.row == 0 and to.row == last_row
        )

        if row_crosses_border:
            row = last_row
        else:
            row = min(fr.row, to.row)

        last_col = self.params.cols - 1
        col_crosses_border = (fr.col == last_col and to.col == 0) or (
            fr.col == 0 and to.col == last_col
        )

        if col_crosses_border:
            col = last_col
        else:
            col = min(fr.col, to.col)

        if fr.row != to.row:
            return self._beta_rows[row][col]
        if fr.col != to.col:
            return self._beta_cols[row][col]

    def visualize(self):
        mat = np.zeros((self.params.rows, self.params.cols, 3), dtype=np.uint8)

        max_angle_deg = 90.0

        for r in range(self.params.rows):
            for c in range(self.params.cols):
                if self.depth_image[r][c] < 0.001:
                    continue
                row_angle = self._beta_rows[r][c]
                col_angle = self._beta_cols[r][c]

                row_color = int(255 * degrees(row_angle) / max_angle_deg)
                col_color = int(255 * degrees(col_angle) / max_angle_deg)

                mat[r, c, 0] = 255 - row_color
                mat[r, c, 1] = 255 - col_color

        return mat

    @staticmethod
    def satisfies_threshold(angle, _radian_threshold):
        return angle > _radian_threshold


AngleDiffType = deferred_type()
AngleDiffType.define(AngleDiff.class_type.instance_type)
