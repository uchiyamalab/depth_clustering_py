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
from math import radians

import cv2
import numpy as np
from numba import njit, float32, uint8

from .simple_diff import SimpleDiff
from .linear_image_labeler import SimpleDiffLinearImageLabeler, PixelCoord


@njit(uint8[:, :](uint8))
def get_uniform_kernel(window_size):
    if window_size % 2 == 0:
        raise ValueError("only odd window size allowed")

    kernel = np.zeros((window_size, 1), dtype=np.uint8)
    kernel[0, 0] = 1
    kernel[-1, 0] = 1
    return kernel


def dilate_image(label_image, window_size):
    kernel = get_uniform_kernel(window_size)
    dilated = cv2.dilate(label_image, kernel)
    return dilated


@njit
def dilate_custom(image, window_size):
    h, w = image.shape
    half_w = window_size // 2
    dilated_image = np.copy(image)

    for y in range(half_w, h - half_w):
        for x in range(half_w, w - half_w):
            max_val = np.max(image[y-half_w:y+half_w+1, x-half_w:x+half_w+1])
            dilated_image[y, x] = max_val

    return dilated_image


@njit(float32[:, :](float32[:, :], uint8, float32))
def repair_depth(no_ground_image, step, depth_threshold):
    inpainted_depth = np.copy(no_ground_image)
    rows, cols = inpainted_depth.shape

    for c in range(cols):
        for r in range(rows):
            curr_depth = inpainted_depth[r, c]
            if curr_depth < 0.001:
                counter = 0
                sum_depths = 0.0
                for i in range(1, step):
                    if r - i < 0:
                        continue
                    for j in range(1, step):
                        if r + j > rows - 1:
                            continue
                        prev = inpainted_depth[r - i, c]
                        next_depth = inpainted_depth[r + j, c]
                        if prev > 0.001 and next_depth > 0.001 and \
                           abs(prev - next_depth) < depth_threshold:
                            sum_depths += prev + next_depth
                            counter += 2
                if counter > 0:
                    inpainted_depth[r, c] = sum_depths / counter

    return inpainted_depth


@njit
def zero_out_ground_bfs_jit(
    image, angle_image, angle_threshold, kernel_size, params
):

    start_thresh = radians(30)

    rows = params.rows
    cols = params.cols

    image_labeler = SimpleDiffLinearImageLabeler(
        rows, cols, angle_threshold, SimpleDiff(angle_image),
    )
    label_image = np.zeros((rows, cols), dtype=np.uint16)

    for c in range(cols):
        r = rows - 1
        while (r > 0 and image[r][c] < 0.001):
            r -= 1
        current_label = label_image[r][c]
        if current_label > 0:
            # this coord was already labeled, skip
            continue
        if angle_image[r][c] > start_thresh:
            continue

        current_coord = PixelCoord(r, c)
        image_labeler.label_one_component(label_image, image, 1, current_coord)

    kernel_size = max(kernel_size - 2, 3)
    dilated = dilate_custom(label_image, window_size=5)
    res = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            if dilated[r][c] == 0:
                res[r][c] = image[r][c]
    return res


@njit
def create_angle_image_jit(depth_image, params):
    rows = params.rows
    cols = params.cols
    angle_image = np.zeros((rows, cols), dtype=np.float32)
    x_mat = np.zeros((rows, cols), dtype=np.float32)
    y_mat = np.zeros((rows, cols), dtype=np.float32)

    sines_vec = params.row_angles_sines
    cosines_vec = params.row_angles_cosines

    x_mat[0, :] = depth_image[0, :] * cosines_vec[0]
    y_mat[0, :] = depth_image[0, :] * sines_vec[0]

    for r in range(1, rows):
        x_mat[r, :] = depth_image[r, :] * cosines_vec[r]
        y_mat[r, :] = depth_image[r, :] * sines_vec[r]
        dx = np.abs(x_mat[r, :] - x_mat[r - 1, :])
        dy = np.abs(y_mat[r, :] - y_mat[r - 1, :])
        angle_image[r, :] = np.arctan2(dy, dx)

    return angle_image


class DepthGroundRemover:

    def __init__(self, params, window_size, ground_remove_angle):
        self.params = params
        self.window_size = window_size
        self.ground_remove_angle = ground_remove_angle

    def on_new_object_received(self, raw_depth_image):
        depth_image = repair_depth(raw_depth_image, 5, 1.0)
        angle_image = self.create_angle_image(depth_image)
        smoothed_image = self.apply_savitsky_golay_smoothing(
            angle_image,
            self.window_size,
        )
        no_ground_image = self.zero_out_ground_bfs(
            depth_image,
            smoothed_image,
            self.ground_remove_angle,
            self.window_size
        )
        return no_ground_image

    def zero_out_ground_bfs(
        self, image, angle_image, angle_threshold, kernel_size
    ):
        return zero_out_ground_bfs_jit(
            image, angle_image, angle_threshold, kernel_size, self.params,
        )

    def create_angle_image(self, depth_image):
        return create_angle_image_jit(depth_image, self.params)

    def apply_savitsky_golay_smoothing(self, image, window_size):
        kernel = self.get_savitsky_golay_kernel(window_size)
        smoothed_image = cv2.filter2D(
            image, -1, kernel, borderType=cv2.BORDER_REFLECT101
        )
        return smoothed_image

    def get_savitsky_golay_kernel(self, window_size):
        if window_size % 2 == 0:
            raise ValueError("only odd window size allowed")

        if window_size not in [5, 7, 9, 11]:
            raise ValueError("bad window size")

        kernel = np.zeros((window_size, 1), dtype=np.float32)
        if window_size == 5:
            kernel[:, 0] = [-3.0, 12.0, 17.0, 12.0, -3.0]
            kernel /= 35.0

        elif window_size == 7:
            kernel[:, 0] = [-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]
            kernel /= 21.0

        elif window_size == 9:
            kernel[:, 0] = [
                -21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0
            ]
            kernel /= 231.0

        elif window_size == 11:
            kernel[:, 0] = [
                -36.0, 9.0, 44.0, 69.0, 84.0, 89.0, 84.0,
                69.0, 44.0, 9.0, -36.0
            ]
            kernel /= 429.0

        return kernel
