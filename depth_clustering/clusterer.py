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

from collections import defaultdict

import numpy as np
from numba import int64, njit
from numba.typed import dictobject

from .angle_diff import AngleDiff
from .linear_image_labeler import LinearImageLabeler


@njit(fastmath=True)
def compute_labels(input_image, params, angle_threshold):
    angle_diff = AngleDiff(input_image, params)
    labeler = LinearImageLabeler(params.rows, params.cols, angle_threshold, angle_diff)
    l_mat = labeler.compute_labels(input_image)
    return l_mat


@njit(fastmath=True)
def filter_clusters(label_mat, min_cluster_size=10, max_cluster_size=3000):
    result = np.copy(label_mat)

    labels_counter = dictobject.new_dict(int64, int64)
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            label = result[r][c]
            v = labels_counter.get(label, 0)
            labels_counter[label] = v + 1

    labels_to_erase = set()
    for label, count in labels_counter.items():
        if count < min_cluster_size or count > max_cluster_size:
            labels_to_erase.add(label)

    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            if result[r][c] in labels_to_erase:
                result[r][c] = 0
    return result


def compute_labels_with_filtering(
    input_image, params, min_cluster_size=10, max_cluster_size=3000
):
    label_mat = compute_labels(input_image, params)
    result = filter_clusters(label_mat, min_cluster_size, max_cluster_size)
    return result


def calculate_segmented_point_clouds(label_mat, pc_image):
    result = defaultdict(list)

    rows, cols = label_mat.shape
    for r in range(rows):
        for c in range(cols):
            label = label_mat[r][c]
            if label == 0:
                continue
            p = pc_image[r][c]
            result[label].append(p)

    for k, l in result.items():
        result[k] = np.array(l)

    return result
