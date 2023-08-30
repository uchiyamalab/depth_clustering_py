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

from numba import deferred_type, float32
from numba.experimental import jitclass


@jitclass(
    [
        ("source_image", float32[:, :]),
    ]
)
class SimpleDiff:
    def __init__(self, source_image):

        self.source_image = source_image

    def diff_at(self, fr, to):
        """
        Substituting "from" to "fr" since "from" is a reserved word in Python
        """
        assert fr.row != to.row or fr.col != to.col

        return abs(
            self.source_image[fr.row][fr.col] -
            self.source_image[to.row][to.col]
        )

    @staticmethod
    def satisfies_threshold(value, threshold):
        return value < threshold


SimpleDiffType = deferred_type()
SimpleDiffType.define(SimpleDiff.class_type.instance_type)
