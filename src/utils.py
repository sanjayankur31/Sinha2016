#!/usr/bin/env python3
"""
Enter one line description here.

File:

Copyright 2017 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import random


def create_sparse_list(length, static_w, sparsity):
    """Create one list to use with SetStatus."""
    weights = []
    valid_values = int(length * sparsity)
    weights = (
        ([float(static_w), ] * valid_values) +
        ([0., ] * int(length - valid_values)))

    random.shuffle(weights)
    return weights


def fill_matrix(weightlist, static_w, sparsity):
    """Create a weight matrix to use in syn dict."""
    weights = []
    for row in weightlist:
        if isinstance(row, (list, tuple)):
            rowlen = len(row)
            valid_values = int(rowlen * sparsity)
            arow = (
                ([float(static_w), ] * valid_values) +
                ([0., ] * int(rowlen - valid_values))
            )
            random.shuffle(arow)
            weights.append(arow)
    return weights


def setup_matrix(pre_dim, post_dim, static_w, sparsity):
    """Create a weight matrix to use in syn dict."""
    weights = []
    valid_values = int(post_dim * sparsity)
    for i in range(0, pre_dim):
        arow = (
            ([float(static_w), ] * valid_values) +
            ([0., ] * int(post_dim - valid_values))
        )
        random.shuffle(arow)
        weights.append(arow)
    return weights
