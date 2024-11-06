#
# Copyright (C) 2018-2020 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Integrity tests for the low-level module.
"""
import sys

import pytest

import _tsinfer


class TestOutOfMemory:
    """
    Make sure we raise the correct error when out of memory occurs in
    the library code.
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="windows seems to allow initializing with insane # of nodes"
        " (perhaps memory allocation is optimised out at this stage?)",
    )
    def test_tree_sequence_builder_too_many_nodes(self):
        big = 2**62
        with pytest.raises(MemoryError):
            _tsinfer.TreeSequenceBuilder([2], max_nodes=big)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="windows raises an assert error not a memory error with 2**62 edges"
        " (line 149 of object_heap.c)",
    )
    def test_tree_sequence_builder_too_many_edges(self):
        big = 2**62
        with pytest.raises(MemoryError):
            _tsinfer.TreeSequenceBuilder([2], max_edges=big)


class TestAncestorMatcher:
    """
    Tests for the AncestorMatcher C Python interface.
    """

    def test_init(self):
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher()
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher(None)
        tsb = _tsinfer.TreeSequenceBuilder([2])
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher(tsb)
        with pytest.raises(TypeError):
            _tsinfer.AncestorMatcher(tsb, [1])
        for bad_type in [None, {}]:
            with pytest.raises(TypeError):
                _tsinfer.AncestorMatcher(tsb, [1], [1], extended_checks=bad_type)
            with pytest.raises(TypeError):
                _tsinfer.AncestorMatcher(tsb, [1], [1], precision=bad_type)
        for bad_array in [[], [[], []], None, "sdf", [1, 2, 3]]:
            with pytest.raises(ValueError):
                _tsinfer.AncestorMatcher(tsb, bad_array, [1])
            with pytest.raises(ValueError):
                _tsinfer.AncestorMatcher(tsb, [1], bad_array)


class TestTreeSequenceBuilder:
    """
    Tests for the AncestorMatcher C Python interface.
    """

    def test_init(self):
        with pytest.raises(TypeError):
            _tsinfer.TreeSequenceBuilder()

        for bad_type in [None, "sdf", {}]:
            with pytest.raises(TypeError):
                _tsinfer.TreeSequenceBuilder([2], max_nodes=bad_type)
            with pytest.raises(TypeError):
                _tsinfer.TreeSequenceBuilder([2], max_edges=bad_type)

    def test_bad_num_alleles(self):
        for bad_array in [None, "serf", [[], []], ["asdf"], {}]:
            with pytest.raises(ValueError):
                _tsinfer.TreeSequenceBuilder(bad_array)
        with pytest.raises(_tsinfer.LibraryError, match="number of alleles"):
            _tsinfer.TreeSequenceBuilder([1000])

    def test_bad_ancestral_state(self):
        for bad_array in [None, "serf", [[], []], ["asdf"], {}]:
            with pytest.raises(ValueError):
                _tsinfer.TreeSequenceBuilder([2], ancestral_state=bad_array)
        for bad_ancestral_state in [-1, -2]:
            with pytest.raises(_tsinfer.LibraryError, match="Bad ancestral state"):
                _tsinfer.TreeSequenceBuilder([2], ancestral_state=[bad_ancestral_state])

        with pytest.raises(ValueError, match="ancestral state array wrong size"):
            _tsinfer.TreeSequenceBuilder([2, 2, 2], ancestral_state=[])


class TestAncestorBuilder:
    """
    Tests for the AncestorBuilder C Python interface.
    """

    def test_init(self):
        with pytest.raises(TypeError):
            _tsinfer.AncestorBuilder()
        for bad_value in [None, "serf", [[], []], ["asdf"], {}]:
            with pytest.raises(TypeError):
                _tsinfer.AncestorBuilder(num_samples=2, max_sites=bad_value)
                _tsinfer.AncestorBuilder(num_samples=bad_value, max_sites=2)

        for bad_num_samples in [0, 1]:
            with pytest.raises(_tsinfer.LibraryError):
                _tsinfer.AncestorBuilder(num_samples=bad_num_samples, max_sites=0)

    def test_add_site(self):
        ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=10)
        for bad_type in ["sdf", {}, None]:
            with pytest.raises(TypeError):
                ab.add_site(time=bad_type, genotypes=[0, 0])
        for bad_genotypes in ["asdf", [[], []], [0, 1, 2]]:
            with pytest.raises(ValueError):
                ab.add_site(time=0, genotypes=bad_genotypes)

    def test_add_too_many_sites(self):
        for max_sites in range(10):
            ab = _tsinfer.AncestorBuilder(num_samples=2, max_sites=max_sites)
            for _ in range(max_sites):
                ab.add_site(time=1, genotypes=[0, 1])
            for _ in range(2 * max_sites):
                with pytest.raises(_tsinfer.LibraryError) as record:
                    ab.add_site(time=1, genotypes=[0, 1])
                msg = "Cannot add more sites than the specified maximum."
                assert str(record.value) == msg

    # TODO need tester methods for the remaining methonds in the class.
