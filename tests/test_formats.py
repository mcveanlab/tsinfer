#
# Copyright (C) 2018-2022 University of Oxford
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
Tests for the data files.
"""
import datetime
import itertools
import json
import os.path
import sys
import tempfile
import warnings

import lmdb
import msprime
import numcodecs
import numcodecs.blosc as blosc
import numpy as np
import pytest
import tskit
import tsutil
import zarr

import tsinfer
import tsinfer.exceptions as exceptions
import tsinfer.formats as formats

IS_WINDOWS = sys.platform == "win32"


class ConvenienceFunctions:
    """
    Tests for a couple of convenience functions at the top level of formats.py
    """

    def test_np_obj_equal_bad(self):
        obj_array = np.array([None])
        with pytest.raises(AttributeError):
            formats.np_obj_equal(obj_array, None)

    def test_np_obj_equal(self):
        obj_array = np.array([{1, 3}, None])
        assert obj_array.dtype == np.dtype("object")
        assert formats.np_obj_equal(obj_array, np.array([{1, 3}, None]))
        for not_equal in [
            np.array([{1, 4}, None]),
            np.array([{1, 3}, {1, 3}]),
            np.array([None, None]),
            np.array([{}, {}]),
            # Different shapes
            np.array([[{1, 3}, None]]),
            np.array([{1, 3}, None, None]),
            np.array([{1, 3}]),
        ]:
            assert not formats.np_obj_equal(obj_array, not_equal)


class DataContainerMixin:
    """
    Common tests for the the data container classes."
    """

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            self.class_to_test.load("/file/does/not/exist")
        if sys.platform != "win32":
            with pytest.raises(IsADirectoryError):
                self.class_to_test.load("/")
            bad_format_files = ["LICENSE", "/dev/urandom"]
        else:
            # Windows raises a PermissionError not IsADirectoryError when opening a dir
            with pytest.raises(PermissionError):
                self.class_to_test.load("/")
            # No /dev/urandom on Windows
            bad_format_files = ["LICENSE"]
        for bad_format_file in bad_format_files:
            assert os.path.exists(bad_format_file)
            with pytest.raises(exceptions.FileFormatError):
                self.class_to_test.load(bad_format_file)


class TestSampleData(DataContainerMixin):
    """
    Test cases for the sample data file format.
    """

    class_to_test = formats.SampleData

    def verify_data_round_trip(self, ts, input_file):
        def encode_metadata(metadata, schema):
            if schema is None:
                if len(metadata) > 0:
                    metadata = json.loads(metadata)
                else:
                    metadata = None
            return metadata

        schema = ts.metadata_schema.schema
        input_file.metadata_schema = schema
        input_file.metadata = encode_metadata(ts.metadata, schema)

        assert ts.num_sites > 1
        schema = ts.tables.populations.metadata_schema.schema
        input_file.populations_metadata_schema = schema
        for pop in ts.populations():
            input_file.add_population(metadata=encode_metadata(pop.metadata, schema))

        # For testing, depend on the sample nodes being sorted by individual
        schema = ts.tables.individuals.metadata_schema.schema
        input_file.individuals_metadata_schema = schema
        for i, group in itertools.groupby(
            ts.samples(), lambda n: ts.node(n).individual
        ):
            nodes = [ts.node(nd) for nd in group]
            if i == tskit.NULL:
                for node in nodes:
                    input_file.add_individual(
                        ploidy=1,
                        population=node.population,
                        time=node.time,
                    )
            else:
                input_file.add_individual(
                    ploidy=len(nodes),
                    population=nodes[0].population,
                    metadata=encode_metadata(ts.individual(i).metadata, schema),
                    location=ts.individual(i).location,
                    flags=ts.individual(i).flags,
                    time=nodes[0].time,
                )

        schema = ts.tables.sites.metadata_schema.schema
        input_file.sites_metadata_schema = schema
        for v in ts.variants():
            t = np.nan  # default is that a site has no meaningful time
            if len(v.site.mutations) == 1:
                t = ts.node(v.site.mutations[0].node).time
            input_file.add_site(
                v.site.position,
                v.genotypes,
                v.alleles,
                metadata=encode_metadata(v.site.metadata, schema),
                time=t,
            )
        input_file.record_provenance("verify_data_round_trip")
        input_file.finalise()
        assert input_file.format_version == formats.SampleData.FORMAT_VERSION
        assert input_file.format_name == formats.SampleData.FORMAT_NAME
        assert input_file.num_samples == ts.num_samples
        assert input_file.sequence_length == ts.sequence_length
        assert input_file.metadata_schema == ts.metadata_schema.schema
        assert input_file.metadata == ts.metadata
        assert input_file.num_sites == ts.num_sites
        assert input_file.sites_genotypes.dtype == np.int8
        assert input_file.sites_position.dtype == np.float64
        # Take copies to avoid decompressing the data repeatedly.
        pop_metadata = input_file.populations_metadata[:]
        genotypes = input_file.sites_genotypes[:]
        position = input_file.sites_position[:]
        alleles = input_file.sites_alleles[:]
        site_times = input_file.sites_time[:]
        site_metadata = input_file.sites_metadata[:]
        location = input_file.individuals_location[:]
        individual_metadata = input_file.individuals_metadata[:]
        sample_time = input_file.individuals_time[:]
        for j, variant in enumerate(ts.variants()):
            assert variant.site.position == position[j]
            assert np.all(variant.genotypes == genotypes[j])
            assert alleles[j] == list(variant.alleles)
            if len(variant.site.mutations) == 1:
                the_time = ts.node(variant.site.mutations[0].node).time
                assert the_time == site_times[j]
            if variant.site.metadata:
                assert site_metadata[j] == variant.site.metadata
        assert input_file.num_populations == ts.num_populations
        for pop in ts.populations():
            if pop.metadata:
                assert pop_metadata[pop.id] == pop.metadata
        if ts.num_individuals == 0:
            assert input_file.num_individuals == ts.num_samples
        else:
            for individual in ts.individuals():
                assert np.all(individual.location == location[individual.id])
                if individual.metadata:
                    assert individual_metadata[individual.id] == individual.metadata
                for n in individual.nodes:
                    assert ts.node(n).time == sample_time[individual.id]

    @pytest.mark.skipif(
        IS_WINDOWS, reason="windows simultaneous file permissions issue"
    )
    def test_defaults_with_path(self):
        ts = tsutil.get_example_ts(10)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            input_file = formats.SampleData(
                path=filename, sequence_length=ts.sequence_length
            )
            self.verify_data_round_trip(ts, input_file)
            compressor = formats.DEFAULT_COMPRESSOR
            for _, array in input_file.arrays():
                assert array.compressor == compressor
            with tsinfer.load(filename) as other:
                assert other == input_file

    def test_bad_max_file_size(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for bad_size in ["a", "", -1]:
                with pytest.raises(ValueError):
                    formats.SampleData(path=filename, max_file_size=bad_size)
            for bad_size in [[1, 3], np.array([1, 2])]:
                with pytest.raises(TypeError):
                    formats.SampleData(path=filename, max_file_size=bad_size)

    def test_too_small_max_file_size_init(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            # Fail immediately if the max_size is so small we can't even create a file
            filename = os.path.join(tempdir, "samples.tmp")
            with pytest.raises(lmdb.MapFullError):
                formats.SampleData(path=filename, sequence_length=1, max_file_size=1)

    def test_too_small_max_file_size_add(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            base_size = 2**16  # Big enough to allow the initial file to be created
            # Fail during adding a large amount of data
            with pytest.raises(lmdb.MapFullError):
                filename = os.path.join(tempdir, "samples.tmp")
                with formats.SampleData(
                    path=filename, sequence_length=1, max_file_size=base_size
                ) as small_sample_file:
                    small_sample_file.add_site(
                        0,
                        alleles=["0", "1"],
                        genotypes=np.zeros(base_size, dtype=np.int8),
                    )
            # Work around https://github.com/tskit-dev/tsinfer/issues/201
            small_sample_file.data.store.close()

    def test_acceptable_max_file_size(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            # set a reasonably large number of sites and samples, and check we
            # don't bomb out
            n_samples = 2**10
            n_sites = 2**12
            np.random.seed(123)
            filename = os.path.join(tempdir, "samples.tmp")
            with formats.SampleData(
                path=filename,
                sequence_length=n_sites,
                compressor=False,
                max_file_size=None,
            ) as samples:
                for pos in range(n_sites):
                    samples.add_site(
                        pos,
                        alleles=["0", "1"],
                        genotypes=np.random.randint(2, size=n_samples, dtype=np.int8),
                    )
            assert samples.num_sites == n_sites
            assert samples.num_samples == n_samples
            assert samples.file_size > n_samples * n_sites
            samples.close()

    def test_inference_not_supported(self):
        sample_data = formats.SampleData(sequence_length=1)
        with pytest.raises(ValueError):
            sample_data.add_site(0.1, [1, 1], inference=False)

    def test_defaults_no_path(self):
        ts = tsutil.get_example_ts(10)
        with formats.SampleData(sequence_length=ts.sequence_length) as sample_data:
            self.verify_data_round_trip(ts, sample_data)
            for _, array in sample_data.arrays():
                assert array.compressor == formats.DEFAULT_COMPRESSOR

    def test_with_metadata_and_individuals(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(5, ploidy=2)
        with formats.SampleData(sequence_length=ts.sequence_length) as sample_data:
            self.verify_data_round_trip(ts, sample_data)

    def test_access_individuals(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(5, ploidy=2)
        sd = tsinfer.SampleData.from_tree_sequence(ts)
        assert sd.num_individuals > 0
        has_some_metadata = False
        for i, individual in enumerate(sd.individuals()):
            if individual.metadata is not None:
                has_some_metadata = True  # Check that we do compare something sometimes
            assert i == individual.id
            other_ind = sd.individual(i)
            assert other_ind == individual
            other_ind.samples = []
            assert other_ind != individual
        assert has_some_metadata
        assert i == sd.num_individuals - 1

    def test_access_populations(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(5, ploidy=2)
        sd = tsinfer.SampleData.from_tree_sequence(ts)
        assert sd.num_individuals > 0
        has_some_metadata = False
        for i, population in enumerate(sd.populations()):
            if population.metadata:
                has_some_metadata = True  # Check that we do compare something sometimes
            assert population.metadata == sd.population(i).metadata
        assert has_some_metadata
        assert i == sd.num_populations - 1

    def test_from_tree_sequence_bad_times(self):
        n_individuals = 4
        ploidy = 2
        individual_times = np.arange(n_individuals)  # Diploids
        ts = tsutil.get_example_historical_sampled_ts(individual_times, ploidy)
        tables = ts.dump_tables()
        # Associate nodes at different times with a single individual
        nodes_time = tables.nodes.time
        min_time = min(n.time for n in ts.nodes() if not n.is_sample())
        nodes_time[ts.samples()] = np.linspace(0, min_time, n_individuals * ploidy)
        tables.nodes.time = nodes_time
        # Zap out the mutation times to avoid conflicts.
        tables.mutations.time = np.full(ts.num_mutations, tskit.UNKNOWN_TIME)
        bad_ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            formats.SampleData.from_tree_sequence(bad_ts)

    def test_from_tree_sequence_bad_populations(self):
        n_individuals = 4
        ts = tsutil.get_example_ts(n_individuals * 2)  # Diploids
        tables = ts.dump_tables()
        # Associate each sample node with a new population
        for _ in range(n_individuals * 2):
            tables.populations.add_row(metadata={})
        nodes_population = tables.nodes.population
        nodes_population[ts.samples()] = np.arange(n_individuals * 2)
        tables.nodes.population = nodes_population
        for _ in range(n_individuals):
            tables.individuals.add_row(metadata={})
        # Associate nodes with individuals
        nodes_individual = tables.nodes.individual
        nodes_individual[ts.samples()] = np.repeat(
            np.arange(n_individuals, dtype=tables.nodes.individual.dtype), 2
        )
        tables.nodes.individual = nodes_individual
        bad_ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            formats.SampleData.from_tree_sequence(bad_ts)

    def test_from_tree_sequence_simple(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        sd2 = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        assert sd1.data_equal(sd2)

    def test_from_tree_sequence_variable_allele_number(self):
        ts = tsutil.get_example_ts(10)
        # Create > 2 alleles by scattering mutations on the tree nodes at the first site
        tables = ts.dump_tables()
        # We can't have mixed know and unknown times.
        tables.mutations.time = np.full(ts.num_mutations, tskit.UNKNOWN_TIME)
        focal_site = ts.site(0)
        tree = ts.at(focal_site.position)
        # Reset the initial mutation to lie above the root, for correct mutation order
        nodes = tables.mutations.node
        nodes[0] = tree.root
        tables.mutations.node = nodes
        for i, node in enumerate(tree.nodes()):
            if i % 3:  # add above every 3rd node - should create many alleles
                tables.mutations.add_row(site=0, node=node, derived_state=str(i + 2))
        # Create < 2 alleles by adding a non-variable site at the end
        extra_last_pos = (ts.site(ts.num_sites - 1).position + ts.sequence_length) / 2
        tables.sites.add_row(position=extra_last_pos, ancestral_state="0", metadata={})
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        assert len(ts.site(0).mutations) > 1
        assert len(ts.site(ts.num_sites - 1).mutations) == 0
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        num_alleles = sd1.num_alleles()
        for var in ts.variants():
            assert len(var.alleles) == num_alleles[var.site.id]
        sd2 = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        assert sd1.data_equal(sd2)

    def test_from_tree_sequence_with_metadata(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(5, 2)
        # Remove individuals
        tables = ts.dump_tables()
        tables.individuals.clear()
        tables.nodes.individual = np.full(
            ts.num_nodes, tskit.NULL, dtype=tables.nodes.individual.dtype
        )
        ts_no_individuals = tables.tree_sequence()
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts_no_individuals, sd1)
        sd2 = formats.SampleData.from_tree_sequence(
            ts_no_individuals, use_sites_time=True
        )
        assert sd1.data_equal(sd2)

    def test_from_tree_sequence_with_metadata_and_individuals(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(5, ploidy=3)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        sd2 = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        sd1.assert_data_equal(sd2)

    def test_from_historical_tree_sequence_with_times(self):
        n_indiv = 5
        ploidy = 2
        individual_times = np.arange(n_indiv)
        ts = tsutil.get_example_historical_sampled_ts(individual_times, ploidy)
        # Test on a tree seq containing an individual with no nodes
        keep_samples = [u for i in ts.individuals() for u in i.nodes if i.id < n_indiv]
        ts = ts.simplify(samples=keep_samples, filter_individuals=False)
        sd1 = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, sd1)
        sd2 = formats.SampleData.from_tree_sequence(
            ts, use_sites_time=True, use_individuals_time=True
        )
        assert sd1.data_equal(sd2)
        # Fails if use_individuals_time is not set
        sd2 = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        assert not sd1.data_equal(sd2)

    def test_from_tree_sequence_no_times(self):
        n_indiv = 5
        ploidy = 2
        individual_times = np.arange(n_indiv + 1)
        ts = tsutil.get_example_historical_sampled_ts(individual_times, ploidy)
        # Test on a tree seq containing an individual with no nodes
        keep_samples = [u for i in ts.individuals() for u in i.nodes if i.id < n_indiv]
        ts = ts.simplify(samples=keep_samples, filter_individuals=False)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        assert sd1.num_individuals == n_indiv
        assert np.all(sd1.individuals_time[:] == 0)

    def test_from_tree_sequence_time_incompatibilities(self):
        ploidy = 2
        individual_times = np.arange(5)
        ts = tsutil.get_example_historical_sampled_ts(individual_times, ploidy)
        with pytest.raises(ValueError, match="Incompatible timescales"):
            _ = formats.SampleData.from_tree_sequence(ts, use_individuals_time=True)
        # Similar error if no individuals in the TS
        tables = ts.dump_tables()
        tables.individuals.clear()
        tables.nodes.individual = np.full(
            tables.nodes.num_rows, tskit.NULL, dtype=tables.nodes.individual.dtype
        )
        ts = tables.tree_sequence()
        with pytest.raises(ValueError, match="Incompatible timescales"):
            _ = formats.SampleData.from_tree_sequence(ts, use_individuals_time=True)

    def test_chunk_size(self):
        ts = tsutil.get_example_ts(4, mutation_rate=0.005)
        assert ts.num_sites > 50
        for chunk_size in [1, 2, 3, ts.num_sites - 1, ts.num_sites, ts.num_sites + 1]:
            input_file = formats.SampleData(
                sequence_length=ts.sequence_length, chunk_size=chunk_size
            )
            self.verify_data_round_trip(ts, input_file)
            for name, array in input_file.arrays():
                assert array.chunks[0] == chunk_size
                if name.endswith("genotypes"):
                    assert array.chunks[1] == chunk_size

    def test_filename(self):
        ts = tsutil.get_example_ts(14)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            input_file = formats.SampleData(
                sequence_length=ts.sequence_length, path=filename
            )
            assert os.path.exists(filename)
            assert not os.path.isdir(filename)
            self.verify_data_round_trip(ts, input_file)
            # Make a copy so that we can close the original and reopen it
            # without hitting simultaneous file access problems on Windows
            input_copy = input_file.copy(path=None)
            input_file.close()
            other_input_file = formats.SampleData.load(filename)
            assert other_input_file is not input_copy
            # Can't use eq here because UUIDs will not be equal.
            assert other_input_file.data_equal(input_copy)
            other_input_file.close()

    def test_chunk_size_file_equal(self):
        ts = tsutil.get_example_ts(13)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            files = []
            for chunk_size in [5, 7]:
                filename = os.path.join(tempdir, f"samples_{chunk_size}.tmp")
                files.append(filename)
                with formats.SampleData(
                    sequence_length=ts.sequence_length,
                    path=filename,
                    chunk_size=chunk_size,
                ) as input_file:
                    self.verify_data_round_trip(ts, input_file)
                    assert input_file.sites_genotypes.chunks == (chunk_size, chunk_size)
            # Now reload the files and check they are equal
            with formats.SampleData.load(files[0]) as input_file0:
                with formats.SampleData.load(files[1]) as input_file1:
                    # Can't use eq here because UUIDs will not be equal.
                    assert input_file0.data_equal(input_file1)

    @pytest.mark.slow
    def test_compressor(self):
        ts = tsutil.get_example_ts(11, random_seed=123)
        compressors = [
            None,
            formats.DEFAULT_COMPRESSOR,
            blosc.Blosc(cname="zlib", clevel=1, shuffle=blosc.NOSHUFFLE),
        ]
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            for i, compressor in enumerate(compressors):
                filename = os.path.join(tempdir, f"samples_{i}.tmp")
                for path in [None, filename]:
                    with formats.SampleData(
                        sequence_length=ts.sequence_length,
                        path=path,
                        compressor=compressor,
                    ) as samples:
                        self.verify_data_round_trip(ts, samples)
                        for _, array in samples.arrays():
                            assert array.compressor == compressor

    def test_multichar_alleles(self):
        ts = tsutil.get_example_ts(5)
        t = ts.dump_tables()
        t.sites.clear()
        t.mutations.clear()
        for site in ts.sites():
            t.sites.add_row(
                site.position, ancestral_state="A" * (site.id + 1), metadata={}
            )
            for mutation in site.mutations:
                t.mutations.add_row(
                    site=site.id, node=mutation.node, derived_state="T" * site.id
                )
        ts = t.tree_sequence()
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)

    def test_str(self):
        ts = tsutil.get_example_ts(5, random_seed=2)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        assert len(str(input_file)) > 0

    def test_eq(self):
        ts = tsutil.get_example_ts(5, random_seed=3)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        assert input_file == input_file
        assert not (input_file == [])
        assert not ({} == input_file)

    def test_provenance(self):
        ts = tsutil.get_example_ts(4, random_seed=10)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        assert input_file.num_provenances == 1
        timestamp = input_file.provenances_timestamp[0]
        iso = datetime.datetime.now().isoformat()
        assert timestamp.split("T")[0] == iso.split("T")[0]
        record = input_file.provenances_record[0]
        assert record["software"]["name"] == "tsinfer"
        a = list(input_file.provenances())
        assert len(a) == 1
        assert a[0][0] == timestamp
        assert a[0][1] == record

    def test_clear_provenance(self):
        ts = tsutil.get_example_ts(4, random_seed=6)
        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        assert input_file.num_provenances == 1
        with pytest.raises(ValueError):
            input_file.clear_provenances()
        input_file = input_file.copy()
        input_file.clear_provenances()
        assert input_file.num_provenances == 0
        input_file.finalise()
        assert input_file.num_provenances == 0

    def test_variant_errors(self):
        input_file = formats.SampleData(sequence_length=10)
        genotypes = [0, 0]
        input_file.add_site(0, alleles=["0", "1"], genotypes=genotypes)
        for bad_position in [-1, 10, 100]:
            with pytest.raises(ValueError):
                input_file.add_site(
                    position=bad_position, alleles=["0", "1"], genotypes=genotypes
                )
        for bad_genotypes in [[0, 2], [-2, 0], [], [0], [0, 0, 0]]:
            with pytest.raises(ValueError):
                input_file.add_site(
                    position=1, alleles=["0", "1"], genotypes=bad_genotypes
                )
        with pytest.raises(ValueError):
            input_file.add_site(position=1, alleles=["0"], genotypes=[0, 1])
        with pytest.raises(ValueError):
            input_file.add_site(position=1, alleles=["0", "1"], genotypes=[0, 2])
        with pytest.raises(ValueError):
            input_file.add_site(position=1, alleles=["0", "0"], genotypes=[0, 2])

    def test_duplicate_sites(self):
        # Duplicate sites are not accepted.
        input_file = formats.SampleData()
        alleles = ["0", "1"]
        genotypes = [0, 1, 1]
        input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)

    def test_unordered_sites(self):
        # Sites must be specified in sorted order.
        input_file = formats.SampleData()
        alleles = ["0", "1"]
        genotypes = [0, 1, 1]
        input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=0.5, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=0.9999, alleles=alleles, genotypes=genotypes)
        input_file.add_site(position=2, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=0.5, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=1.88, alleles=alleles, genotypes=genotypes)

    def test_insufficient_samples(self):
        sample_data = formats.SampleData(sequence_length=10)
        with pytest.raises(ValueError):
            sample_data.add_site(position=0, alleles=["0", "1"], genotypes=[])
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual(ploidy=3)
        with pytest.raises(ValueError):
            sample_data.add_site(position=0, alleles=["0", "1"], genotypes=[0])
        sample_data = formats.SampleData(sequence_length=10)

    def test_add_population_errors(self):
        sample_data = formats.SampleData(sequence_length=10)
        with pytest.raises(TypeError):
            sample_data.add_population(metadata=234)

    def test_add_state_machine(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual()
        with pytest.raises(ValueError):
            sample_data.add_population()

        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_site(0.1, genotypes=[0, 1])
        with pytest.raises(ValueError):
            sample_data.add_population()
        with pytest.raises(ValueError):
            sample_data.add_individual()

    def test_add_population_return(self):
        sample_data = formats.SampleData(sequence_length=10)
        pid = sample_data.add_population({"a": 1})
        assert pid == 0
        pid = sample_data.add_population()
        assert pid == 1
        pid = sample_data.add_population()
        assert pid == 2

    def test_top_level_metadata(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.metadata = {"a": "b"}
        schema = tsinfer.permissive_json_schema()
        schema["properties"]["xyz"] = {"type": "string"}
        sample_data.metadata_schema = schema
        sample_data.add_site(0, [0, 0])
        sample_data.finalise()
        assert sample_data.metadata == {"a": "b"}
        assert sample_data.metadata_schema == schema

    def test_population_metadata(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_population({"a": 1})
        sample_data.add_population({"b": 2})
        sample_data.add_individual(population=0)
        sample_data.add_individual(population=1)
        sample_data.add_site(position=0, genotypes=[0, 1])
        sample_data.finalise()

        assert sample_data.populations_metadata[0] == {"a": 1}
        assert sample_data.populations_metadata[1] == {"b": 2}
        assert sample_data.individuals_population[0] == 0
        assert sample_data.individuals_population[1] == 1

    def test_individual_metadata(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual(metadata={"a": 1})
        sample_data.add_individual(metadata={"b": 2})
        sample_data.add_site(0, [0, 0])
        sample_data.finalise()
        assert sample_data.individuals_metadata[0] == {"a": 1}
        assert sample_data.individuals_metadata[1] == {"b": 2}

    def test_add_individual_time(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual()
        sample_data.add_individual(time=0.5)
        sample_data.add_site(0, [0, 0])
        sample_data.finalise()
        assert sample_data.individuals_time[0] == 0
        assert sample_data.individuals_time[1] == 0.5

    def test_add_individual_return(self):
        sample_data = formats.SampleData(sequence_length=10)
        iid, sids = sample_data.add_individual()
        assert iid == 0
        assert sids == [0]
        iid, sids = sample_data.add_individual(ploidy=1)
        assert iid == 1
        assert sids == [1]
        iid, sids = sample_data.add_individual(ploidy=5)
        assert iid == 2
        assert sids == [2, 3, 4, 5, 6]

    def test_numpy_position(self):
        pos = np.array([5.1, 100], dtype=np.float64)
        with formats.SampleData() as sample_data:
            sample_data.add_site(pos[0], [0, 0])
        assert sample_data.sequence_length == 6.1
        with formats.SampleData(sequence_length=pos[1]) as sample_data:
            sample_data.add_site(pos[0], [0, 0])
        assert sample_data.sequence_length == 100

    def test_get_single_individual(self):
        with formats.SampleData(sequence_length=10) as sample_data:
            sample_data.add_individual(
                ploidy=1, location=[0, 0], metadata={"name": "zero"}
            )
            sample_data.add_individual(
                ploidy=2, location=[1, 1], metadata={"name": "one"}
            )
            sample_data.add_site(0, [0, 0, 0])
        assert sample_data.individual(1).id == 1
        assert np.array_equal(sample_data.individual(1).location, [1, 1])
        assert sample_data.individual(1).metadata == {"name": "one"}
        assert np.sum(sample_data.samples_individual[:] == 1) == 2

    def test_add_individual_errors(self):
        sample_data = formats.SampleData(sequence_length=10)
        with pytest.raises(TypeError):
            sample_data.add_individual(metadata=234)
        with pytest.raises(ValueError):
            sample_data.add_individual(population=0)
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_population()
        with pytest.raises(ValueError):
            sample_data.add_individual(population=1)
        with pytest.raises(ValueError):
            sample_data.add_individual(location="x234")
        with pytest.raises(ValueError):
            sample_data.add_individual(ploidy=0)
        with pytest.raises(ValueError):
            sample_data.add_individual(time=None)
        with pytest.raises(ValueError):
            sample_data.add_individual(time=[1, 2])

    def test_no_data(self):
        sample_data = formats.SampleData(sequence_length=10)
        with pytest.raises(ValueError):
            sample_data.finalise()

    def test_no_sites(self):
        sample_data = formats.SampleData(sequence_length=10)
        sample_data.add_individual()
        with pytest.raises(ValueError):
            sample_data.finalise()

    def test_add_site_return(self):
        sample_data = formats.SampleData(sequence_length=10)
        sid = sample_data.add_site(0, [0, 1])
        assert sid == 0
        sid = sample_data.add_site(1, [0, 1])
        assert sid == 1

    def test_sites(self):
        ts = tsutil.get_example_ts(11)
        assert ts.num_sites > 1
        input_file = formats.SampleData.from_tree_sequence(ts)

        all_sites = input_file.sites()
        for s1, variant in zip(ts.sites(), ts.variants()):
            s2 = next(all_sites)
            assert s1.id == s2.id
            assert s1.position == s2.position
            assert s1.ancestral_state == s2.ancestral_state
            assert variant.alleles == s2.alleles
        assert next(all_sites, None) is None, None

    def test_sites_subset(self):
        ts = tsutil.get_example_ts(11)
        assert ts.num_sites > 1
        input_file = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        assert list(input_file.sites([])) == []
        index = np.arange(input_file.num_sites)
        site_list = list(input_file.sites())
        assert list(input_file.sites(index)) == site_list
        assert list(input_file.sites(index[::-1])) == site_list[::-1]
        index = np.arange(input_file.num_sites // 2)
        result = list(input_file.sites(index))
        assert len(result) == len(index)
        for site, j in zip(result, index):
            assert site == site_list[j]

        with pytest.raises(IndexError):
            list(input_file.sites([10000]))

    def test_variants(self):
        ts = tsutil.get_example_ts(11)
        assert ts.num_sites > 1
        input_file = formats.SampleData.from_tree_sequence(ts)

        all_variants = input_file.variants()
        for v1 in ts.variants():
            v2 = next(all_variants)
            assert v1.site.id == v2.site.id
            assert v1.site.position == v2.site.position
            assert v1.site.ancestral_state == v2.site.ancestral_state
            assert v1.alleles == v2.alleles
            assert np.array_equal(v1.genotypes, v2.genotypes)
        assert next(all_variants, None) is None, None

    def test_variants_subset_sites(self):
        ts = tsutil.get_example_ts(4, mutation_rate=0.004)
        assert ts.num_sites > 50
        for chunk_size in [1, 2, 3, ts.num_sites - 1, ts.num_sites, ts.num_sites + 1]:
            input_file = formats.SampleData(
                sequence_length=ts.sequence_length, chunk_size=chunk_size
            )
            self.verify_data_round_trip(ts, input_file)
            # Bad lowest value
            v = input_file.variants(sites=[-1, 0, 2])
            with pytest.raises(ValueError):
                next(v)
            # Bad order
            v = input_file.variants(sites=[20, 0, 40])
            with pytest.raises(ValueError):
                next(v)
            # Bad highest value
            v = input_file.variants(sites=[0, 20, ts.num_sites])
            with pytest.raises(ValueError):
                next(v)
            sites = [0, 20, 40]
            v = input_file.variants(sites=sites)
            for every_variant in input_file.variants():
                if every_variant.site in sites:
                    assert every_variant == next(v)

    def test_all_haplotypes(self):
        ts = tsutil.get_example_ts(13, random_seed=111)
        assert ts.num_sites > 1
        input_file = formats.SampleData.from_tree_sequence(ts)

        G = ts.genotype_matrix()
        j = 0
        for index, h in input_file.haplotypes():
            assert np.array_equal(h, G[:, j])
            assert index == j
            j += 1
        assert j == ts.num_samples

        j = 0
        for index, h in input_file.haplotypes(np.arange(ts.num_samples)):
            assert np.array_equal(h, G[:, j])
            assert index == j
            j += 1
        assert j == ts.num_samples

        j = 0
        for index, h in input_file.haplotypes(sites=np.arange(ts.num_sites)):
            assert np.array_equal(h, G[:, j])
            assert index == j
            j += 1
        assert j == ts.num_samples

    def test_haplotypes_index_errors(self):
        ts = tsutil.get_example_ts(13, random_seed=19)
        assert ts.num_sites > 1
        input_file = formats.SampleData.from_tree_sequence(ts)
        with pytest.raises(ValueError):
            list(input_file.haplotypes([1, 0]))
        with pytest.raises(ValueError):
            list(input_file.haplotypes([0, 1, 2, -1]))
        with pytest.raises(ValueError):
            list(input_file.haplotypes([0, 1, 2, 2]))
        with pytest.raises(ValueError):
            list(input_file.haplotypes(np.arange(10)[::-1]))

        # Out of bounds sample index.
        with pytest.raises(ValueError):
            list(input_file.haplotypes([13]))
        with pytest.raises(ValueError):
            list(input_file.haplotypes([3, 14]))

    def test_haplotypes_subsets(self):
        ts = tsutil.get_example_ts(25)
        assert ts.num_sites > 1
        input_file = formats.SampleData.from_tree_sequence(ts)

        subsets = [
            [],
            [0],
            [1],
            [21],
            [22],
            [0, 1],
            [1, 2],
            [4, 5],
            [10, 11],
            [23, 24],
            [0, 1, 2],
            [1, 2, 3],
            [4, 5, 6],
            [1, 10, 20],
            [0, 1, 2, 3],
            [0, 10, 11, 20],
            [10, 15, 20, 21],
            np.arange(24),
            1 + np.arange(24),
            np.arange(25),
        ]
        G = ts.genotype_matrix()
        for subset in subsets:
            j = 0
            for index, h in input_file.haplotypes(subset):
                assert np.array_equal(h, G[:, subset[j]])
                assert index == subset[j]
                j += 1
            assert j == len(subset)

    def test_ts_with_invariant_sites(self):
        ts = tsutil.get_example_ts(5)
        t = ts.dump_tables()
        positions = {site.position for site in ts.sites()}
        for j in range(10):
            pos = 1 / (j + 1)
            if pos not in positions:
                t.sites.add_row(position=pos, ancestral_state="0", metadata={})
                positions.add(pos)
        assert len(positions) > ts.num_sites
        t.sort()
        ts = t.tree_sequence()

        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)
        assert len(str(input_file)) > 0

    def test_ts_with_root_mutations(self):
        ts = tsutil.get_example_ts(5)
        t = ts.dump_tables()
        positions = {site.position for site in ts.sites()}
        for tree in ts.trees():
            pos = tree.interval[0]
            if pos not in positions:
                site_id = t.sites.add_row(
                    position=pos, ancestral_state="0", metadata={}
                )
                t.mutations.add_row(site=site_id, node=tree.root, derived_state="1")
                positions.add(pos)
        assert len(positions) > ts.num_sites
        t.sort()
        ts = t.tree_sequence()

        input_file = formats.SampleData(sequence_length=ts.sequence_length)
        self.verify_data_round_trip(ts, input_file)

    def test_copy_error_wrong_mode(self):
        data = formats.SampleData()
        with pytest.raises(ValueError):
            data.copy()
        data = formats.SampleData()
        with pytest.raises(ValueError):
            data.copy()
        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1])
        data.finalise()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Zarr emits a FutureWarning about object arrays here.
            copy = data.copy()
        with pytest.raises(ValueError):
            copy.copy()

    def test_error_not_build_mode(self):
        # Cannot build after finalising.
        with formats.SampleData() as input_file:
            alleles = ["0", "1"]
            genotypes = [0, 1, 1]
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_site(position=1, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.add_provenance(datetime.datetime.now().isoformat(), {})
        with pytest.raises(ValueError):
            input_file._check_build_mode()

    def test_error_not_write_mode(self):
        # Cannot finalise twice.
        with formats.SampleData() as input_file:
            alleles = ["0", "1"]
            genotypes = [0, 1, 1]
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        with pytest.raises(ValueError):
            input_file.finalise()
        with pytest.raises(ValueError):
            input_file._check_write_modes()

    def test_error_not_edit_mode(self):
        # Can edit after copy but not after finalising.
        with formats.SampleData() as input_file:
            alleles = ["0", "1"]
            genotypes = [0, 1, 1]
            input_file.add_site(position=0, alleles=alleles, genotypes=genotypes)
        editable_sample_data = input_file.copy()
        # Try editing: use setter in the normal way
        editable_sample_data.sites_time = [0.0]
        # Try editing: use setter via setattr
        setattr(editable_sample_data, "sites_time", [1.0])  # noqa: B010
        editable_sample_data.add_provenance(datetime.datetime.now().isoformat(), {})
        editable_sample_data.finalise()
        with pytest.raises(ValueError):
            setattr(editable_sample_data, "sites_time", [0.0])  # noqa: B010
        with pytest.raises(ValueError):
            editable_sample_data.add_provenance(
                datetime.datetime.now().isoformat(),
                {},
            )

    def test_copy_new_uuid(self):
        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1])
        data.finalise()
        copy = data.copy()
        copy.finalise()
        assert copy.uuid != data.uuid
        assert copy.data_equal(data)

    def test_copy_update_sites_time(self):
        with formats.SampleData() as data:
            for j in range(4):
                data.add_site(
                    position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0], time=0.5
                )
        assert list(data.sites_time) == [0.5, 0.5, 0.5, 0.5]

        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for copy_path in [None, filename]:
                copy = data.copy(path=copy_path)
                copy.finalise()
                assert copy.data_equal(data)
                if copy_path is not None:
                    copy.close()
                with data.copy(path=copy_path) as copy:
                    time = [0.0, 1.1, 2.2, 3.3]
                    copy.sites_time = time
                assert not copy.data_equal(data)
                assert list(copy.sites_time) == time
                assert list(data.sites_time) == [0.5, 0.5, 0.5, 0.5]
                if copy_path is not None:
                    copy.close()

    def test_update_sites_time_bad_data(self):
        def set_value(data, value):
            data.sites_time = value

        data = formats.SampleData()
        for j in range(4):
            data.add_site(
                position=j, alleles=["0", "1"], genotypes=[0, 1, 1, 0], time=0.5
            )
        data.finalise()
        assert list(data.sites_time) == [0.5, 0.5, 0.5, 0.5]
        copy = data.copy()
        for bad_shape in [[], np.arange(100, dtype=np.float64), np.zeros((2, 2))]:
            with pytest.raises((ValueError, TypeError)):
                set_value(copy, bad_shape)
        for bad_data in [["a", "b", "c", "d"]]:
            with pytest.raises(ValueError):
                set_value(copy, bad_data)

    def test_update_sites_time_non_copy_mode(self):
        def set_value(data, value):
            data.sites_time = value

        data = formats.SampleData()
        data.add_site(position=0, alleles=["0", "1"], genotypes=[0, 1, 1, 0])
        with pytest.raises(ValueError):
            set_value(data, [1.0])
        data.finalise()
        with pytest.raises(ValueError):
            set_value(data, [1.0])

    @pytest.mark.skipif(
        IS_WINDOWS, reason="windows simultaneous file permissions issue"
    )
    def test_overwrite_partial(self):
        # Check that we can correctly overwrite partially written and
        # unfinalised files. See
        # https://github.com/tskit-dev/tsinfer/issues/64
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            for _ in range(10):
                data = formats.SampleData(path=filename)
                for j in range(10):
                    data.add_site(j, [0, 1])

    def test_sequence_length(self):
        data = formats.SampleData(sequence_length=2)
        data.add_site(position=0, genotypes=[0, 1, 1, 0])
        data.finalise()
        assert data.sequence_length == 2
        # The default sequence length should be the last site + 1.
        data = formats.SampleData()
        data.add_site(position=0, genotypes=[0, 1, 1, 0])
        data.finalise()
        assert data.sequence_length == 1

    def test_too_many_alleles(self):
        with tsinfer.SampleData() as sample_data:
            sample_data.add_site(0, [0, 0], alleles=[str(x) for x in range(64)])
            for num_alleles in [65, 66, 100]:
                with pytest.raises(ValueError):
                    sample_data.add_site(
                        0, [0, 0], alleles=[str(x) for x in range(num_alleles)]
                    )

    def test_num_alleles_with_missing(self):
        u = tskit.MISSING_DATA
        sites_by_samples = np.array(
            [
                [u, u, u, 1],
                [u, u, u, 1],
                [u, u, u, 1],
                [u, 0, 0, 1],
                [u, 0, 1, 1],
                [u, 0, 1, 0],
            ],
            dtype=np.int8,
        )
        with tsinfer.SampleData() as sd:
            for col in range(sites_by_samples.shape[1]):
                genos = sites_by_samples[:, col]
                alleles = [None if x == u else str(x) for x in np.unique(genos)]
                if alleles[0] is None:
                    alleles = alleles[1:] + [None]  # Put None at the end
                sd.add_site(col, genos, alleles=alleles)
        assert np.all(sd.num_alleles() == np.array([0, 1, 2, 2]))

    def test_append_sites(self):
        pos = [0, 2000, 5000, 10000]
        ts = tsutil.get_example_individuals_ts_with_metadata(4, sequence_length=pos[-1])
        sd1 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[0:2]]))
        sd2 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[1:3]]))
        sd3 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[2:]]))
        sd = sd1.copy()  # put into write mode
        sd.append_sites(sd2, sd3)
        sd.finalise()
        sd.assert_data_equal(tsinfer.SampleData.from_tree_sequence(ts))
        # Test that the full file passes though invisibly if no args given
        sd_full = sd.copy()
        sd_full.append_sites()
        sd_full.finalise()
        sd_full.assert_data_equal(tsinfer.SampleData.from_tree_sequence(ts))

    def test_append_sites_bad_order(self):
        pos = [0, 2000, 5000, 10000]
        ts = tsutil.get_example_individuals_ts_with_metadata(4, sequence_length=pos[-1])
        sd1 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[0:2]]))
        sd2 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[1:3]]))
        sd3 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[2:]]))
        sd = sd1.copy()  # put into write mode
        with pytest.raises(ValueError, match="ascending"):
            sd.append_sites(sd3, sd2)

    def test_append_sites_incompatible_files(self):
        pos = [0, 2000, 5000, 10000]
        ts = tsutil.get_example_individuals_ts_with_metadata(4, sequence_length=pos[-1])
        sd1 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[0:2]]))
        mid_ts = ts.keep_intervals([pos[1:3]])
        sd2 = tsinfer.SampleData.from_tree_sequence(mid_ts)
        sd3 = tsinfer.SampleData.from_tree_sequence(ts.keep_intervals([pos[2:]]))
        # Fails if altered SD is not in write mode
        with pytest.raises(ValueError, match="build"):
            sd1.append_sites(sd2, sd3)
        # Fails if added SDs are in write mode
        sd = sd1.copy()  # put into write mode
        sd.append_sites(sd2, sd3)  # now works
        with pytest.raises(ValueError, match="finalise"):
            sd.append_sites(sd2.copy(), sd3)
        sd = sd1.copy()  # put into write mode

        # Wrong seq length
        sd2 = tsinfer.SampleData.from_tree_sequence(mid_ts.rtrim())
        with pytest.raises(ValueError, match="length"):
            sd.append_sites(sd2, sd3)
        # Wrong num samples
        sd2 = tsinfer.SampleData.from_tree_sequence(mid_ts.simplify(list(range(7))))
        with pytest.raises(ValueError, match="samples"):
            sd.append_sites(sd2, sd3)
        # Wrong individuals
        tables = mid_ts.dump_tables()
        tables.individuals.packset_metadata([b"{}"] * mid_ts.num_individuals)
        sd2 = tsinfer.SampleData.from_tree_sequence(tables.tree_sequence())
        with pytest.raises(ValueError, match="individuals"):
            sd.append_sites(sd2, sd3)
        # Wrong populations
        tables = mid_ts.dump_tables()
        tables.populations.packset_metadata([b"{}"] * mid_ts.num_populations)
        sd2 = tsinfer.SampleData.from_tree_sequence(tables.tree_sequence())
        with pytest.raises(ValueError, match="populations"):
            sd.append_sites(sd2, sd3)
        # Wrong format version
        sd.data.attrs[tsinfer.FORMAT_VERSION_KEY] = (-1, 0)
        sd2 = tsinfer.SampleData.from_tree_sequence(mid_ts)
        with pytest.raises(ValueError, match="format"):
            sd.append_sites(sd2, sd3)

    def test_bad_format_version(self):
        with tsinfer.SampleData() as sd:
            sd.add_site(0, [0, 0])
        # This is the easiest way to trigger the format check
        # - it's all a bit convoluted.
        with pytest.raises(tsinfer.FileFormatError):
            with sd.copy() as copy:
                copy.data.attrs[tsinfer.FORMAT_NAME_KEY] = "xyz"
        with pytest.raises(tsinfer.FileFormatTooOld):
            with sd.copy() as copy:
                copy.data.attrs[tsinfer.FORMAT_VERSION_KEY] = 1, 0

        with pytest.raises(tsinfer.FileFormatTooNew):
            with sd.copy() as copy:
                copy.data.attrs[tsinfer.FORMAT_VERSION_KEY] = 100, 0


class TestSampleDataMetadataSchemas:
    def test_metadata_schemas_default(self):
        with formats.SampleData() as sample_data:
            sample_data.add_site(0, [0, 0])
        assert sample_data.metadata_schema == tsinfer.permissive_json_schema()
        # Tables default to None for backward compatibility.
        assert sample_data.populations_metadata_schema is None
        assert sample_data.individuals_metadata_schema is None
        assert sample_data.sites_metadata_schema is None

    def test_metadata_schemas_non_json_codec(self):
        bad_schema = tsinfer.permissive_json_schema()
        bad_schema["codec"] = "struct"
        with formats.SampleData() as sample_data:
            sample_data.add_site(0, [0, 0])
            with pytest.raises(ValueError):
                sample_data.metadata_schema = bad_schema
            with pytest.raises(ValueError):
                sample_data.populations_metadata_schema = bad_schema
            with pytest.raises(ValueError):
                sample_data.individuals_metadata_schema = bad_schema
            with pytest.raises(ValueError):
                sample_data.sites_metadata_schema = bad_schema

    def test_set_top_level_metadata_schema(self):
        example_schema = tsinfer.permissive_json_schema()
        example_schema["properties"]["xyz"] = {"type": "string"}

        with formats.SampleData() as sample_data:
            assert sample_data.metadata_schema == tsinfer.permissive_json_schema()
            sample_data.metadata_schema = example_schema
            assert sample_data.metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
        assert sample_data.metadata_schema == example_schema

        with formats.SampleData() as sample_data:
            sample_data.add_site(0, [0, 0])
            with pytest.raises(ValueError):
                sample_data.metadata_schema = None
        assert sample_data.metadata_schema == tsinfer.permissive_json_schema()

    def test_set_population_metadata_schema(self):
        example_schema = tsinfer.permissive_json_schema()
        with formats.SampleData() as sample_data:
            assert sample_data.populations_metadata_schema is None
            sample_data.populations_metadata_schema = example_schema
            assert sample_data.populations_metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
        assert sample_data.populations_metadata_schema == example_schema

        with formats.SampleData() as sample_data:
            assert sample_data.populations_metadata_schema is None
            sample_data.populations_metadata_schema = example_schema
            assert sample_data.populations_metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
            sample_data.populations_metadata_schema = None
        assert sample_data.populations_metadata_schema is None

    def test_set_individual_metadata_schema(self):
        example_schema = tsinfer.permissive_json_schema()
        with formats.SampleData() as sample_data:
            assert sample_data.individuals_metadata_schema is None
            sample_data.individuals_metadata_schema = example_schema
            assert sample_data.individuals_metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
        assert sample_data.individuals_metadata_schema == example_schema

        with formats.SampleData() as sample_data:
            assert sample_data.individuals_metadata_schema is None
            sample_data.individuals_metadata_schema = example_schema
            assert sample_data.individuals_metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
            sample_data.individuals_metadata_schema = None
        assert sample_data.individuals_metadata_schema is None

    def test_set_site_metadata_schema(self):
        example_schema = tsinfer.permissive_json_schema()
        with formats.SampleData() as sample_data:
            assert sample_data.sites_metadata_schema is None
            sample_data.sites_metadata_schema = example_schema
            assert sample_data.sites_metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
        assert sample_data.sites_metadata_schema == example_schema

        with formats.SampleData() as sample_data:
            assert sample_data.sites_metadata_schema is None
            sample_data.sites_metadata_schema = example_schema
            assert sample_data.sites_metadata_schema == example_schema
            sample_data.add_site(0, [0, 0])
            sample_data.sites_metadata_schema = None
        assert sample_data.sites_metadata_schema is None


class TestSampleDataSubset:
    """
    Tests for the subset() method of SampleData.
    """

    def test_no_arguments(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        # No arguments gives the same data
        subset = sd1.subset()
        sd1.assert_data_equal(subset)
        subset = sd1.subset(individuals=np.arange(sd1.num_individuals))
        sd1.assert_data_equal(subset)
        subset = sd1.subset(sites=np.arange(sd1.num_sites))
        sd1.assert_data_equal(subset)
        assert subset.num_provenances == sd1.num_provenances + 1

    def verify_subset_data(self, source, individuals, sites):
        subset = source.subset(individuals, sites)
        assert source.sequence_length == subset.sequence_length
        assert subset.num_individuals == len(individuals)
        assert subset.populations_equal(source)
        assert subset.num_sites == len(sites)
        assert subset.num_populations == source.num_populations
        samples = []
        subset_inds = iter(subset.individuals())
        for ind_id in individuals:
            ind = source.individual(ind_id)
            samples.extend(ind.samples)
            subset_ind = next(subset_inds)
            assert subset_ind.time == ind.time
            assert subset_ind.location == ind.location
            assert subset_ind.metadata == ind.metadata
            assert len(subset_ind.samples) == len(ind.samples)
        assert len(samples) == subset.num_samples

        subset_variants = iter(subset.variants())
        j = 0
        for source_var in source.variants():
            if source_var.site.id in sites:
                subset_var = next(subset_variants)
                assert subset_var.site.id == j
                assert source_var.site.position == subset_var.site.position
                assert (
                    source_var.site.ancestral_state == subset_var.site.ancestral_state
                )
                assert source_var.site.metadata == subset_var.site.metadata
                assert source_var.site.alleles == subset_var.site.alleles
                assert np.array_equal(
                    source_var.genotypes[samples], subset_var.genotypes
                )
                j += 1
        assert j == len(sites)

    def test_one_sample(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        G1 = ts.genotype_matrix()
        # Because this is a haploid tree sequence we can use the
        # individual and sample IDs interchangably.
        cols = [3]
        rows = np.arange(ts.num_sites)
        subset = sd1.subset(individuals=cols, sites=rows)
        G2 = np.array([v.genotypes for v in subset.variants()])
        assert np.array_equal(G1[rows][:, cols], G2)
        self.verify_subset_data(sd1, cols, rows)

    def test_simple_case(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        G1 = ts.genotype_matrix()
        # Because this is a haploid tree sequence we can use the
        # individual and sample IDs interchangably.
        cols = [0, 3, 5]
        rows = [1, 7, 8, 10]
        subset = sd1.subset(individuals=cols, sites=rows)
        G2 = np.array([v.genotypes for v in subset.variants()])
        assert np.array_equal(G1[rows][:, cols], G2)
        self.verify_subset_data(sd1, cols, rows)

    def test_reordering_individuals(self):
        ts = tsutil.get_example_ts(10)
        sd = formats.SampleData.from_tree_sequence(ts)
        ind = np.arange(sd.num_individuals)[::-1]
        subset = sd.subset(individuals=ind)
        assert not sd.data_equal(subset)
        assert np.array_equal(sd.sites_genotypes[:][:, ind], subset.sites_genotypes[:])

    def test_mixed_diploid_metadata(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(10, ploidy=2)
        sd = formats.SampleData.from_tree_sequence(ts)
        N = sd.num_individuals
        M = sd.num_sites
        self.verify_subset_data(sd, range(N), range(M))
        self.verify_subset_data(sd, range(N)[::-1], range(M)[::-1])
        self.verify_subset_data(sd, range(N // 2), range(M // 2))
        self.verify_subset_data(sd, range(N // 2, N), range(M // 2, M))
        self.verify_subset_data(sd, [0, 1], range(M))
        self.verify_subset_data(sd, [1, 0], range(M))
        self.verify_subset_data(sd, [0, N - 1], range(M))

    def test_mixed_triploid_metadata(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(18, ploidy=3)
        sd = formats.SampleData.from_tree_sequence(ts)
        N = sd.num_individuals
        M = sd.num_sites
        self.verify_subset_data(sd, range(N), range(M))
        self.verify_subset_data(sd, range(N)[::-1], range(M)[::-1])
        self.verify_subset_data(sd, range(N // 2), range(M // 2))
        self.verify_subset_data(sd, range(N // 2, N), range(M // 2, M))
        self.verify_subset_data(sd, [0, 1], range(M))
        self.verify_subset_data(sd, [1, 0], range(M))
        self.verify_subset_data(sd, [0, N - 1], range(M))

    def test_errors(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        with pytest.raises(ValueError):
            sd1.subset(sites=[])
        with pytest.raises(ValueError):
            sd1.subset(individuals=[])
        # Individual IDs out of bounds

        with pytest.raises(ValueError):
            sd1.subset(individuals=[-1, 0, 1])
        with pytest.raises(ValueError):
            sd1.subset(individuals=[10, 0, 1])
        # Site IDs out of bounds
        with pytest.raises(ValueError):
            sd1.subset(sites=[-1, 0, 1])
        with pytest.raises(ValueError):
            sd1.subset(sites=[ts.num_sites, 0, 1])
        # Duplicate IDs
        with pytest.raises(ValueError):
            sd1.subset(individuals=[0, 0])
        with pytest.raises(ValueError):
            sd1.subset(sites=[0, 0])

    @pytest.mark.skipif(
        IS_WINDOWS, reason="windows simultaneous file permissions issue"
    )
    def test_file_kwargs(self):
        # Make sure we pass kwards on to the SampleData constructor as
        # required.
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample-data")
            sd1.subset(path=path)
            assert os.path.exists(path)
            sd2 = formats.SampleData.load(path)
            assert sd1.data_equal(sd2)

    def test_sequence_length_change(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        max_site_to_use = sd1.num_sites // 2
        new_seq_length = sd1.sites_position[max_site_to_use + 1]
        assert new_seq_length < sd1.sequence_length
        assert max_site_to_use > 0
        sd2 = sd1.subset(
            sites=np.arange(max_site_to_use),
            sequence_length=new_seq_length,
        )
        assert sd2.sequence_length == new_seq_length


class TestSampleDataMerge:
    """
    Tests for the sample data merge operation.
    """

    def test_finalised(self):
        ts1 = tsutil.get_example_ts(2)
        sd1 = formats.SampleData.from_tree_sequence(ts1)
        sd1_copy = sd1.copy()
        with pytest.raises(ValueError, match="not finalised"):
            sd1_copy.merge(sd1)
        with pytest.raises(ValueError, match="not finalised"):
            sd1.merge(sd1_copy)

    def test_different_sequence_lengths(self):
        ts1 = tsutil.get_example_ts(2, sequence_length=10000)
        sd1 = formats.SampleData.from_tree_sequence(ts1)
        ts2 = tsutil.get_example_ts(2, sequence_length=10001)
        sd2 = formats.SampleData.from_tree_sequence(ts2)
        with pytest.raises(ValueError):
            sd1.merge(sd2)

    def test_mismatch_ancestral_state(self):
        # Difference ancestral states
        ts = tsutil.get_example_ts(2)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        tables = ts.dump_tables()
        tables.sites.ancestral_state += 2
        sd2 = formats.SampleData.from_tree_sequence(tables.tree_sequence())
        with pytest.raises(ValueError):
            sd1.merge(sd2)

    def verify(self, sd1, sd2):
        sd3 = sd1.merge(sd2)
        n1 = sd1.num_samples
        n2 = sd2.num_samples
        assert sd3.num_samples == n1 + n2
        assert sd3.num_individuals == sd1.num_individuals + sd2.num_individuals
        assert sd1.sequence_length == sd3.sequence_length
        assert sd2.sequence_length == sd3.sequence_length

        for new_ind, old_ind in zip(sd3.individuals(), sd1.individuals()):
            assert new_ind == old_ind
        new_inds = list(sd3.individuals())[sd1.num_individuals :]
        old_inds = list(sd2.individuals())
        assert len(new_inds) == len(old_inds)
        for new_ind, old_ind in zip(new_inds, old_inds):
            assert new_ind.id == old_ind.id + sd1.num_individuals
            old_samples = [sid + sd1.num_samples for sid in old_ind.samples]
            assert new_ind.samples == old_samples
            assert new_ind.location == old_ind.location
            assert new_ind.time == old_ind.time
            assert new_ind.metadata == old_ind.metadata
            assert new_ind.population == old_ind.population + sd1.num_populations

        for new_sample, old_sample in zip(sd3.samples(), sd1.samples()):
            assert new_sample == old_sample
        new_samples = list(sd3.samples())[sd1.num_samples :]
        old_samples = list(sd2.samples())
        assert len(new_samples) == len(old_samples)
        for new_sample, old_sample in zip(new_samples, old_samples):
            assert new_sample.id == old_sample.id + sd1.num_samples
            assert new_sample.individual == old_sample.individual + sd1.num_individuals

        for new_pop, old_pop in zip(sd3.populations(), sd1.populations()):
            assert new_pop == old_pop
        new_pops = list(sd3.populations())[sd1.num_populations :]
        old_pops = list(sd2.populations())
        assert len(new_pops) == len(old_pops)
        for new_pop, old_pop in zip(new_pops, old_pops):
            assert new_pop.id == old_pop.id + sd1.num_populations
            assert new_pop.metadata == old_pop.metadata

        sd1_sites = set(sd1.sites_position)
        sd2_sites = set(sd2.sites_position)
        assert set(sd3.sites_position) == sd1_sites | sd2_sites
        sd1_variants = {var.site.position: var for var in sd1.variants()}
        sd2_variants = {var.site.position: var for var in sd2.variants()}
        for var in sd3.variants():
            pos = var.site.position
            sd1_var = sd1_variants.get(pos, None)
            sd2_var = sd2_variants.get(pos, None)
            if sd1_var is not None and sd2_var is not None:
                assert var.site.ancestral_state == sd1_var.site.ancestral_state
                assert np.array_equal(var.site.time, sd1_var.site.time, equal_nan=True)
                assert np.array_equal(var.site.time, sd2_var.site.time, equal_nan=True)
                assert var.site.metadata == sd1_var.site.metadata
                assert var.site.metadata == sd2_var.site.metadata
                alleles = {}
                missing_data = False
                for allele in sd1_var.site.alleles + sd2_var.site.alleles:
                    if allele is not None:
                        alleles[allele] = len(alleles)
                    else:
                        missing_data = True
                if missing_data:
                    alleles[None] = len(alleles)
                assert var.site.alleles == tuple(alleles.keys())
                for j in range(n1):
                    assert var.genotypes[j] == sd1_var.genotypes[j]
                for j in range(n2):
                    old_allele = sd2_var.site.alleles[sd2_var.genotypes[j]]
                    if old_allele is None:
                        assert var.genotypes[n1 + j] == tskit.MISSING_DATA
                    else:
                        new_allele = var.site.alleles[var.genotypes[n1 + j]]
                        assert new_allele == old_allele

    def test_merge_identical(self):
        n = 10
        ts = tsutil.get_example_ts(n)
        sd1 = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        sd2 = sd1.merge(sd1)
        assert sd2.num_sites == sd1.num_sites
        assert sd2.num_samples == 2 * sd1.num_samples
        for var1, var2 in zip(sd1.variants(), sd2.variants()):
            assert var1.site == var2.site
            assert np.array_equal(var1.genotypes, var2.genotypes[:n])
            assert np.array_equal(var1.genotypes, var2.genotypes[n:])
        self.verify(sd1, sd1)
        self.verify(sd2, sd1)

    def test_merge_distinct(self):
        n = 10
        ts = tsutil.get_example_ts(n, random_seed=1)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        ts = tsutil.get_example_ts(n, random_seed=2)
        sd2 = formats.SampleData.from_tree_sequence(ts)
        assert len(set(sd1.sites_position) & set(sd2.sites_position)) == 0

        sd3 = sd1.merge(sd2)
        assert sd3.num_sites == sd1.num_sites + sd2.num_sites
        assert sd3.num_samples == sd1.num_samples + sd2.num_samples
        pos_map = {var.site.position: var for var in sd3.variants()}
        for var in sd1.variants():
            merged_var = pos_map[var.site.position]
            assert merged_var.site.position == var.site.position
            assert merged_var.site.alleles[:-1] == var.site.alleles
            assert merged_var.site.ancestral_state == var.site.ancestral_state
            assert merged_var.site.metadata == var.site.metadata
            assert merged_var.site.alleles[-1] is None
            assert np.array_equal(var.genotypes, merged_var.genotypes[:n])
            assert np.all(merged_var.genotypes[n:] == tskit.MISSING_DATA)
        for var in sd2.variants():
            merged_var = pos_map[var.site.position]
            assert merged_var.site.position == var.site.position
            assert merged_var.site.alleles[:-1] == var.site.alleles
            assert merged_var.site.ancestral_state == var.site.ancestral_state
            assert merged_var.site.metadata == var.site.metadata
            assert merged_var.site.alleles[-1] is None
            assert np.array_equal(var.genotypes, merged_var.genotypes[n:])
            assert np.all(merged_var.genotypes[:n] == tskit.MISSING_DATA)
        self.verify(sd1, sd2)
        self.verify(sd2, sd1)

    def test_merge_overlapping_sites(self):
        ts = tsutil.get_example_ts(4, random_seed=1)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        tables = ts.dump_tables()
        # Change the position of the first and last sites to we have
        # overhangs at either side.
        position = tables.sites.position
        position[0] += 1e-8
        position[-1] -= 1e-8
        tables.sites.position = position
        ts = tables.tree_sequence()
        sd2 = formats.SampleData.from_tree_sequence(ts)
        assert (
            len(set(sd1.sites_position) & set(sd2.sites_position)) == sd1.num_sites - 2
        )
        self.verify(sd1, sd2)
        self.verify(sd2, sd1)

    def test_individuals_metadata_identical(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(5)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        self.verify(sd1, sd1)

    def test_individuals_metadata_distinct(self):
        ts = tsutil.get_example_individuals_ts_with_metadata(
            5, ploidy=2, discrete_genome=False
        )  # Put sites at different positions
        sd1 = formats.SampleData.from_tree_sequence(ts)
        ts = tsutil.get_example_individuals_ts_with_metadata(
            3, ploidy=3, discrete_genome=False
        )
        sd2 = formats.SampleData.from_tree_sequence(ts)
        assert len(set(sd1.sites_position) & set(sd2.sites_position)) == 0
        self.verify(sd1, sd2)
        self.verify(sd2, sd1)

    def test_different_alleles_same_sites(self):
        ts = tsutil.get_example_ts(5, mutation_model=msprime.BinaryMutationModel())
        sd1 = formats.SampleData.from_tree_sequence(ts)
        tables = ts.dump_tables()
        tables.mutations.derived_state += 1  # "0" -> "1", "1"-> "2", etc
        sd2 = formats.SampleData.from_tree_sequence(tables.tree_sequence())
        self.verify(sd1, sd2)
        self.verify(sd2, sd1)
        sd3 = sd1.merge(sd2)
        for var in sd3.variants():
            assert var.site.alleles == ("0", "1", "2")

    def test_missing_data(self):
        u = tskit.MISSING_DATA
        sites_by_samples = np.array(
            [
                [u, u, u, 1, 1, 0, 1, 1, 1],
                [u, u, u, 1, 1, 0, 1, 1, 0],
                [u, u, u, 1, 0, 1, 1, 0, 1],
                [u, 0, 0, 1, 1, 1, 1, u, u],
                [u, 0, 1, 1, 1, 0, 1, u, u],
                [u, 1, 1, 0, 0, 0, 0, u, u],
            ],
            dtype=np.int8,
        )
        with tsinfer.SampleData() as sd1:
            for col in range(sites_by_samples.shape[1]):
                sd1.add_site(col, sites_by_samples[:, col])
        self.verify(sd1, sd1)
        with tsinfer.SampleData(sd1.sequence_length) as sd2:
            for col in range(4):
                sd2.add_site(col, sites_by_samples[:, col])
        self.verify(sd1, sd2)
        self.verify(sd2, sd1)

    @pytest.mark.skipif(
        IS_WINDOWS, reason="windows simultaneous file permissions issue"
    )
    def test_file_kwargs(self):
        # Make sure we pass kwards on to the SampleData constructor as
        # required.
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample-data")
            sd2 = sd1.merge(sd1, path=path)
            assert os.path.exists(path)
            sd3 = formats.SampleData.load(path)
            assert sd2.data_equal(sd3)


class TestMinSiteTimes:
    """
    Test cases for sample data's min_site_times function
    """

    def test_no_historical(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts, use_sites_time=True)
        # No arguments and individuals_only=True should give array of zeros
        bounds_individuals_only = sd1.min_site_times(individuals_only=True)
        assert np.array_equal(bounds_individuals_only, np.zeros(sd1.num_sites))
        bounds = sd1.min_site_times()
        assert np.array_equal(bounds, sd1.sites_time[:])

    def test_simple_case(self):
        individual_times = [0, 0, 0.5, 1]
        ts = tsutil.get_example_historical_sampled_ts(individual_times, ploidy=1)
        sd1 = formats.SampleData.from_tree_sequence(
            ts, use_sites_time=True, use_individuals_time=True
        )
        time_bound_individuals_only = sd1.min_site_times(individuals_only=True)
        # Because this is a haploid tree sequence we can use the
        # individual and sample IDs interchangably.
        assert np.all(
            np.in1d(
                time_bound_individuals_only, np.concatenate([[0], individual_times])
            )
        )
        G1 = ts.genotype_matrix()
        older_derived = G1[:, 3] == 1
        assert np.all(time_bound_individuals_only[older_derived] == 1)
        only_younger_derived = np.logical_and(G1[:, 2] == 1, G1[:, 3] == 0)
        assert np.all(time_bound_individuals_only[only_younger_derived] == 0.5)
        no_historical_derived = np.all(G1[:2:4] != 1)
        assert np.all(time_bound_individuals_only[no_historical_derived] == 0)
        time_bound = sd1.min_site_times()
        assert np.array_equal(time_bound, sd1.sites_time[:])

    def test_errors(self):
        ts = tsutil.get_example_ts(10)
        sd1 = formats.SampleData.from_tree_sequence(ts)
        individuals_time = sd1.individuals_time[:]
        neg_times_sd1 = sd1.copy()
        neg_times_sd1.individuals_time[:] = individuals_time - 1
        neg_times_sd1.finalise()
        with pytest.raises(ValueError):
            neg_times_sd1.min_site_times()


class TestAncestorData(DataContainerMixin):
    """
    Test cases for the ancestor data file format.
    """

    class_to_test = formats.AncestorData

    def get_example_data(
        self, sample_size, sequence_length, num_ancestors, use_sites_time=True
    ):
        ts = msprime.simulate(
            sample_size,
            mutation_rate=10,
            recombination_rate=1,
            length=sequence_length,
            random_seed=100,
        )
        sample_data = formats.SampleData.from_tree_sequence(
            ts, use_sites_time=use_sites_time
        )

        num_sites = sample_data.num_sites
        ancestors = []
        for j in reversed(range(num_ancestors)):
            haplotype = np.full(num_sites, tskit.MISSING_DATA, dtype=np.int8)
            start = j
            end = max(num_sites - j, start + 1)
            assert start < end
            haplotype[start:end] = 0
            if start + j < end:
                haplotype[start + j : end] = 1
            assert np.all(haplotype[:start] == tskit.MISSING_DATA)
            assert np.all(haplotype[end:] == tskit.MISSING_DATA)
            focal_sites = np.array([start + k for k in range(j)], dtype=np.int32)
            focal_sites = focal_sites[focal_sites < end]
            haplotype[focal_sites] = 1
            ancestors.append((start, end, 2 * j + 1, focal_sites, haplotype))
        return sample_data, ancestors

    def assert_ancestor_full_span(self, ancestor_data, indexes):
        assert np.all(ancestor_data.ancestors_start[:][indexes] == 0)
        assert np.all(
            ancestor_data.ancestors_end[:][indexes] == ancestor_data.num_sites
        )

    def verify_data_round_trip(self, sample_data, ancestor_data, ancestors):
        for i, (start, end, t, focal_sites, haplotype) in enumerate(ancestors):
            assert i == ancestor_data.add_ancestor(
                start, end, t, focal_sites, haplotype[start:end]
            )
        ancestor_data.record_provenance("verify_data_round_trip")
        ancestor_data.finalise()

        assert len(ancestor_data.uuid) > 0
        assert ancestor_data.sample_data_uuid == sample_data.uuid
        assert ancestor_data.sequence_length == sample_data.sequence_length
        assert ancestor_data.format_name == formats.AncestorData.FORMAT_NAME
        assert ancestor_data.format_version == formats.AncestorData.FORMAT_VERSION
        assert ancestor_data.num_ancestors == len(ancestors)

        ancestors_list = [anc.haplotype for anc in ancestor_data.ancestors()]
        stored_start = ancestor_data.ancestors_start[:]
        stored_end = ancestor_data.ancestors_end[:]
        stored_time = ancestor_data.ancestors_time[:]
        stored_ancestors = ancestor_data.ancestors_haplotype[:]
        stored_focal_sites = ancestor_data.ancestors_focal_sites[:]
        stored_length = ancestor_data.ancestors_length[:]
        for j, (start, end, t, focal_sites, haplotype) in enumerate(ancestors):
            assert stored_start[j] == start
            assert stored_end[j] == end
            assert stored_time[j] == t
            assert np.array_equal(stored_focal_sites[j], focal_sites)
            assert np.array_equal(stored_ancestors[j], haplotype[start:end])
            assert np.array_equal(ancestors_list[j], haplotype[start:end])
        pos = list(ancestor_data.sites_position[:]) + [ancestor_data.sequence_length]
        for j, anc in enumerate(ancestor_data.ancestors()):
            assert stored_start[j] == anc.start
            assert stored_end[j] == anc.end
            assert stored_time[j] == anc.time
            assert np.array_equal(stored_focal_sites[j], anc.focal_sites)
            assert np.array_equal(stored_ancestors[j], anc.haplotype)
            length = pos[anc.end] - pos[anc.start]
            assert stored_length[j] == length

    def test_defaults_no_path(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        for _, array in ancestor_data.arrays():
            assert array.compressor == formats.DEFAULT_COMPRESSOR

    @pytest.mark.skipif(
        IS_WINDOWS, reason="windows simultaneous file permissions issue"
    )
    def test_defaults_with_path(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            ancestor_data = tsinfer.AncestorData(sample_data, path=filename)
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            compressor = formats.DEFAULT_COMPRESSOR
            for _, array in ancestor_data.arrays():
                assert array.compressor == compressor
            with tsinfer.load(filename) as other:
                assert other == ancestor_data

    def test_bad_max_file_size(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            for bad_size in ["a", "", -1]:
                with pytest.raises(ValueError):
                    formats.AncestorData(
                        sample_data, path=filename, max_file_size=bad_size
                    )
            for bad_size in [[1, 3], np.array([1, 2])]:
                with pytest.raises(TypeError):
                    formats.AncestorData(
                        sample_data, path=filename, max_file_size=bad_size
                    )

    def test_provenance(self):
        sample_data, ancestors = self.get_example_data(10, 10, 40)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        assert ancestor_data.num_provenances == sample_data.num_provenances + 1

        timestamp = ancestor_data.provenances_timestamp[-1]
        iso = datetime.datetime.now().isoformat()
        assert timestamp.split("T")[0] == iso.split("T")[0]
        record = ancestor_data.provenances_record[-1]
        assert record["software"]["name"] == "tsinfer"
        a = list(ancestor_data.provenances())
        assert a[-1][0] == timestamp
        assert a[-1][1] == record
        for j, (timestamp, record) in enumerate(sample_data.provenances()):
            assert timestamp == a[j][0]
            assert record == a[j][1]

    def test_chunk_size(self):
        N = 20
        for chunk_size in [1, 2, 3, N - 1, N, N + 1]:
            sample_data, ancestors = self.get_example_data(6, 1, num_ancestors=N)
            ancestor_data = tsinfer.AncestorData(sample_data, chunk_size=chunk_size)
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            assert ancestor_data.ancestors_haplotype.chunks == (chunk_size,)
            assert ancestor_data.ancestors_focal_sites.chunks == (chunk_size,)
            assert ancestor_data.ancestors_start.chunks == (chunk_size,)
            assert ancestor_data.ancestors_end.chunks == (chunk_size,)
            assert ancestor_data.ancestors_time.chunks == (chunk_size,)

    def test_filename(self):
        sample_data, ancestors = self.get_example_data(10, 2, 40)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "ancestors.tmp")
            ancestor_data = tsinfer.AncestorData(sample_data, path=filename)
            assert os.path.exists(filename)
            assert not os.path.isdir(filename)
            self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
            # Make a copy so that we can close the original and reopen it
            # without hitting simultaneous file access problems on Windows
            ancestor_copy = ancestor_data.copy(path=None)
            ancestor_data.close()
            other_ancestor_data = formats.AncestorData.load(filename)
            assert other_ancestor_data is not ancestor_copy
            # Can't use eq here because UUIDs will not be equal.
            assert other_ancestor_data.data_equal(ancestor_copy)
            other_ancestor_data.close()

    def test_chunk_size_file_equal(self):
        N = 60
        sample_data, ancestors = self.get_example_data(22, 16, N)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            files = []
            for chunk_size in [5, 7]:
                filename = os.path.join(tempdir, f"samples_{chunk_size}.tmp")
                files.append(filename)
                with tsinfer.AncestorData(
                    sample_data, path=filename, chunk_size=chunk_size
                ) as ancestor_data:
                    self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
                    assert ancestor_data.ancestors_haplotype.chunks == (chunk_size,)
            # Now reload the files and check they are equal
            with formats.AncestorData.load(files[0]) as file0:
                with formats.AncestorData.load(files[1]) as file1:
                    assert file0.data_equal(file1)

    def test_add_ancestor_errors(self):
        sample_data, ancestors = self.get_example_data(22, 16, 30)
        ancestor_data = tsinfer.AncestorData(sample_data)
        num_sites = ancestor_data.num_sites
        haplotype = np.zeros(num_sites, dtype=np.int8)
        ancestor_data.add_ancestor(
            start=0, end=num_sites, time=1, focal_sites=[], haplotype=haplotype
        )
        for bad_start in [-1, -100, num_sites, num_sites + 1]:
            with pytest.raises(ValueError):
                ancestor_data.add_ancestor(
                    start=bad_start,
                    end=num_sites,
                    time=0,
                    focal_sites=[],
                    haplotype=haplotype,
                )
        for bad_end in [-1, 0, num_sites + 1, 10 * num_sites]:
            with pytest.raises(ValueError):
                ancestor_data.add_ancestor(
                    start=0, end=bad_end, time=1, focal_sites=[], haplotype=haplotype
                )
        for bad_time in [-1, 0]:
            with pytest.raises(ValueError):
                ancestor_data.add_ancestor(
                    start=0,
                    end=num_sites,
                    time=bad_time,
                    focal_sites=[],
                    haplotype=haplotype,
                )
        with pytest.raises(ValueError):
            ancestor_data.add_ancestor(
                start=0,
                end=num_sites,
                time=1,
                focal_sites=[],
                haplotype=np.zeros(num_sites + 1, dtype=np.int8),
            )
        # Haplotypes must be < num_alleles
        with pytest.raises(ValueError):
            ancestor_data.add_ancestor(
                start=0,
                end=num_sites,
                time=1,
                focal_sites=[],
                haplotype=np.zeros(num_sites, dtype=np.int8) + 2,
            )
        # focal sites must be within start:end
        with pytest.raises(ValueError):
            ancestor_data.add_ancestor(
                start=1,
                end=num_sites,
                time=1,
                focal_sites=[0],
                haplotype=np.ones(num_sites - 1, dtype=np.int8),
            )
        with pytest.raises(ValueError):
            ancestor_data.add_ancestor(
                start=0,
                end=num_sites - 2,
                time=1,
                focal_sites=[num_sites - 1],
                haplotype=np.ones(num_sites, dtype=np.int8),
            )
        # Older ancestors must be added first
        with pytest.raises(ValueError):
            ancestor_data.add_ancestor(
                start=0, end=num_sites, time=2, focal_sites=[], haplotype=haplotype
            )

    def test_iterator(self):
        sample_data, ancestors = self.get_example_data(6, 10, 10)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        assert ancestor_data.num_ancestors > 1
        assert ancestor_data.num_ancestors == len(ancestors)
        for ancestor, new_ancestor in zip(ancestors, ancestor_data.ancestors()):
            assert ancestor[0] == new_ancestor.start
            assert ancestor[1] == new_ancestor.end
            assert ancestor[2] == new_ancestor.time
            assert np.all(ancestor[3] == new_ancestor.focal_sites)

    def test_equals(self):
        sample_data, ancestors = self.get_example_data(6, 1, 2)
        with tsinfer.AncestorData(sample_data) as ancestor_data:
            num_sites = ancestor_data.num_sites
            haplotype = np.ones(num_sites, dtype=np.int8)
            ancestor_data.add_ancestor(
                start=0, end=num_sites, time=1, focal_sites=[], haplotype=haplotype
            )
            ancestor_data.add_ancestor(
                start=0, end=num_sites, time=1, focal_sites=[], haplotype=haplotype
            )
        assert ancestor_data.ancestor(0) == ancestor_data.ancestor(0)
        assert ancestor_data.ancestor(0) != ancestor_data.ancestor(1)  # IDs differ

    def test_accessor(self):
        sample_data, ancestors = self.get_example_data(6, 10, 10)
        ancestor_data = tsinfer.AncestorData(sample_data)
        self.verify_data_round_trip(sample_data, ancestor_data, ancestors)
        for i, new_ancestor in enumerate(ancestor_data.ancestors()):
            assert new_ancestor == ancestor_data.ancestor(i)

    @pytest.mark.skipif(
        IS_WINDOWS, reason="windows simultaneous file permissions issue"
    )
    def test_zero_sequence_length(self):
        # Mangle a sample data file to force a zero sequence length.
        ts = msprime.simulate(10, mutation_rate=2, random_seed=5)
        with tempfile.TemporaryDirectory(prefix="tsinf_format_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            with tsinfer.SampleData(path=filename) as sample_data:
                for var in ts.variants():
                    sample_data.add_site(var.site.position, var.genotypes)
            store = zarr.LMDBStore(filename, subdir=False)
            data = zarr.open(store=store, mode="w+")
            data.attrs["sequence_length"] = 0
            store.close()
            sample_data = tsinfer.load(filename)
            assert sample_data.sequence_length == 0
            with pytest.raises(ValueError):
                tsinfer.generate_ancestors(sample_data)

    def test_bad_insert_proxy_samples(self):
        sample_data, ancestor_haps = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.AncestorData(sample_data)
        with pytest.raises(ValueError, match="not finalised"):
            ancestors.insert_proxy_samples(sample_data)
        self.verify_data_round_trip(sample_data, ancestors, ancestor_haps)
        sample_data = sample_data.copy()
        with pytest.raises(ValueError, match="not finalised"):
            ancestors.insert_proxy_samples(sample_data, require_same_sample_data=False)
        # Check it does work when finalised
        sample_data.finalise()
        ancestors.insert_proxy_samples(sample_data, require_same_sample_data=False)

    def test_insert_proxy_bad_sample_data(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        # by default, sample_data must be the same
        sd_copy, _ = self.get_example_data(10, 10, num_ancestors=40)
        with pytest.raises(ValueError):
            ancestors.insert_proxy_samples(sd_copy)
        # But works if we don't require same data
        ancestors.insert_proxy_samples(sd_copy, require_same_sample_data=False)
        # Unless seq lengths differ
        sd_copy, _ = self.get_example_data(10, sequence_length=11, num_ancestors=40)
        with pytest.raises(ValueError, match="sequence length"):
            ancestors.insert_proxy_samples(sd_copy, require_same_sample_data=False)

    def test_insert_proxy_site_changes(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        nonsingletons = np.where(np.sum(sample_data.sites_genotypes, axis=1) > 1)[0]
        sd_copy = sample_data.subset(sites=nonsingletons)
        # Should be able to use a sd subset if only non-inference sites are missing
        ancestors.insert_proxy_samples(sd_copy, require_same_sample_data=False)
        # But if we remove a *full inference* site, we should always fail
        sd_copy = sample_data.subset(sites=nonsingletons[1:])
        with pytest.raises(ValueError, match="positions.*missing"):
            ancestors.insert_proxy_samples(sd_copy, require_same_sample_data=False)

    def test_insert_proxy_bad_samples(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        for bad_id in [-1, "a", 10, np.nan, np.inf, -np.inf]:
            with pytest.raises(IndexError):
                ancestors.insert_proxy_samples(sample_data, sample_ids=[bad_id])

    def test_insert_proxy_no_samples(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_extra = ancestors.insert_proxy_samples(sample_data, sample_ids=[])
        assert ancestors != ancestors_extra  # UUIDs should differ ...
        assert ancestors.data_equal(ancestors_extra)  # but data be identical

    def test_insert_proxy_1_sample(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        used_sites = np.isin(sample_data.sites_position[:], ancestors.sites_position[:])
        for i in (0, 6, 9):
            ancestors_extra = ancestors.insert_proxy_samples(
                sample_data, sample_ids=[i]
            )
            assert ancestors.num_ancestors + 1 == ancestors_extra.num_ancestors
            inserted = -1  # inserted one should always be the last
            self.assert_ancestor_full_span(ancestors_extra, [inserted])
            assert np.array_equal(
                ancestors_extra.ancestors_haplotype[inserted],
                sample_data.sites_genotypes[:, i][used_sites],
            )

    def test_insert_proxy_sample_ids(self):
        ids = [1, 4]
        sd, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sd).insert_proxy_samples(
            sd, sample_ids=ids
        )
        inference_sites = np.isin(sd.sites_position[:], ancestors.sites_position[:])
        anc_sample_ids = ancestors.ancestors_sample_id[:]
        assert np.sum(anc_sample_ids != tskit.NULL) == len(ids)
        for sample_id in ids:
            assert np.sum(anc_sample_ids == sample_id) == 1
            anc_sample = np.where(anc_sample_ids == sample_id)[0][0]
            assert ancestors.ancestors_start[anc_sample] == 0
            assert ancestors.ancestors_end[anc_sample] == ancestors.num_sites
            assert len(ancestors.ancestors_focal_sites[anc_sample]) == 0

            haplotype = next(sd.haplotypes([sample_id], sites=inference_sites))[1]
            assert np.all(ancestors.ancestors_haplotype[anc_sample] == haplotype)

    def test_insert_proxy_different_sample_data(self):
        ids = [1, 4]
        sd, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sd)
        sd_copy, _ = self.get_example_data(10, 10, num_ancestors=40)
        ancestors_extra = ancestors.insert_proxy_samples(
            sd_copy, sample_ids=ids, require_same_sample_data=False
        )
        assert np.all(ancestors_extra.ancestors_sample_id[:] == tskit.NULL)

    def test_insert_proxy_sample_provenance(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_extra = ancestors.insert_proxy_samples(sample_data, sample_ids=[6])
        for anc_prov, sd_prov in itertools.zip_longest(
            ancestors_extra.provenances(), ancestors.provenances()
        ):
            if sd_prov is None:
                params = anc_prov[1]["parameters"]
                assert params["command"] == "insert_proxy_samples"
            else:
                assert anc_prov == sd_prov

    def test_insert_proxy_time_historical_samples(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        site_times = sample_data.sites_time[:]  # From a simulation => sites have times
        assert not np.all(tskit.is_unknown_time(site_times))
        min_time = np.min(site_times[site_times > 0])
        sample_data = sample_data.copy()
        time = sample_data.individuals_time[:]
        assert len(time) == 10
        assert np.all(time == 0)
        historical_sample_time = min_time / 2  # Smaller than the smallest freq value
        time[9] = historical_sample_time
        sample_data.individuals_time[:] = time
        sample_data.finalise()
        ancestors = tsinfer.generate_ancestors(sample_data)
        assert np.min(ancestors.ancestors_time[:]) > historical_sample_time
        used_sites = np.isin(sample_data.sites_position[:], ancestors.sites_position[:])
        epsilon = (ancestors.ancestors_time[-1] - historical_sample_time) / 100
        G = sample_data.sites_genotypes

        # By default, insert_proxy_samples should insert the single historical proxy
        ancestors_extra = ancestors.insert_proxy_samples(sample_data)
        assert ancestors.num_ancestors + 1 == ancestors_extra.num_ancestors
        self.assert_ancestor_full_span(ancestors_extra, [-1])
        assert np.array_equal(
            ancestors_extra.ancestors_haplotype[-1], G[:, 9][used_sites]
        )
        assert np.array_equal(
            ancestors_extra.ancestors_time[-1], historical_sample_time + epsilon
        )
        assert np.sum(ancestors_extra.ancestors_sample_id[:] != tskit.NULL) == 1
        assert ancestors_extra.ancestors_sample_id[-1] == 9

        # Test 2 proxies, one historical, specified in different ways / orders
        s_ids = np.array([9, 0])
        assert not np.array_equal(G[:, 9][used_sites], G[:, 0][used_sites])
        for i in (s_ids, s_ids[::-1], s_ids[[1, 1, 0]]):  # All equivalent
            ancestors_extra = ancestors.insert_proxy_samples(sample_data, sample_ids=i)
            assert ancestors.num_ancestors + len(s_ids) == ancestors_extra.num_ancestors
            inserted = [-1, -2]
            self.assert_ancestor_full_span(ancestors_extra, inserted)
            # Older sample
            assert np.array_equal(
                ancestors_extra.ancestors_haplotype[-2], G[:, 9][used_sites]
            )
            assert np.array_equal(
                ancestors_extra.ancestors_time[-2], historical_sample_time + epsilon
            )
            # Younger sample
            assert np.array_equal(
                ancestors_extra.ancestors_haplotype[-1], G[:, 0][used_sites]
            )
            assert np.array_equal(ancestors_extra.ancestors_time[-1], epsilon)
            assert np.sum(ancestors_extra.ancestors_sample_id[:] != tskit.NULL) == 2
            assert ancestors_extra.ancestors_sample_id[-2] == 9
            assert ancestors_extra.ancestors_sample_id[-1] == 0

    def test_insert_proxy_sample_epsilon(self):
        sample_data, _ = self.get_example_data(10, 10, 40)
        ancestors = tsinfer.generate_ancestors(sample_data)
        min_time = ancestors.ancestors_time[-1]
        s_ids = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            ancestors.insert_proxy_samples(
                sample_data, sample_ids=s_ids, epsilon=[min_time * 0.1, min_time * 0.2]
            )

        for e in (min_time * 0.05, [min_time * 0.1, min_time * 0.2, min_time * 0.3]):
            ancestors_extra = ancestors.insert_proxy_samples(
                sample_data, sample_ids=s_ids, epsilon=e
            )
            inserted = [-1, -2, -3]
            assert np.allclose(ancestors_extra.ancestors_time[:][inserted], e)

    def test_insert_proxy_sample_oldest(self):
        # Test all historical samples are older than the oldest site
        sample_data, _ = self.get_example_data(10, 10, 40)
        max_time = np.max(sample_data.sites_time[:])
        sample_data = sample_data.copy()
        time = sample_data.individuals_time[:]
        historical_sample_time = max_time * 2  # Smaller than the smallest freq value
        s_ids = np.array([4, 5])
        time[s_ids] = historical_sample_time
        sample_data.individuals_time[:] = time
        sample_data.finalise()
        ancestors = tsinfer.generate_ancestors(sample_data)
        with pytest.raises(ValueError, match="allow_mutation"):
            ancestors.insert_proxy_samples(sample_data)
        ancestors_extra = ancestors.insert_proxy_samples(
            sample_data, allow_mutation=True
        )
        assert ancestors.num_ancestors + 2 == ancestors_extra.num_ancestors
        inserted = [0, 1]  # Should be the oldest ancestors
        e = np.diff(ancestors.ancestors_time[:][::-1])
        e = np.min(e[e > 0]) / 100
        assert np.allclose(
            ancestors_extra.ancestors_time[:][inserted], historical_sample_time + e
        )

    def test_bad_input_truncate_ancestors(self):
        sample_data, _ = self.get_example_data(10, 10, 40, use_sites_time=False)
        ancestors = tsinfer.generate_ancestors(sample_data)
        with pytest.raises(ValueError, match="Time bounds cannot be negative"):
            ancestors.truncate_ancestors(-1, -2)
        with pytest.raises(ValueError, match="cannot be zero or negative"):
            ancestors.truncate_ancestors(0.3, 0.5, -2)
            ancestors.truncate_ancestors(0.3, 0.5, 0)
        with pytest.raises(ValueError, match="Upper bound must be >= lower bound"):
            ancestors.truncate_ancestors(0.3, 0.1)
        with pytest.raises(ValueError, match="greater than older ancestor"):
            ancestors.truncate_ancestors(10, 20)
        with pytest.raises(ValueError, match="No ancestors in time bound"):
            ancestors.truncate_ancestors(0.555555, 0.555555)

    def test_length_multiplier(self):
        sample_data, _ = self.get_example_data(10, 10, 40, use_sites_time=False)
        ancestors = tsinfer.generate_ancestors(sample_data)
        trunc_anc = ancestors.truncate_ancestors(0.9, 1, 1)
        assert np.array_equal(ancestors.ancestors_length, trunc_anc.ancestors_length)

    def test_ancestors_truncated_length(self):
        sample_data, _ = self.get_example_data(10, 10, 40, use_sites_time=False)
        ancestors = tsinfer.generate_ancestors(sample_data)
        lower_limit = 0.4
        upper_limit = 0.6
        trunc_anc = ancestors.truncate_ancestors(lower_limit, upper_limit, 1)
        original_lengths = ancestors.ancestors_length[:]
        trunc_lengths = trunc_anc.ancestors_length[:]
        # Check that ancestors older than upper_limit have been cut down
        trunc_targets = np.logical_and(
            trunc_anc.ancestors_time[:] > upper_limit, trunc_anc.ancestors_time[:] < 1
        )
        assert np.all(trunc_lengths[trunc_targets] <= original_lengths[trunc_targets])
        assert np.array_equal(trunc_lengths[-2:], original_lengths[-2:])
        time = ancestors.ancestors_time[:]
        # Test younger haplotypes have been left alone
        assert np.array_equal(
            trunc_lengths[time < upper_limit], original_lengths[time < upper_limit]
        )
        for orig_anc, trunc_anc in zip(  # noqa: B020
            ancestors.ancestors(), trunc_anc.ancestors()
        ):
            assert orig_anc.time == trunc_anc.time
            assert np.array_equal(orig_anc.focal_sites, trunc_anc.focal_sites)
            if orig_anc.time >= upper_limit:
                assert orig_anc.end >= trunc_anc.end
                assert np.array_equal(
                    orig_anc.haplotype[
                        trunc_anc.start
                        - orig_anc.start : trunc_anc.end
                        - orig_anc.start
                    ],
                    trunc_anc.haplotype,
                )
            else:
                assert orig_anc.start == trunc_anc.start
                assert orig_anc.end == trunc_anc.end
                assert np.array_equal(orig_anc.haplotype, trunc_anc.haplotype)

    def test_truncate_extreme_interval(self):
        # Test all haplotypes are untouched when specifying interval of min-max time
        sample_data, _ = self.get_example_data(10, 10, 40, use_sites_time=True)
        ancestors = tsinfer.generate_ancestors(sample_data)
        time = ancestors.ancestors_time[:]
        trunc_anc = ancestors.truncate_ancestors(np.min(time), np.max(time), 1)
        for orig_anc, trunc_anc in zip(  # noqa: B020
            ancestors.ancestors(), trunc_anc.ancestors()
        ):
            assert orig_anc.start == trunc_anc.start
            assert orig_anc.end == trunc_anc.end
            assert orig_anc.time == trunc_anc.time
            assert np.array_equal(orig_anc.focal_sites, trunc_anc.focal_sites)
            assert np.array_equal(orig_anc.haplotype, trunc_anc.haplotype)
        # Test all haplotypes are untouched when specifying interval of 0-1
        sample_data, _ = self.get_example_data(10, 10, 40, use_sites_time=False)
        ancestors = tsinfer.generate_ancestors(sample_data)
        time = ancestors.ancestors_time[:]
        trunc_anc = ancestors.truncate_ancestors(0, 1, 1)
        for orig_anc, trunc_anc in zip(  # noqa: B020
            ancestors.ancestors(), trunc_anc.ancestors()
        ):
            assert orig_anc.start == trunc_anc.start
            assert orig_anc.end == trunc_anc.end
            assert orig_anc.time == trunc_anc.time
            assert np.array_equal(orig_anc.focal_sites, trunc_anc.focal_sites)
            assert np.array_equal(orig_anc.haplotype, trunc_anc.haplotype)

    def test_one_haplotype_truncated(self):
        # Test truncating one haplotype (the oldest with a focal site) works as expected
        sample_data, _ = self.get_example_data(10, 10, 40, use_sites_time=True)
        ancestors = tsinfer.generate_ancestors(sample_data)
        sites_time = sample_data.sites_time[:]
        oldest_site = np.max(sites_time)
        midpoint = np.median(sites_time)
        trunc_anc = ancestors.truncate_ancestors(midpoint, oldest_site, 1)
        for orig_anc, trunc_anc in zip(  # noqa: B020
            ancestors.ancestors(), trunc_anc.ancestors()
        ):
            assert orig_anc.time == trunc_anc.time
            assert np.array_equal(orig_anc.focal_sites, trunc_anc.focal_sites)

    def test_multiple_focal_sites(self):
        # Test that ancestor with 2 focal sites spaced far apart isn't cut down
        sample_data = tsinfer.SampleData()
        sample_data.add_site(0, [1, 1, 1, 0])
        sample_data.add_site(5, [1, 0, 0, 1])
        sample_data.add_site(10, [1, 1, 1, 0])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData(sample_data)
        ancestor_data.add_ancestor(0, 3, 0.6666, [0, 2], [1, 1, 0])
        ancestor_data.add_ancestor(1, 2, 0.333, [1], [1])
        ancestor_data.finalise()
        trunc_anc = ancestor_data.truncate_ancestors(0.3, 0.4, 1)
        assert np.array_equal(
            trunc_anc.ancestors_haplotype[-1], ancestor_data.ancestors_haplotype[-1]
        )


class BufferedItemWriterMixin:
    """
    Tests to ensure that the buffered item writer works as expected.
    """

    def filter_warnings_verify_round_trip(self, source):
        # Zarr currently emits an error when dealing with object arrays.
        # https://github.com/zarr-developers/zarr/issues/257
        # As a workaround, we filter this warning.
        # This should be removed when the bug has been fixed upstream.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.verify_round_trip(source)

    def verify_round_trip(self, source):
        """
        Verify that we can round trip the specified mapping of arrays
        using the buffered item writer.
        """
        # Create a set of empty arrays like the originals.
        dest = {}
        num_rows = -1
        for key, array in source.items():
            dest[key] = zarr.empty_like(array)
            if num_rows == -1:
                num_rows = array.shape[0]
            assert num_rows == array.shape[0]
        assert num_rows != -1
        writer = formats.BufferedItemWriter(dest, num_threads=self.num_threads)
        for j in range(num_rows):
            row = {key: array[j] for key, array in source.items()}
            writer.add(**row)
        writer.flush()

        for key, source_array in source.items():
            dest_array = dest[key]
            if source_array.dtype.str == "|O":
                # Object arrays have to be treated differently.
                assert source_array.shape == dest_array.shape
                for a, b in zip(source_array, dest_array):
                    if isinstance(a, np.ndarray):
                        assert np.array_equal(a, b)
                    else:
                        assert a == b
            else:
                assert np.array_equal(source_array[:], dest_array[:])
            assert source_array.chunks == dest_array.chunks
        return dest

    def test_one_array(self):
        self.verify_round_trip({"a": zarr.ones(10)})

    def test_two_arrays(self):
        self.verify_round_trip({"a": zarr.ones(10), "b": zarr.zeros(10)})

    def verify_dtypes(self, chunk_size=None):
        n = 100
        if chunk_size is None:
            chunk_size = 100
        dtypes = [np.int8, np.uint8, np.int32, np.uint32, np.float64, np.float32]
        source = {
            str(dtype): zarr.array(np.arange(n, dtype=dtype), chunks=(chunk_size,))
            for dtype in dtypes
        }
        dest = self.verify_round_trip(source)
        for dtype in dtypes:
            assert dest[str(dtype)].dtype == dtype

    def test_mixed_dtypes(self):
        self.verify_dtypes()

    def test_mixed_dtypes_chunk_size_1(self):
        self.verify_dtypes(1)

    def test_mixed_dtypes_chunk_size_2(self):
        self.verify_dtypes(2)

    def test_mixed_dtypes_chunk_size_3(self):
        self.verify_dtypes(3)

    def test_mixed_dtypes_chunk_size_10000(self):
        self.verify_dtypes(10000)

    def test_2d_array(self):
        a = zarr.array(np.arange(100).reshape((10, 10)))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_1_1(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 1))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_1_2(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 2))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_2_1(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 2))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_1_100(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(1, 100))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_100_1(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(100, 1))
        self.verify_round_trip({"a": a})

    def test_2d_array_chunk_size_10_10(self):
        a = zarr.array(np.arange(100).reshape((10, 10)), chunks=(5, 10))
        self.verify_round_trip({"a": a})

    def test_3d_array(self):
        a = zarr.array(np.arange(27).reshape((3, 3, 3)))
        self.verify_round_trip({"a": a})

    def test_3d_array_chunks_size_1_1_1(self):
        a = zarr.array(np.arange(27).reshape((3, 3, 3)), chunks=(1, 1, 1))
        self.verify_round_trip({"a": a})

    def test_ragged_array_int32(self):
        n = 10
        z = zarr.empty(n, dtype="array:i4")
        for j in range(n):
            z[j] = np.arange(j)
        self.filter_warnings_verify_round_trip({"z": z})

    def test_square_object_array_int32(self):
        n = 10
        z = zarr.empty(n, dtype="array:i4")
        for j in range(n):
            z[j] = np.arange(n)
        self.filter_warnings_verify_round_trip({"z": z})

    def test_json_object_array(self):
        for chunks in [1, 2, 5, 10, 100]:
            n = 10
            z = zarr.empty(
                n, dtype=object, object_codec=numcodecs.JSON(), chunks=(chunks,)
            )
            for j in range(n):
                z[j] = {str(k): k for k in range(j)}
            self.filter_warnings_verify_round_trip({"z": z})

    def test_empty_string_list(self):
        z = zarr.empty(1, dtype=object, object_codec=numcodecs.JSON(), chunks=(2,))
        z[0] = ["", ""]
        self.filter_warnings_verify_round_trip({"z": z})

    def test_mixed_chunk_sizes(self):
        source = {"a": zarr.zeros(10, chunks=(1,)), "b": zarr.zeros(10, chunks=(2,))}
        with pytest.raises(ValueError):
            formats.BufferedItemWriter(source)


class TestBufferedItemWriterSynchronous(BufferedItemWriterMixin):
    num_threads = 0


class TestBufferedItemWriterThreads1(BufferedItemWriterMixin):
    num_threads = 1


class TestBufferedItemWriterThreads2(BufferedItemWriterMixin):
    num_threads = 2


class TestBufferedItemWriterThreads20(BufferedItemWriterMixin):
    num_threads = 20
