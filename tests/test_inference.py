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
Tests for the inference code.
"""
import io
import itertools
import json
import os.path
import random
import string
import tempfile
import unittest
import unittest.mock as mock

import msprime
import numpy as np
import pytest
import tskit
import tsutil

import _tsinfer
import tsinfer
import tsinfer.eval_util as eval_util


def get_random_data_example(num_samples, num_sites, seed=42, num_states=2):
    np.random.seed(seed)
    G = np.random.randint(num_states, size=(num_sites, num_samples)).astype(np.int8)
    return G, np.arange(num_sites)


class TestUnfinalisedErrors:
    def make_ancestor_data_unfinalised(self, path=None):
        with tsinfer.SampleData(path=path, sequence_length=2) as sample_data:
            sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
            with pytest.raises(ValueError):
                tsinfer.AncestorData(sample_data)
        if path is not None:
            sample_data.close()

    def match_ancestors_ancestors_unfinalised(self, path=None):
        with tsinfer.SampleData(sequence_length=2) as sample_data:
            sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        with tsinfer.AncestorData(sample_data, path=path) as ancestor_data:
            ancestor_data.add_ancestor(
                start=0,
                end=1,
                time=2.0,
                focal_sites=[0],
                haplotype=np.array([1], dtype=np.int8),
            )
            # match_ancestors fails when ancestors unfinalised
            with pytest.raises(ValueError):
                tsinfer.match_ancestors(sample_data, ancestor_data)
        if path is not None:
            ancestor_data.close()

    def test_make_ancestor_data(self):
        self.make_ancestor_data_unfinalised()

    def test_make_ancestor_data_file(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_inference_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            self.make_ancestor_data_unfinalised(filename)

    def test_match_ancestors_ancestors(self):
        self.match_ancestors_ancestors_unfinalised()

    def test_match_ancestors_ancestors_file(self):
        with tempfile.TemporaryDirectory(prefix="tsinf_inference_test") as tempdir:
            filename = os.path.join(tempdir, "samples.tmp")
            self.match_ancestors_ancestors_unfinalised(filename)

    def test_generate_ancestors(self):
        with tsinfer.SampleData(sequence_length=2) as sample_data:
            sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
            with pytest.raises(ValueError):
                tsinfer.generate_ancestors(sample_data)
        tsinfer.generate_ancestors(sample_data)

    def test_match_ancestors_samples(self):
        with tsinfer.SampleData(sequence_length=2) as sample_data:
            sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        # match_ancestors fails when samples unfinalised
        unfinalised = tsinfer.SampleData(sequence_length=2)
        unfinalised.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        with pytest.raises(ValueError):
            tsinfer.match_ancestors(unfinalised, ancestor_data)

    def test_match_samples_unfinalised(self):
        with tsinfer.SampleData(sequence_length=2) as sample_data:
            sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        anc_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        sample_data = tsinfer.SampleData(sequence_length=2)
        sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, anc_ts)
        sample_data.finalise()
        tsinfer.match_samples(sample_data, anc_ts)

    def test_augment_ancestors_unfinalised(self):
        with tsinfer.SampleData(sequence_length=2) as sample_data:
            sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        sample_data = tsinfer.SampleData(sequence_length=2)
        sample_data.add_site(1, genotypes=[0, 1, 1, 0], alleles=["G", "C"])
        with pytest.raises(ValueError):
            tsinfer.augment_ancestors(
                sample_data, ancestor_data, np.arange(sample_data.num_samples)
            )


class TestRoundTrip:
    """
    Test that we can round-trip data tsinfer.
    """

    def assert_lossless(self, ts, genotypes, positions, alleles, sequence_length):
        assert ts.sequence_length == sequence_length
        assert ts.num_sites == len(positions)
        # Make sure we've computed the mutation parents properly.
        tables = ts.dump_tables()
        tables.compute_mutation_parents()
        assert np.array_equal(ts.tables.mutations.parent, tables.mutations.parent)
        for v in ts.variants():
            site_id = v.site.id
            missing = genotypes[site_id] == tskit.MISSING_DATA
            non_missing = genotypes[site_id] != tskit.MISSING_DATA
            assert v.position == positions[site_id]
            if alleles is None or len(alleles[site_id]) == 2:
                assert np.array_equal(
                    genotypes[site_id, non_missing], v.genotypes[non_missing]
                )
            else:
                a = alleles[site_id]
                assert v.site.ancestral_state == a[0]
                assert set(v.alleles) <= set(a)
                a1 = np.array(a)
                a2 = np.array(v.alleles)
                assert np.array_equal(
                    a1[genotypes[site_id, non_missing]], a2[v.genotypes[non_missing]]
                )
            assert np.all(v.genotypes[missing] >= 0)

    def create_sample_data(
        self,
        genotypes,
        positions,
        alleles,
        sequence_length,
        site_times,
        individual_times,
    ):
        if sequence_length is None:
            sequence_length = positions[-1] + 1
        with tsinfer.SampleData(sequence_length=sequence_length) as sample_data:
            for i in range(genotypes.shape[1]):
                t = 0 if individual_times is None else individual_times[i]
                sample_data.add_individual(ploidy=1, time=t)
            for j in range(genotypes.shape[0]):
                t = None if site_times is None else site_times[j]
                site_alleles = None if alleles is None else alleles[j]
                sample_data.add_site(positions[j], genotypes[j], site_alleles, time=t)
        return sample_data

    def verify_data_round_trip(
        self,
        genotypes,
        positions,
        alleles=None,
        sequence_length=None,
        site_times=None,
        individual_times=None,
    ):
        sample_data = self.create_sample_data(
            genotypes, positions, alleles, sequence_length, site_times, individual_times
        )
        test_params = [
            {"engine": tsinfer.PY_ENGINE},
            {"engine": tsinfer.C_ENGINE},
            {"simplify": True},
            {"simplify": False},
            {"path_compression": True},
            {"path_compression": False},
        ]
        for params in test_params:
            ts = tsinfer.infer(sample_data, **params)
            self.assert_lossless(
                ts, genotypes, positions, alleles, sample_data.sequence_length
            )
            assert ts.num_provenances > 0

    def verify_round_trip(self, ts):
        positions = [site.position for site in ts.sites()]
        alleles = [v.alleles for v in ts.variants()]
        times = np.array([ts.node(site.mutations[0].node).time for site in ts.sites()])
        self.verify_data_round_trip(
            ts.genotype_matrix(),
            positions,
            alleles,
            ts.sequence_length,
            site_times=times,
        )
        # Do the same with pathological times. We add one to make sure there are no zeros
        times += 1
        self.verify_data_round_trip(
            ts.genotype_matrix(),
            positions,
            alleles,
            ts.sequence_length,
            site_times=times[::-1],
        )

    def test_simple_example(self):
        rho = 2
        ts = msprime.simulate(
            10, mutation_rate=10, recombination_rate=rho, random_seed=1
        )
        assert ts.num_sites > 0
        self.verify_round_trip(ts)

    def test_single_locus(self):
        ts = msprime.simulate(5, mutation_rate=1, recombination_rate=0, random_seed=2)
        assert ts.num_sites > 0
        self.verify_round_trip(ts)

    def test_single_locus_two_samples(self):
        ts = msprime.simulate(2, mutation_rate=1, recombination_rate=0, random_seed=3)
        assert ts.num_sites > 0
        self.verify_round_trip(ts)

    def test_two_samples_one_site(self):
        self.verify_data_round_trip(np.array([[1, 1]]), [0])

    def test_two_samples_two_sites(self):
        self.verify_data_round_trip(np.array([[1, 1], [0, 1]]), [0, 1])

    def test_random_data_invariant_sites_ancestral_state(self):
        G, positions = get_random_data_example(24, 35)
        # Set some sites to be invariant for the ancestral state
        G[10, :] = 0
        G[15, :] = 0
        G[20, :] = 0
        G[22, :] = 0
        self.verify_data_round_trip(G, positions)

    def test_random_data_invariant_sites(self):
        G, positions = get_random_data_example(39, 25)
        # Set some sites to be invariant
        G[10, :] = 1
        G[15, :] = 0
        G[20, :] = 1
        G[22, :] = 0
        self.verify_data_round_trip(G, positions)

    def test_triallelic(self, small_ts_fixture):
        mutation_0_node = next(small_ts_fixture.mutations()).node
        parent_to_mutation_0 = small_ts_fixture.first().parent(mutation_0_node)
        tables = small_ts_fixture.dump_tables()
        # Add another mutation at site 0
        tables.mutations.add_row(0, node=mutation_0_node, derived_state="2")
        mutation_nodes = tables.mutations.node
        mutation_nodes[0] = parent_to_mutation_0
        tables.mutations.node = mutation_nodes
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        # Check the first site is now triallelic
        assert next(ts.variants()).num_alleles == 3
        self.verify_round_trip(ts)

    def test_n_allelic(self):
        G = np.zeros((10, 64), dtype=int)
        G[:, 0] = 1
        G[0] = np.arange(64)
        alleles = [[str(x) for x in np.unique(a)] for a in G]
        self.verify_data_round_trip(G, np.arange(G.shape[0]), alleles=alleles)

    def test_too_many_alleles(self):
        # Max number of alleles for map_mutations is 64
        G = np.zeros((10, 65), dtype=int)
        G[:, 0] = 1
        G[0] = np.arange(65)
        alleles = [[str(x) for x in np.unique(a)] for a in G]
        with pytest.raises(ValueError):
            self.verify_data_round_trip(G, np.arange(G.shape[0]), alleles=alleles)

    def test_not_all_alleles_in_genotypes(self):
        G = np.zeros((10, 10), dtype=int)
        G = np.zeros((10, 10), dtype=int)
        G[:, 0] = 1
        G[0] = np.repeat(np.arange(4, 9), 2)  # Miss some out
        alleles = [[str(x) for x in np.arange(np.max(a) + 1)] for a in G]
        self.verify_data_round_trip(G, np.arange(G.shape[0]), alleles=alleles)

    def test_all_derived(self):
        G = np.zeros((10, 10), dtype=int)
        self.verify_data_round_trip(G, np.arange(G.shape[0]))

    def test_all_derived_or_ancestral(self):
        G = np.zeros((10, 10), dtype=int)
        G[::2] = 1
        self.verify_data_round_trip(G, np.arange(G.shape[0]))

    def test_random_data_large_example(self):
        G, positions = get_random_data_example(20, 30)
        self.verify_data_round_trip(G, positions)

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            S, positions = get_random_data_example(5, 10)
            self.verify_data_round_trip(S, positions)

    def test_unreferenced_individuals(self, small_sd_fixture):
        sd = small_sd_fixture.copy()
        n = sd.num_samples
        assert n % 2 == 0
        # We've made it pretty hard to remove samples without removing their individuals
        # Reverse individual ids & remove the last sample => individual 0 unreferenced
        sd.data["samples/individual"][:] = sd.data["samples/individual"][:][::-1]
        sd.data["samples/individual"].resize(n - 1)
        sd.data["sites/genotypes"].resize(sd.num_sites, n - 1)
        sd.finalise()
        assert sd.num_samples != sd.num_individuals
        ts = tsinfer.infer(sd)
        assert ts.num_individuals == n
        for sd_sample, ts_sample_id in zip(sd.samples(), ts.samples()):
            assert sd_sample.individual > 0
            assert sd_sample.individual == ts.node(ts_sample_id).individual

    def test_unreferenced_populations(self):
        """
        Check that the population IDs stay the same, even when unused populations exist
        """
        population_configurations = [
            msprime.PopulationConfiguration(sample_size=1),
            msprime.PopulationConfiguration(sample_size=1),
            msprime.PopulationConfiguration(sample_size=1),
        ]
        migration_matrix = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
        ts = msprime.simulate(
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,
            mutation_rate=5,
            random_seed=16,
        )
        sd = tsinfer.SampleData.from_tree_sequence(ts)
        sd = sd.subset(individuals=[1, 2])  # Remove the first individual + pop
        assert sd.num_populations != sd.num_samples
        ts_inferred = tsinfer.infer(sd)
        assert ts.num_populations == ts_inferred.num_populations
        for sd_sample, ts_sample_id in zip(sd.samples(), ts_inferred.samples()):
            sd_individual = sd.individual(sd_sample.individual)
            assert sd_individual.population > 0
            assert sd_individual.population == ts_inferred.node(ts_sample_id).population


class TestAugmentedAncestorsRoundTrip(TestRoundTrip):
    """
    Tests that we correctly round trip data when we have augmented ancestors.
    """

    def verify_data_round_trip(
        self,
        genotypes,
        positions,
        alleles=None,
        sequence_length=None,
        site_times=None,
        individual_times=None,
    ):
        sample_data = self.create_sample_data(
            genotypes, positions, alleles, sequence_length, site_times, individual_times
        )
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestors)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            augmented_ts = tsinfer.augment_ancestors(
                sample_data,
                ancestors_ts,
                np.arange(sample_data.num_samples),
                engine=engine,
            )
            ts = tsinfer.match_samples(sample_data, augmented_ts, engine=engine)
            self.assert_lossless(
                ts, genotypes, positions, alleles, sample_data.sequence_length
            )


class TestSampleMutationsRoundTrip(TestRoundTrip):
    """
    Test that we can round-trip data when we allow mutations over samples.
    """

    def verify_data_round_trip(
        self,
        genotypes,
        positions,
        alleles=None,
        sequence_length=None,
        site_times=None,
        individual_times=None,
    ):
        sample_data = self.create_sample_data(
            genotypes, positions, alleles, sequence_length, site_times, individual_times
        )
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestors)
        rho = [1e-9, 1e-3, 0.1]
        mis = [1e-9, 1e-3, 0.1]
        engines = [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]
        for rec, mis, engine in itertools.product(rho, mis, engines):
            # Set the arrays directly
            recombination = np.full(max(ancestors_ts.num_sites - 1, 0), rec)
            mismatch = np.full(ancestors_ts.num_sites, mis)
            ts = tsinfer.match_samples(
                sample_data,
                ancestors_ts,
                recombination=recombination,
                mismatch=mismatch,
                engine=engine,
            )
            self.assert_lossless(
                ts, genotypes, positions, alleles, sample_data.sequence_length
            )


class TestTruncateAncestorsRoundTrip(TestRoundTrip):
    """
    Tests that we can correctly round trip data when we truncate ancestral haplotypes
    """

    def verify_data_round_trip(
        self,
        genotypes,
        positions,
        alleles=None,
        sequence_length=None,
        site_times=None,
        individual_times=None,
    ):
        sample_data = self.create_sample_data(
            genotypes, positions, alleles, sequence_length, site_times, individual_times
        )
        ancestors = tsinfer.generate_ancestors(sample_data)
        time = np.sort(ancestors.ancestors_time[:])
        if len(time) > 0:  # Some tests produce an AncestorData file with no ancestors
            lower_bound = np.min(time)
            upper_bound = np.max(time)
            midpoint = np.median(time)
            params = [
                (lower_bound, upper_bound, 0.1),
                (lower_bound, upper_bound, 1),
                (midpoint, midpoint + (midpoint / 2), 1),
            ]
        else:
            params = [(0.4, 0.6, 1), (0, 1, 10)]
        for param in params:
            truncated_ancestors = ancestors.truncate_ancestors(
                param[0], param[1], param[2]
            )
            engines = [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]
            for engine in engines:
                ancestors_ts = tsinfer.match_ancestors(
                    sample_data, truncated_ancestors, engine=engine
                )
                ts = tsinfer.match_samples(
                    sample_data,
                    ancestors_ts,
                    engine=engine,
                )
                self.assert_lossless(
                    ts, genotypes, positions, alleles, sample_data.sequence_length
                )


class TestSparseAncestorsRoundTrip(TestRoundTrip):
    """
    Tests that we correctly round trip data when we generate the sparsest possible
    set of ancestral haplotypes.
    """

    def verify_data_round_trip(
        self,
        genotypes,
        positions,
        alleles=None,
        sequence_length=None,
        site_times=None,
        individual_times=None,
    ):
        sample_data = self.create_sample_data(
            genotypes, positions, alleles, sequence_length, site_times, individual_times
        )

        num_alleles = sample_data.num_alleles()
        with tsinfer.AncestorData(sample_data) as ancestor_data:
            t = np.sum(num_alleles) + 1
            for j in range(sample_data.num_sites):
                for allele in range(num_alleles[j] - 1):
                    ancestor_data.add_ancestor(j, j + 1, t, [j], [allele])
                    t -= 1
        engines = [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]
        for engine in engines:
            ancestors_ts = tsinfer.match_ancestors(
                sample_data, ancestor_data, engine=engine
            )
            ts = tsinfer.match_samples(
                sample_data,
                ancestors_ts,
                recombination_rate=1e-3,
                engine=engine,
            )
            self.assert_lossless(
                ts, genotypes, positions, alleles, sample_data.sequence_length
            )

    # Skipping these tests as the HMM is currently not working properly
    # for > 2 alleles, and we we have a guard on this just to make
    # sure that no user-data uses the faulty engine. Renable these
    # when the HMM is fixed.

    @pytest.mark.skip("Not currently working for > 2 alleles; #415")
    def test_triallelic(self):
        pass

    @pytest.mark.skip("Not currently working for > 2 alleles; #415")
    def test_n_allelic(self):
        pass

    @pytest.mark.skip("Not currently working for > 2 alleles; #415")
    def test_not_all_alleles_in_genotypes(self):
        pass


class TestMissingDataRoundTrip(TestRoundTrip):
    """
    Tests that we correctly round trip genotypes when missing data is present.
    We expect all non-missing values to be exactly round tripped and missing
    data to be non-missing in the output.
    """

    def verify_data_round_trip(
        self,
        genotypes,
        positions,
        alleles=None,
        sequence_length=None,
        site_times=None,
        individual_times=None,
    ):
        genotypes = genotypes.copy()
        m, n = genotypes.shape
        for j in range(m):
            genotypes[j, j % n] = tskit.MISSING_DATA
        sample_data = self.create_sample_data(
            genotypes, positions, alleles, sequence_length, site_times, individual_times
        )

        engines = [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]
        for engine in engines:
            ts = tsinfer.infer(
                sample_data,
                recombination_rate=1e-3,
                precision=10,
                engine=engine,
            )
            self.assert_lossless(
                ts, genotypes, positions, alleles, sample_data.sequence_length
            )


class TestNonInferenceSitesRoundTrip:
    """
    Test that we can round-trip data when we have various combinations
    of inference and non inference sites.
    """

    def verify_round_trip(self, genotypes, exclude_sites):
        assert genotypes.shape[0] == exclude_sites.shape[0]
        with tsinfer.SampleData() as sample_data:
            for j in range(genotypes.shape[0]):
                sample_data.add_site(j, genotypes[j])
        exclude_positions = sample_data.sites_position[:][exclude_sites]
        for simplify in [False, True]:
            output_ts = tsinfer.infer(
                sample_data, simplify=simplify, exclude_positions=exclude_positions
            )
            for tree in output_ts.trees():
                for site in tree.sites():
                    inf_type = json.loads(site.metadata)["inference_type"]
                    if exclude_sites[site.id]:
                        assert inf_type in (
                            tsinfer.INFERENCE_PARSIMONY,
                            tsinfer.INFERENCE_NONE,
                        )
                    else:
                        assert inf_type == tsinfer.INFERENCE_FULL
                    f = np.sum(genotypes[site.id])
                    if f == 0:
                        assert len(site.mutations) == 0
                    elif f == output_ts.num_samples:
                        assert len(site.mutations) == 1
                        assert site.mutations[0].node == tree.root
                    assert len(site.mutations) < output_ts.num_samples
            assert np.array_equal(genotypes, output_ts.genotype_matrix())

    def test_simple_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=10)
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        self.verify_round_trip(genotypes, ~inference)

    def test_half_sites_single_tree(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=20)
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        inference[::2] = False
        self.verify_round_trip(genotypes, ~inference)

    def test_simple_many_trees(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=10)
        assert ts.num_trees > 2
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        self.verify_round_trip(genotypes, ~inference)

    def test_half_sites_many_trees(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=11)
        assert ts.num_trees > 2
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        inference[::2] = False
        self.verify_round_trip(genotypes, ~inference)

    def test_zero_inference_sites(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        inference = np.sum(genotypes, axis=1) > 1
        inference[:] = False
        self.verify_round_trip(genotypes, ~inference)

    def test_random_data(self):
        genotypes, _ = get_random_data_example(20, 50, seed=12345)
        inference = np.sum(genotypes, axis=1) > 1
        inference[::2] = False
        self.verify_round_trip(genotypes, ~inference)


class TestZeroNonInferenceSites(unittest.TestCase):
    """
    Test the case where we have no non-inference sites.
    """

    def verify(self, sample_data):
        with self.assertLogs("tsinfer.inference", level="INFO") as logs:
            ts = tsinfer.infer(sample_data)
        messages = [record.msg for record in logs.records]
        assert "Skipping additional site mapping" in messages
        tsinfer.verify(sample_data, ts)
        return ts

    def test_many_sites(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        non_singletons = np.sum(genotypes, axis=1) > 1
        genotypes = genotypes[non_singletons]
        m = genotypes.shape[1]
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for g, position in zip(genotypes, np.arange(m)):
                sample_data.add_site(position, g)
        self.verify(sample_data)

    def test_many_sites_letter_alleles(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        assert ts.num_sites > 2
        genotypes = ts.genotype_matrix()
        non_singletons = np.sum(genotypes, axis=1) > 1
        genotypes = genotypes[non_singletons]
        m = genotypes.shape[1]
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for g, position in zip(genotypes, np.arange(m)):
                sample_data.add_site(position, g, alleles=["A", "G"])
        self.verify(sample_data)

    def test_one_site(self):
        genotypes = np.array([[1, 1, 0]])
        m = genotypes.shape[1]
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for g, position in zip(genotypes, np.arange(m)):
                sample_data.add_site(position, g)
        self.verify(sample_data)


class TestZeroInferenceSites:
    """
    Tests for the degenerate case in which we have no inference sites.
    """

    def verify(self, genotypes):
        genotypes = np.array(genotypes, dtype=np.int8)
        m = genotypes.shape[0]
        with tsinfer.SampleData(sequence_length=m + 1) as sample_data:
            for j in range(m):
                sample_data.add_site(j, genotypes[j])
        exclude_positions = sample_data.sites_position
        for path_compression in [False, True]:
            output_ts = tsinfer.infer(
                sample_data,
                path_compression=path_compression,
                exclude_positions=exclude_positions,
            )
            for tree in output_ts.trees():
                assert tree.num_roots == 1
            for site in output_ts.sites():
                inf_type = json.loads(site.metadata)["inference_type"]
                if len(site.mutations) == 0:
                    assert inf_type == tsinfer.INFERENCE_NONE
                else:
                    assert inf_type == tsinfer.INFERENCE_PARSIMONY

    def test_many_sites(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=4, random_seed=21)
        assert ts.num_sites > 2
        self.verify(ts.genotype_matrix())

    def test_one_site(self):
        self.verify([[0, 0]])
        self.verify([[0, 1]])
        self.verify([[1, 0]])
        self.verify([[1, 1]])
        self.verify([[1, 1, 1]])

    def test_two_sites(self):
        self.verify([[0, 0], [0, 0]])
        self.verify([[1, 1], [1, 1]])
        self.verify([[0, 0, 0], [0, 0, 0]])
        self.verify([[0, 1, 0], [1, 0, 0]])

    def test_three_sites(self):
        self.verify([[0, 0], [0, 0], [0, 0]])
        self.verify([[1, 1], [1, 1], [1, 1]])


def random_string(rng, max_len=10):
    """
    Uses the specified random generator to generate a random string.
    """
    s = ""
    for _ in range(rng.randint(0, max_len)):
        s += rng.choice(string.ascii_letters)
    return s


def get_multichar_alleles_example(sample_size):
    """
    Returns an example dataset with multichar alleles.
    """
    ts = msprime.simulate(10, mutation_rate=10, recombination_rate=1, random_seed=5)
    assert ts.num_sites > 2
    sample_data = tsinfer.SampleData(sequence_length=1)
    rng = random.Random(32)
    all_alleles = []
    for variant in ts.variants():
        ancestral = random_string(rng)
        derived = ancestral
        while derived == ancestral:
            derived = random_string(rng)
        alleles = ancestral, derived
        sample_data.add_site(variant.site.position, variant.genotypes, alleles)
        all_alleles.append(alleles)
    sample_data.finalise()
    return sample_data


class TestMetadataRoundTrip:
    """
    Tests if we can round-trip various forms of metadata.
    """

    def test_multichar_alleles(self):
        ts = msprime.simulate(10, mutation_rate=10, recombination_rate=1, random_seed=5)
        assert ts.num_sites > 2
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_alleles = []
        for variant in ts.variants():
            ancestral = random_string(rng)
            derived = ancestral
            while derived == ancestral:
                derived = random_string(rng)
            alleles = ancestral, derived
            sample_data.add_site(variant.site.position, variant.genotypes, alleles)
            all_alleles.append(alleles)
        sample_data.finalise()

        for j, alleles in enumerate(sample_data.sites_alleles[:]):
            assert all_alleles[j] == tuple(alleles)

        output_ts = tsinfer.infer(sample_data)
        inferred_alleles = [variant.alleles for variant in output_ts.variants()]
        assert inferred_alleles == all_alleles

    def test_ts_metadata(self):
        metadata = {f"x_{j}": j for j in range(10)}
        schema = tsinfer.permissive_json_schema()
        for j in range(10):
            name = f"x_{j}"
            metadata[name] = j
            schema = tsinfer.add_to_schema(schema, name, {"type": "number"})
        with tsinfer.SampleData(sequence_length=1) as sample_data:
            sample_data.metadata = metadata
            sample_data.metadata_schema = schema
            sample_data.add_site(0.5, [0, 1])
        output_ts = tsinfer.infer(sample_data)
        assert output_ts.metadata == metadata
        assert output_ts.metadata_schema.schema == schema

    @pytest.mark.parametrize("use_schema", [True, False])
    def test_site_metadata(self, use_schema):
        ts = msprime.simulate(11, mutation_rate=5, recombination_rate=2, random_seed=15)
        assert ts.num_sites > 2
        sample_data = tsinfer.SampleData(sequence_length=1)
        if use_schema:
            sample_data.sites_metadata_schema = tsinfer.permissive_json_schema()
        rng = random.Random(32)
        all_metadata = []
        for variant in ts.variants():
            metadata = {str(j): random_string(rng) for j in range(rng.randint(0, 5))}
            sample_data.add_site(
                variant.site.position,
                variant.genotypes,
                alleles=["A", "T"],
                metadata=metadata,
            )
            all_metadata.append(metadata)
        sample_data.finalise()

        for j, metadata in enumerate(sample_data.sites_metadata[:]):
            assert all_metadata[j] == metadata

        for variant in sample_data.variants():
            assert all_metadata[variant.site.id] == variant.site.metadata

        output_ts = tsinfer.infer(sample_data)
        for variant in output_ts.variants():
            site = variant.site
            decoded_metadata = (
                site.metadata if use_schema else json.loads(site.metadata)
            )
            assert "inference_type" in decoded_metadata
            value = decoded_metadata.pop("inference_type")
            # Only singletons should be parsimony sites in this simple case
            if np.sum(variant.genotypes > 0) == 1:
                assert value == tsinfer.INFERENCE_PARSIMONY
            else:
                assert value == tsinfer.INFERENCE_FULL
            assert decoded_metadata == all_metadata[site.id]

    @pytest.mark.parametrize("no_schema", [True, False])
    def test_population_metadata(self, no_schema):
        ts = msprime.simulate(12, mutation_rate=5, random_seed=16)
        assert ts.num_sites > 2
        sample_data = tsinfer.SampleData(sequence_length=1)
        if no_schema:
            sample_data.populations_metadata_schema = None
        rng = random.Random(32)
        all_metadata = []
        for j in range(ts.num_samples):
            metadata = {str(j): random_string(rng) for j in range(rng.randint(0, 5))}
            sample_data.add_population(metadata=metadata)
            all_metadata.append(metadata)
        for j in range(ts.num_samples):
            sample_data.add_individual(population=j)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles
            )
        sample_data.finalise()

        for j, metadata in enumerate(sample_data.populations_metadata[:]):
            assert all_metadata[j] == metadata
        output_ts = tsinfer.infer(sample_data)
        output_metadata = [
            json.loads(population.metadata) if no_schema else population.metadata
            for population in output_ts.populations()
        ]
        assert all_metadata == output_metadata
        for j, sample in enumerate(output_ts.samples()):
            node = output_ts.node(sample)
            assert node.population == j

    @pytest.mark.parametrize("use_schema", [True, False])
    def test_individual_metadata(self, use_schema):
        ts = msprime.simulate(11, mutation_rate=5, random_seed=16)
        assert ts.num_sites > 2
        sample_data = tsinfer.SampleData(sequence_length=1)
        if use_schema:
            sample_data.individuals_metadata_schema = tsinfer.permissive_json_schema()
        rng = random.Random(32)
        all_metadata = []
        for j in range(ts.num_samples):
            metadata = {str(j): random_string(rng) for j in range(rng.randint(0, 5))}
            sample_data.add_individual(metadata=metadata)
            all_metadata.append(metadata)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles
            )
        sample_data.finalise()

        for j, metadata in enumerate(sample_data.individuals_metadata[:]):
            assert all_metadata[j] == metadata
        output_ts = tsinfer.infer(sample_data)
        output_metadata = [
            individual.metadata if use_schema else json.loads(individual.metadata)
            for individual in output_ts.individuals()
        ]
        # output metadata can have some extra fields, e.g. "sample_data_time"
        # so check all_metadata is "contained in" output_metadata
        for all, output in zip(all_metadata, output_metadata):
            assert all.items() <= output.items()

    @pytest.mark.parametrize("use_schema", [True, False])
    def test_individual_metadata_subset(self, use_schema):
        ts = msprime.simulate(15, mutation_rate=4, random_seed=16)
        assert ts.num_sites > 2
        sample_data = tsinfer.SampleData(sequence_length=1)
        if use_schema:
            sample_data.individuals_metadata_schema = tsinfer.permissive_json_schema()
        rng = random.Random(132)
        all_metadata = []
        for _ in range(ts.num_samples):
            sample_data.add_population()
        for j in range(ts.num_samples):
            metadata = {str(j): random_string(rng) for j in range(rng.randint(1, 6))}
            location = [rng.randint(-100, 100)]
            sample_data.add_individual(
                metadata=metadata, location=location, population=j
            )
            all_metadata.append(metadata)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles
            )
        sample_data.finalise()

        output_ts = tsinfer.infer(sample_data)
        output_metadata = [
            individual.metadata if use_schema else json.loads(individual.metadata)
            for individual in output_ts.individuals()
        ]
        # output metadata can have some extra fields, e.g. "sample_data_time"
        # so check all_metadata is "contained in" output_metadata
        for all, output in zip(all_metadata, output_metadata):
            assert all.items() <= output.items()
        for j, metadata in enumerate(sample_data.individuals_metadata[:]):
            assert all_metadata[j].items() <= metadata.items()

        # Now do this for various subsets of the data and make sure
        # that metadata comes through correctly.
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestors)
        for subset in [[0], [0, 1], [1], [2, 3, 4, 5]]:
            t1 = output_ts.simplify(subset).dump_tables()
            assert len(t1.individuals.metadata) > 0
            assert len(t1.individuals.location) > 0
            t1.provenances.clear()
            t2 = tsinfer.match_samples(
                sample_data, ancestors_ts, indexes=subset
            ).dump_tables()
            t2.simplify()
            assert t1.equals(t2, ignore_provenance=True)

    def test_individual_location(self):
        ts = msprime.simulate(12, mutation_rate=5, random_seed=16)
        assert ts.num_sites > 2
        sample_data = tsinfer.SampleData(sequence_length=1)
        rng = random.Random(32)
        all_locations = []
        for j in range(ts.num_samples // 2):
            location = np.array([rng.random() for _ in range(j)])
            sample_data.add_individual(location=location, ploidy=2)
            all_locations.append(location)
        for variant in ts.variants():
            sample_data.add_site(
                variant.site.position, variant.genotypes, variant.alleles
            )
        sample_data.finalise()

        for j, location in enumerate(sample_data.individuals_location[:]):
            assert np.array_equal(all_locations[j], location)
        output_ts = tsinfer.infer(sample_data)
        assert output_ts.num_individuals == len(all_locations)
        for location, individual in zip(all_locations, output_ts.individuals()):
            assert np.array_equal(location, individual.location)

    @pytest.mark.parametrize("use_schema", [True, False])
    def test_historical_individuals(self, use_schema):
        samples = [msprime.Sample(population=0, time=0) for i in range(10)]
        rng = random.Random(32)
        ages = [rng.random(), rng.random()]
        historical_samples = [
            msprime.Sample(population=0, time=ages[i // 2]) for i in range(4)
        ]
        samples = samples + historical_samples
        ts = msprime.simulate(samples=samples, mutation_rate=5, random_seed=16)
        with tsinfer.SampleData(sequence_length=1) as sample_data:
            if use_schema:
                sample_data.individuals_metadata_schema = (
                    tsinfer.permissive_json_schema()
                )
            all_times = []
            for j in range(ts.num_samples // 2):
                time = samples[2 * j].time
                sample_data.add_individual(time=time, ploidy=2)
                all_times.append(time)
            for variant in ts.variants():
                sample_data.add_site(
                    variant.site.position, variant.genotypes, variant.alleles
                )
        for j, time in enumerate(sample_data.individuals_time[:]):
            assert np.array_equal(all_times[j], time)
        output_ts = tsinfer.infer(sample_data)
        assert output_ts.num_individuals == len(all_times)
        flags = output_ts.tables.nodes.flags
        flags_for_historical_sample = (
            tsinfer.NODE_IS_HISTORICAL_SAMPLE | tskit.NODE_IS_SAMPLE
        )
        for time, individual in zip(all_times, output_ts.individuals()):
            for node in individual.nodes:
                if time != 0:
                    assert flags[node] == flags_for_historical_sample
            if time != 0:
                md = (
                    individual.metadata
                    if use_schema
                    else json.loads(individual.metadata)
                )
                assert np.array_equal(time, md["sample_data_time"])

    def test_from_standard_tree_sequence(self):
        """
        Test that we can roundtrip most user-specified data (e.g. metadata, etc) from
        a tree seq, through a sample data file, back to an inferred tree sequence, as
        long as individuals are defined in the original tree seq.
        """
        n_indiv = 5
        ploidy = 2  # Diploids
        ts = tsutil.get_example_individuals_ts_with_metadata(
            n_indiv, ploidy, skip_last=False
        )
        ts_inferred = tsinfer.infer(tsinfer.SampleData.from_tree_sequence(ts))
        assert ts.sequence_length == ts_inferred.sequence_length
        assert ts.metadata_schema.schema == ts_inferred.metadata_schema.schema
        assert ts.metadata == ts_inferred.metadata
        assert ts.tables.populations.equals(
            ts_inferred.tables.populations, ignore_metadata=True
        )
        # Check all the metadata in pops is in inferred pops (in addition to names etc)
        for pop, pop_inferred in zip(ts.populations(), ts_inferred.populations()):
            for key in pop.metadata.keys():
                assert pop.metadata[key] == pop_inferred.metadata[key]
        assert ts.num_individuals == ts_inferred.num_individuals
        for i1, i2 in zip(ts.individuals(), ts_inferred.individuals()):
            assert list(i1.location) == list(i2.location)
            assert i1.flags == i2.flags
            assert tsutil.json_metadata_is_subset(i1.metadata, i2.metadata)
        # Unless inference is perfect, internal nodes may differ, but sample nodes
        # should be identical
        for n1, n2 in zip(ts.samples(), ts_inferred.samples()):
            assert ts.node(n1) == ts_inferred.node(n2)
        # Sites can have metadata added by the inference process, but inferred site
        # metadata should always include all the metadata in the original ts
        for s1, s2 in zip(ts.sites(), ts_inferred.sites()):
            assert s1.position == s2.position
            assert s1.ancestral_state == s2.ancestral_state
            assert tsutil.json_metadata_is_subset(s1.metadata, s2.metadata)

    def test_from_historical_tree_sequence(self):
        """
        Test that we can roundtrip most user-specified data (e.g. metadata, etc) from
        a tree sequence with non-contemporaneous samples. These are a special case, as
        we don't have a sensible time scale to set the node times, so the original node
        times get placed in metadata, and a flag set
        """
        n_indiv = 5
        ploidy = 2  # Diploids
        individual_times = np.arange(n_indiv)
        ts = tsutil.get_example_historical_sampled_ts(individual_times, ploidy)
        ts_inferred = tsinfer.infer(
            tsinfer.SampleData.from_tree_sequence(
                ts, use_sites_time=True, use_individuals_time=True
            )
        )
        assert ts.sequence_length == ts_inferred.sequence_length
        assert ts.metadata_schema == ts_inferred.metadata_schema
        assert ts.metadata == ts_inferred.metadata
        assert ts.tables.populations == ts_inferred.tables.populations
        # Historical individuals have metadata added by the inference process
        # specifying the original time of the samples with which they are associated
        for i1, i2 in zip(ts.individuals(), ts_inferred.individuals()):
            assert i1.flags == i2.flags
            assert np.array_equal(i1.location, i2.location)
            assert np.array_equal(i1.nodes, i2.nodes)
            assert tsutil.json_metadata_is_subset(i1.metadata, i2.metadata)
        # Sample nodes can have tsinfer.NODE_IS_HISTORICAL_SAMPLE in flags, and need not
        # have the same node time
        for n1, n2 in zip(ts.samples(), ts_inferred.samples()):
            node1 = ts.node(n1)
            node2 = ts_inferred.node(n2)
            assert node1.population == node2.population
            assert node1.individual == node2.individual
            if node2.flags & tsinfer.NODE_IS_HISTORICAL_SAMPLE == 0:
                assert node1.time == node2.time
                assert node1.flags == node2.flags
            else:
                node2_other_flags = node2.flags ^ tsinfer.NODE_IS_HISTORICAL_SAMPLE
                assert node1.flags == node2_other_flags
        # Sites can have metadata added by the inference process, but inferred site
        # metadata should always include all the metadata in the original ts
        for s1, s2 in zip(ts.sites(), ts_inferred.sites()):
            assert s1.position == s2.position
            assert s1.ancestral_state == s2.ancestral_state
            assert tsutil.json_metadata_is_subset(i1.metadata, i2.metadata)


class TestThreads:
    def test_equivalance(self):
        rho = 2
        ts = msprime.simulate(5, mutation_rate=2, recombination_rate=rho, random_seed=2)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        ts1 = tsinfer.infer(sample_data, num_threads=1)
        ts2 = tsinfer.infer(sample_data, num_threads=5)
        assert ts1.equals(ts2, ignore_provenance=True)


class TestAncestorGeneratorsEquivalant:
    """
    Tests for the ancestor generation process.
    """

    def verify_ancestor_generator(self, genotypes, times=None, num_threads=0):
        m, n = genotypes.shape
        with tsinfer.SampleData() as sample_data:
            for j in range(m):
                t = None if times is None else times[j]
                sample_data.add_site(j, genotypes[j], time=t)

        adc = tsinfer.generate_ancestors(
            sample_data, engine=tsinfer.C_ENGINE, num_threads=num_threads
        )
        adp = tsinfer.generate_ancestors(
            sample_data, engine=tsinfer.PY_ENGINE, num_threads=num_threads
        )

        # TODO clean this up when we're finished mucking around with the
        # ancestor generator.
        # print()
        # print(adc.ancestors_start[:])
        # print(adp.ancestors_start[:])
        # assert np.array_equal(adc.ancestors_start[:], adp.ancestors_start[:])

        # print("end:")
        # print(adc.ancestors_end[:])
        # print(adp.ancestors_end[:])
        # assert np.array_equal(adc.ancestors_end[:], adp.ancestors_end[:])

        # print("focal_sites:")
        # print(adc.ancestors_focal_sites[:])
        # print(adp.ancestors_focal_sites[:])
        # for fc, fp in zip(adc.ancestors_focal_sites[:], adp.ancestors_focal_sites[:]):
        #     assert np.array_equal(fc, fp)

        # print("haplotype:")
        # print(adc.ancestors_haplotype[:])
        # print()
        # print(adp.ancestors_haplotype[:])

        # j = 0
        # for h1, h2 in zip(adc.ancestors_haplotype[:], adp.ancestors_haplotype[:]):
        #     if not np.array_equal(h1, h2):
        #         print("ANCESTOR = ", j)
        #         print(h1)
        #         print(h2)
        #         print(adp.ancestors_focal_sites[j])
        #         # print(adc.ancestors_focal_sites[j])
        #         # print(adc.ancestors_start[j])
        #         # print(adc.ancestors_end[j])
        #     j += 1
        # print(adc)
        # print(adp)
        assert adp.data_equal(adc)
        return adp, adc

    def verify_tree_sequence(self, ts):
        self.verify_ancestor_generator(ts.genotype_matrix())
        t = np.array([ts.node(site.mutations[0].node).time for site in ts.sites()])
        self.verify_ancestor_generator(ts.genotype_matrix(), t)
        # Give some pathological times.
        t += 1
        t = t[::-1]
        self.verify_ancestor_generator(ts.genotype_matrix(), t)

    def test_no_recombination(self):
        ts = msprime.simulate(
            20, length=1, recombination_rate=0, mutation_rate=1, random_seed=1
        )
        assert ts.num_sites > 0 and ts.num_sites < 50
        self.verify_tree_sequence(ts)

    def test_with_recombination_short(self):
        ts = msprime.simulate(
            20, length=1, recombination_rate=1, mutation_rate=1, random_seed=1
        )
        assert ts.num_trees > 1
        assert ts.num_sites > 0 and ts.num_sites < 50
        self.verify_tree_sequence(ts)

    def test_with_recombination_long(self):
        ts = msprime.simulate(
            20, length=50, recombination_rate=1, mutation_rate=1, random_seed=1
        )
        assert ts.num_trees > 1
        assert ts.num_sites > 100
        self.verify_tree_sequence(ts)

    def test_random_data(self):
        G, _ = get_random_data_example(20, 50, seed=1234)
        # G, _ = get_random_data_example(20, 10)
        self.verify_ancestor_generator(G)

    def test_random_data_threads(self):
        G, _ = get_random_data_example(20, 50, seed=1234)
        # G, _ = get_random_data_example(20, 10)
        self.verify_ancestor_generator(G, num_threads=4)

    def test_random_data_missing(self):
        G, _ = get_random_data_example(20, 50, seed=1234, num_states=3)
        G[G == 2] = tskit.MISSING_DATA
        # G, _ = get_random_data_example(20, 10)
        self.verify_ancestor_generator(G)

    def test_all_missing_at_adjacent_site(self):
        u = tskit.MISSING_DATA
        G = np.array(
            [
                [1, 1, 0, 0, 0, 0],  # Site 0
                [u, u, 0, 1, 1, 1],  # Site 1
                [1, 1, 0, 0, 0, 0],  # Site 2
                [u, u, 1, 0, 1, 1],  # Site 3
                [1, 1, 1, 1, 0, 0],
            ]
        )
        adp, _ = self.verify_ancestor_generator(G)
        site_0_anc = [i for i, fs in enumerate(adp.ancestors_focal_sites[:]) if 0 in fs]
        assert len(site_0_anc) == 1
        site_0_anc = site_0_anc[0]
        # Sites 0 and 2 should share the same ancestor
        assert np.all(adp.ancestors_focal_sites[:][site_0_anc] == [0, 2])
        focal_site_0_haplotype = adp.ancestors_haplotype[:][site_0_anc]
        # High freq sites with all missing data (e.g. for sites 1 & 3 in the ancestral
        # haplotype focussed on sites 0 & 2) should default to tskit.MISSING_DATA
        expected_hap_focal_site_0 = [1, u, 1, u, 1]
        assert np.all(focal_site_0_haplotype == expected_hap_focal_site_0)

    def test_with_recombination_long_threads(self):
        ts = msprime.simulate(
            20, length=50, recombination_rate=1, mutation_rate=1, random_seed=1
        )
        assert ts.num_trees > 1
        assert ts.num_sites > 100
        self.verify_ancestor_generator(ts.genotype_matrix(), num_threads=3)


class TestGeneratedAncestors:
    """
    Ensures we work correctly with the ancestors recovered from the
    simulations.
    """

    def verify_inserted_ancestors(self, ts):
        # Verifies that we can round-trip the specified tree sequence
        # using the generated ancestors. NOTE: this must be an SMC
        # consistent tree sequence!
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as sample_data:
            for v in ts.variants():
                sample_data.add_site(v.position, v.genotypes, v.alleles)
        ancestor_data = tsinfer.AncestorData(sample_data)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()

        A = np.full(
            (ancestor_data.num_sites, ancestor_data.num_ancestors),
            tskit.MISSING_DATA,
            dtype=np.int8,
        )
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]
        ancestors = ancestor_data.ancestors_haplotype[:]
        for j in range(ancestor_data.num_ancestors):
            A[start[j] : end[j], j] = ancestors[j]
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ancestors_ts = tsinfer.match_ancestors(
                sample_data, ancestor_data, engine=engine
            )
            tsinfer.check_ancestors_ts(ancestors_ts)
            assert ancestor_data.num_sites == ancestors_ts.num_sites
            assert ancestor_data.num_ancestors == ancestors_ts.num_samples
            assert np.array_equal(ancestors_ts.genotype_matrix(), A)
            inferred_ts = tsinfer.match_samples(
                sample_data, ancestors_ts, engine=engine
            )
            assert np.array_equal(inferred_ts.genotype_matrix(), ts.genotype_matrix())

    def test_no_recombination(self):
        ts = msprime.simulate(
            20,
            length=1,
            recombination_rate=0,
            mutation_rate=1,
            random_seed=1,
            model="smc_prime",
        )
        assert ts.num_sites > 0 and ts.num_sites < 50
        self.verify_inserted_ancestors(ts)

    def test_small_sample_high_recombination(self):
        ts = msprime.simulate(
            4,
            length=1,
            recombination_rate=5,
            mutation_rate=1,
            random_seed=1,
            model="smc_prime",
        )
        assert ts.num_sites > 0 and ts.num_sites < 50 and ts.num_trees > 3
        self.verify_inserted_ancestors(ts)

    def test_high_recombination(self):
        ts = msprime.simulate(
            30,
            length=1,
            recombination_rate=5,
            mutation_rate=1,
            random_seed=1,
            model="smc_prime",
        )
        assert ts.num_sites > 0 and ts.num_sites < 50 and ts.num_trees > 3
        self.verify_inserted_ancestors(ts)


class TestBuildAncestors:
    """
    Tests for the generate_ancestors function.
    """

    def test_bad_exclude_sites(self):
        # Only things that can be interpreted as a 1D double array
        # should be accepted.
        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.5, [1, 1])
        with pytest.raises(ValueError):
            tsinfer.generate_ancestors(sample_data, exclude_positions=[[None]])

        with pytest.raises(ValueError):
            tsinfer.generate_ancestors(sample_data, exclude_positions=["not", 1.1])

    def test_bad_focal_sites(self):
        # Can't generate an ancestor for a site with no mutations
        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.5, [0, 0])
        for engine, error in [
            (tsinfer.C_ENGINE, _tsinfer.LibraryError),
            (tsinfer.PY_ENGINE, ValueError),
        ]:
            with tsinfer.formats.AncestorData(sample_data) as ancestor_data:
                g = np.zeros(2, dtype=np.int8)
                h = np.zeros(1, dtype=np.int8)
            generator = tsinfer.AncestorsGenerator(
                sample_data, ancestor_data, engine=engine
            )
            generator.ancestor_builder.add_site(1, g)
            with pytest.raises(error):
                generator.ancestor_builder.make_ancestor([0], h)

    def test_mixed_freq_and_user_times(self):
        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.4, [0, 1, 1])
            sample_data.add_site(0.8, [0, 1, 1], time=np.nan)
        tsinfer.generate_ancestors(sample_data)  # Should work

        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.4, [0, 1, 1], time=0.5)
            sample_data.add_site(0.8, [0, 1, 1], time=np.nan)
        tsinfer.generate_ancestors(sample_data)  # Should work

        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.4, [0, 1, 1], time=0.5)
            sample_data.add_site(0.8, [0, 1, 1])
        with pytest.raises(ValueError):
            tsinfer.generate_ancestors(sample_data)

    def test_nan_sites(self):
        # Sites whose time is marked as NaN but are not tskit.UNKNOWN_TIME have
        # a meaningless concept of time and should not be marked for full inference
        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.2, [1, 1, 0])
            sample_data.add_site(0.4, [1, 1, 0], time=np.nan)
            sample_data.add_site(0.6, [1, 1, 0])
        ancestors = tsinfer.generate_ancestors(sample_data)
        assert ancestors.num_sites == 2

    def get_simulated_example(self, ts):
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        return sample_data, ancestor_data

    def verify_ancestors(self, sample_data, ancestor_data):
        ancestors = ancestor_data.ancestors_haplotype[:]
        position = sample_data.sites_position[:]
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]
        times = ancestor_data.ancestors_time[:]
        focal_sites = ancestor_data.ancestors_focal_sites[:]

        assert ancestor_data.num_ancestors == ancestors.shape[0]
        assert ancestor_data.num_ancestors == times.shape[0]
        assert ancestor_data.num_ancestors == start.shape[0]
        assert ancestor_data.num_ancestors == end.shape[0]
        assert ancestor_data.num_ancestors == focal_sites.shape[0]
        assert set(ancestor_data.sites_position[:]) <= set(position)
        # The first ancestor must be all zeros.
        assert start[0] == 0
        assert end[0] == ancestor_data.num_sites
        assert list(focal_sites[0]) == []
        assert np.all(ancestors[0] == 0)

        used_sites = []
        for j in range(ancestor_data.num_ancestors):
            a = ancestors[j]
            assert a.shape[0] == end[j] - start[j]
            h = np.zeros(ancestor_data.num_sites, dtype=np.uint8)
            h[start[j] : end[j]] = a
            assert np.all(h[start[j] : end[j]] != tskit.MISSING_DATA)
            assert np.all(h[focal_sites[j]] == 1)
            used_sites.extend(focal_sites[j])
            assert times[j] > 0
            if j > 0:
                assert times[j - 1] >= times[j]
        assert sorted(used_sites) == list(range(ancestor_data.num_sites))

        # The provenance should be same as in the samples data file, plus an
        # extra row.
        assert ancestor_data.num_provenances == sample_data.num_provenances + 1
        for j in range(sample_data.num_provenances):
            assert (
                ancestor_data.provenances_record[j] == sample_data.provenances_record[j]
            )
            assert (
                ancestor_data.provenances_timestamp[j]
                == sample_data.provenances_timestamp[j]
            )

    def test_simulated_no_recombination(self):
        ts = msprime.simulate(10, mutation_rate=10, random_seed=10)
        assert ts.num_sites > 10
        sample_data, ancestor_data = self.get_simulated_example(ts)
        self.verify_ancestors(sample_data, ancestor_data)

    def test_simulated_recombination(self):
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=10, random_seed=10
        )
        assert ts.num_sites > 10
        sample_data, ancestor_data = self.get_simulated_example(ts)
        self.verify_ancestors(sample_data, ancestor_data)
        # Make sure we have at least one partial ancestor.
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]
        assert np.min(end - start) < ancestor_data.num_sites

    def test_random_data(self):
        n = 20
        m = 50
        G, positions = get_random_data_example(n, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        self.verify_ancestors(sample_data, ancestor_data)

    def test_oldest_ancestors(self):
        m = 50
        G, positions = get_random_data_example(20, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        assert ancestor_data.num_ancestors > 2
        times = ancestor_data.ancestors_time[:]
        unique_times = np.unique(times)
        ultimate_root_time = unique_times[-1]
        root_time = unique_times[-2]
        oldest_non_root = unique_times[-3]
        assert np.sum(times == root_time) == 1  # root ancestor at unique time
        assert np.sum(times == ultimate_root_time) == 1  # ultimate anc at unique time
        expected_time_diff = oldest_non_root / len(unique_times[:-2])
        assert np.isclose(ultimate_root_time - root_time, expected_time_diff)
        assert np.isclose(root_time - oldest_non_root, expected_time_diff)


class TestAncestorsTreeSequence:
    """
    Tests for the output of the match_ancestors function.
    """

    def verify(self, sample_data, mismatch_ratio=None, recombination_rate=None):
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        for path_compression in [True, False]:
            ancestors_ts = tsinfer.match_ancestors(
                sample_data,
                ancestor_data,
                path_compression=path_compression,
                mismatch_ratio=mismatch_ratio,
                recombination_rate=recombination_rate,
            )
            tsinfer.check_ancestors_ts(ancestors_ts)
            tables = ancestors_ts.dump_tables()

            # Make sure we've computed the mutation parents properly.
            tables.compute_mutation_parents()
            assert np.array_equal(
                ancestors_ts.tables.mutations.parent, tables.mutations.parent
            )
            assert np.array_equal(
                tables.sites.position, ancestor_data.sites_position[:]
            )

            assert ancestors_ts.num_samples == ancestor_data.num_ancestors
            H = ancestors_ts.genotype_matrix().T
            for ancestor in ancestor_data.ancestors():
                assert np.array_equal(
                    H[ancestor.id, ancestor.start : ancestor.end], ancestor.haplotype
                )

            # The provenance should be same as in the ancestors data file, plus an
            # extra row.
            assert ancestor_data.num_provenances + 1 == ancestors_ts.num_provenances
            for j in range(ancestor_data.num_provenances):
                p = ancestors_ts.provenance(j)
                assert ancestor_data.provenances_record[j] == json.loads(p.record)
                assert ancestor_data.provenances_timestamp[j] == p.timestamp

            # Ancestors indicated in node metadata should have the same age as their node
            ancestors_time = ancestor_data.ancestors_time[:]
            num_ancestor_nodes = 0
            for n in ancestors_ts.nodes():
                md = json.loads(n.metadata) if n.metadata else {}
                if tsinfer.is_pc_ancestor(n.flags):
                    assert not ("ancestor_data_id" in md)
                else:
                    assert "ancestor_data_id" in md
                    assert ancestors_time[md["ancestor_data_id"]] == n.time
                    num_ancestor_nodes += 1
            assert num_ancestor_nodes == ancestor_data.num_ancestors

    def test_no_recombination(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=234)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)
        self.verify(sample_data, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=100, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=0.01, recombination_rate=1e-3)

    def test_recombination(self):
        ts = msprime.simulate(
            10, mutation_rate=2, recombination_rate=2, random_seed=233
        )
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)
        self.verify(sample_data, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=100, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=0.01, recombination_rate=1e-3)

    def test_random_data(self):
        n = 25
        m = 50
        G, positions = get_random_data_example(n, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        self.verify(sample_data)
        self.verify(sample_data, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=100, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=0.01, recombination_rate=1e-3)

    def test_acgt_mutations(self):
        ts = msprime.simulate(10, recombination_rate=2, random_seed=233)
        ts = msprime.mutate(
            ts,
            rate=2,
            random_seed=1234,
            model=msprime.InfiniteSites(msprime.NUCLEOTIDES),
        )
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)
        self.verify(sample_data, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=100, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=0.01, recombination_rate=1e-3)

    def test_multi_char_alleles(self):
        sample_data = get_multichar_alleles_example(10)
        self.verify(sample_data)
        self.verify(sample_data, mismatch_ratio=100, recombination_rate=1e-9)
        self.verify(sample_data, mismatch_ratio=0.01, recombination_rate=1e-3)


class TestAncestorsTreeSequenceFlags:
    """
    Checks that arbitrary flags can be set in the ancestors tree
    sequence and recovered in the final ts.
    """

    def verify(self, sample_data, ancestors_ts):
        source_flags = ancestors_ts.tables.nodes.flags
        for engine in [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]:
            for path_compression in [True, False]:
                ts = tsinfer.match_samples(
                    sample_data,
                    ancestors_ts,
                    path_compression=path_compression,
                    simplify=False,
                    engine=engine,
                )
                nodes = ts.tables.nodes
                flags = nodes.flags[: source_flags.shape[0]]
                # Anything that's marked as a sample in the ancestors should be a
                # 0 in the final outout
                samples = np.where(source_flags == 1)[0]
                assert np.all(flags[samples] == 0)
                # Anything that's not marked as a sample should be equal in both.
                non_samples = np.where(source_flags != 1)[0]
                assert np.all(flags[non_samples] == source_flags[non_samples])

    def test_no_flags_changes(self):
        ts = msprime.simulate(
            10, mutation_rate=2, recombination_rate=2, random_seed=233
        )
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestors)
        self.verify(samples, ancestors_ts)

    def test_append_nodes(self):
        ts = msprime.simulate(
            10, mutation_rate=2, recombination_rate=2, random_seed=233
        )
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestors)
        tables = ancestors_ts.dump_tables()
        tables.nodes.add_row(flags=1 << 15, time=1.1)
        tables.nodes.add_row(flags=1 << 16, time=1.1)
        tables.nodes.add_row(flags=1 << 17, time=1.1)
        tables.nodes.add_row(flags=1 << 18, time=1.0)
        self.verify(samples, tables.tree_sequence())


class TestAncestorsTreeSequenceIndividuals:
    """
    Checks that we can have individuals in the ancestors tree sequence and
    that they are correctly preserved in the final TS.
    """

    def verify(self, sample_data, ancestors_ts):
        ts = tsinfer.match_samples(sample_data, ancestors_ts, simplify=False)
        assert (
            ancestors_ts.num_individuals + sample_data.num_individuals
            == ts.num_individuals
        )
        # The ancestors individiduals should come first.
        final_individuals = iter(ts.individuals())
        for ind in ancestors_ts.individuals():
            final_ind = next(final_individuals)
            assert final_ind == ind
            # The nodes for this individual should *not* be samples.
            for u in final_ind.nodes:
                node = ts.node(u)
                assert not node.is_sample()

        for ind1, ind2 in zip(final_individuals, sample_data.individuals()):
            assert np.array_equal(ind1.location, ind2.location)
            assert json.loads(ind1.metadata.decode()) == ind2.metadata
            # The nodes for this individual should *not* be samples.
            for u in ind1.nodes:
                node = ts.node(u)
                assert node.is_sample()

    def test_zero_individuals(self):
        ts = msprime.simulate(
            10, mutation_rate=2, recombination_rate=2, random_seed=233
        )
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        ancestors = tsinfer.generate_ancestors(samples)
        ancestors_ts = tsinfer.match_ancestors(samples, ancestors)
        self.verify(samples, ancestors_ts)

    def test_diploid_individuals(self):
        ts = msprime.simulate(
            10, mutation_rate=2, recombination_rate=2, random_seed=233
        )
        tables = ts.dump_tables()
        for j in range(ts.num_samples // 2):
            tables.individuals.add_row(flags=j, location=[j, j], metadata=b"X" * j)
        # Add these individuals to the first n nodes.
        individual = np.zeros(ts.num_nodes, dtype=np.int32) - 1
        x = np.arange(ts.num_samples // 2)
        individual[2 * x] = x
        individual[2 * x + 1] = x
        tables.nodes.set_columns(
            flags=tables.nodes.flags, time=tables.nodes.time, individual=individual
        )
        ts = tables.tree_sequence()
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for j in range(ts.num_samples // 2):
                samples.add_individual(ploidy=2, location=[100 * j], metadata={"X": j})
            for var in ts.variants():
                samples.add_site(var.site.position, var.genotypes)
        ancestors_ts = eval_util.make_ancestors_ts(samples, ts)
        self.verify(samples, ancestors_ts)


class TestMatcher:
    """
    Test features of the base Matcher class from which AncestorMatcher and SampleMatcher
    are derived
    """

    def test_recombination_rate_to_dist(self):
        d = tsinfer.inference.Matcher.recombination_rate_to_dist(10, [1, 1, 2])
        assert np.allclose(d, [0, 10])

    def test_recombination_rate_to_dist_no_dists(self):
        # No distances if 1 site
        d = tsinfer.inference.Matcher.recombination_rate_to_dist(1, [1])
        assert len(d) == 0
        # No distances if 0 sites
        d = tsinfer.inference.Matcher.recombination_rate_to_dist(1, [])
        assert len(d) == 0

    def test_recombination_dist_to_prob(self):
        p_min = 0
        p_max = 0.5
        d = np.array([0, 1e-10, 1e10])
        p = tsinfer.inference.Matcher.recombination_dist_to_prob(d)
        assert p[0] == p_min
        assert np.allclose(p, [p_min, p_min, p_max])

    def test_mismatch_ratio_to_prob_low_dist(self):
        """
        For small distances & low ratios, mismatch prob should be ~ ratio * recomb prob
        """
        dist_cM = np.array([0, 1e-5, 0.01])  # approximation excellent up to 0.01 cM
        d = dist_cM / 100
        for ratio in [0, 1e-2, 1]:
            for num_alleles in [2, 4, 6]:
                r_prob = tsinfer.inference.Matcher.recombination_dist_to_prob(d)
                m_prob = tsinfer.inference.Matcher.mismatch_ratio_to_prob(
                    ratio, d, num_alleles
                )
                assert np.allclose(r_prob * ratio, m_prob, rtol=1e-4)

    def test_mismatch_ratio_to_prob_ratio_1(self):
        """
        mismatch probs == recomb_probs when mismatch_ratio=1 & num_alleles=2
        """
        dist_cM = np.array([0, 1e-5, 1, 100, 10000])
        d = dist_cM / 100
        r_prob = tsinfer.inference.Matcher.recombination_dist_to_prob(d)
        m_prob = tsinfer.inference.Matcher.mismatch_ratio_to_prob(1, d)
        assert np.all(r_prob == m_prob)

    def test_mismatch_ratio_to_prob_max(self):
        """
        Large distances or high ratios max out at 1/num_alleles
        """
        for dist_cM, ratio in [(100, 1), (0.1, 100)]:
            d = dist_cM / 100
            for num_alleles in [2, 4, 8]:
                m_prob = tsinfer.inference.Matcher.mismatch_ratio_to_prob(
                    ratio, d, num_alleles
                )
                np.isclose(m_prob, 1 / num_alleles, rtol=1e-2)

    def test_recombination_dist_to_prob_known(self):
        dist_vs_prob = np.array(
            [
                # Some values calculated separately in R (formatted to line up nicely)
                # dist expected_Pr
                [0.00, 0.00],
                [1e-5, 1e-5],
                [5e-5, 4.99975e-5],
                [1e-4, 9.999e-5],
                [5e-4, 4.9975e-4],
                [1e-3, 9.99e-4],
                [5e-3, 4.97508e-3],
                [1e-2, 9.90067e-3],
                [5e-2, 4.75813e-2],
                [0.10, 0.0906346],
                [0.50, 0.316060],
                [1.00, 0.432332],
                [5.00, 0.499977],
            ]
        )
        distance = dist_vs_prob[:, 0]
        expected_prob = dist_vs_prob[:, 1]
        assert np.allclose(
            tsinfer.inference.Matcher.recombination_dist_to_prob(distance),
            expected_prob,
        )

    def test_mismatch_ratio_to_prob_known(self):
        dist_vs_prob = np.array(
            [
                # Some values calculated separately in R (formatted to line up nicely)
                #
                # dist p:ratio=0.01 p:ratio=1.00 p:ratio=100
                [0.00, 0.000000000, 0.000000000, 0.0000000000],
                [1e-5, 1.000000e-7, 0.999990e-5, 0.9990007e-3],
                [5e-5, 5.000000e-7, 4.999750e-5, 4.9750831e-3],
                [1e-4, 1.000000e-6, 0.999900e-4, 0.9900663e-2],
                [5e-4, 4.999975e-6, 4.997501e-4, 4.7581291e-2],
                [1e-3, 0.999990e-5, 0.999001e-3, 0.9063462e-1],
                [5e-3, 4.999750e-5, 4.975083e-3, 0.3160603],
                [1e-2, 0.999900e-4, 0.990066e-2, 0.4323324],
                [5e-2, 4.997501e-4, 4.758129e-2, 0.4999773],
                [0.10, 0.999001e-3, 0.090634623, 0.5],
                [0.50, 4.975083e-3, 0.316060279, 0.5],
                [1.00, 0.990066e-2, 0.432332358, 0.5],
                [5.00, 4.758129e-2, 0.499977300, 0.5],
            ]
        )
        expected_prob = {}
        distance = dist_vs_prob[:, 0]
        expected_prob[0.01] = dist_vs_prob[:, 1]
        expected_prob[1.00] = dist_vs_prob[:, 2]
        expected_prob[100] = dist_vs_prob[:, 3]
        for k in expected_prob.keys():
            assert np.allclose(
                tsinfer.inference.Matcher.mismatch_ratio_to_prob(k, distance),
                expected_prob[k],
            )


class TestMatchSamples:
    """
    Test specific features of the match_samples stage
    """

    @pytest.mark.skip(
        reason="Bug to be fixed: Issue https://github.com/tskit-dev/tsinfer/issues/493"
    )
    def test_partial_samples(self):
        ts = msprime.simulate(
            10, mutation_rate=2, recombination_rate=2, random_seed=233
        )
        for tree_seq in [ts, eval_util.strip_singletons(ts)]:
            sd = tsinfer.SampleData.from_tree_sequence(tree_seq, use_sites_time=False)
            ts1 = tsinfer.infer(sd)
            ancestors = tsinfer.generate_ancestors(sd)
            anc_ts = tsinfer.match_ancestors(sd, ancestors)
            # test indices missing from start, end, and in the middle
            for samples in (np.arange(8), np.arange(2, 10), np.arange(5) * 2):
                ts2 = tsinfer.match_samples(sd, anc_ts, indexes=samples).simplify()
                assert ts1.simplify(samples).equals(ts2, ignore_provenance=True)

    def test_partial_bad_indexes(self):
        sd = tsinfer.SampleData.from_tree_sequence(
            msprime.simulate(
                10, mutation_rate=2, recombination_rate=2, random_seed=233
            ),
            use_sites_time=False,
        )
        ancestors = tsinfer.generate_ancestors(sd)
        a_ts = tsinfer.match_ancestors(sd, ancestors)
        for bad_samples in [[], [-1, 0], [0, 10]]:
            with pytest.raises(ValueError):
                tsinfer.match_samples(sd, a_ts, indexes=bad_samples)


class AlgorithmsExactlyEqualMixin:
    """
    For small example tree sequences, check that the Python and C implementations
    return precisely the same tree sequence when fed with perfect mutations.
    """

    path_compression_enabled = True
    precision = None

    def infer(self, ts, engine, path_compression=False, precision=None):
        sample_data = tsinfer.SampleData(sequence_length=ts.sequence_length)
        for v in ts.variants():
            sample_data.add_site(v.site.position, v.genotypes, v.alleles)
        sample_data.finalise()

        ancestor_data = tsinfer.AncestorData(sample_data)
        tsinfer.build_simulated_ancestors(sample_data, ancestor_data, ts)
        ancestor_data.finalise()
        ancestors_ts = tsinfer.match_ancestors(
            sample_data,
            ancestor_data,
            engine=engine,
            path_compression=path_compression,
            precision=precision,
            extended_checks=True,
        )
        inferred_ts = tsinfer.match_samples(
            sample_data,
            ancestors_ts,
            engine=engine,
            simplify=True,
            path_compression=path_compression,
            precision=precision,
            extended_checks=True,
        )
        return inferred_ts

    def verify(self, ts):
        tsp = self.infer(
            ts,
            tsinfer.PY_ENGINE,
            path_compression=self.path_compression_enabled,
            precision=self.precision,
        )
        tsc = self.infer(
            ts,
            tsinfer.C_ENGINE,
            path_compression=self.path_compression_enabled,
            precision=self.precision,
        )
        assert ts.num_sites == tsp.num_sites
        assert ts.num_sites == tsc.num_sites
        assert tsc.num_samples == tsp.num_samples
        tables_p = tsp.dump_tables()
        tables_c = tsc.dump_tables()
        assert tables_p.nodes == tables_c.nodes
        assert tables_p.edges == tables_c.edges
        assert tables_p.sites == tables_c.sites
        assert tables_p.mutations == tables_c.mutations

    def test_single_tree(self):
        for seed in range(10):
            ts = msprime.simulate(10, random_seed=seed + 1)
            ts = tsinfer.insert_perfect_mutations(ts)
            self.verify(ts)

    def test_three_samples(self):
        for seed in range(10):
            ts = msprime.simulate(
                3, recombination_rate=1, random_seed=seed + 1, model="smc_prime"
            )
            ts = tsinfer.insert_perfect_mutations(ts)
            self.verify(ts)

    def test_four_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                4,
                recombination_rate=0.1,
                random_seed=seed + 1,
                length=10,
                model="smc_prime",
            )
            ts = tsinfer.insert_perfect_mutations(ts, delta=1 / 8192)
            self.verify(ts)

    def test_five_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                5,
                recombination_rate=0.1,
                random_seed=seed + 100,
                length=10,
                model="smc_prime",
            )
            ts = tsinfer.insert_perfect_mutations(ts, delta=1 / 8192)
            self.verify(ts)

    def test_ten_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                10,
                recombination_rate=0.1,
                random_seed=seed + 200,
                length=10,
                model="smc_prime",
            )
            ts = tsinfer.insert_perfect_mutations(ts, delta=1 / 8192)
            self.verify(ts)

    @pytest.mark.slow
    def test_twenty_samples(self):
        for seed in range(5):
            ts = msprime.simulate(
                20,
                recombination_rate=0.1,
                random_seed=seed + 500,
                length=10,
                model="smc_prime",
            )
            ts = tsinfer.insert_perfect_mutations(ts, delta=1 / 8192)
            self.verify(ts)


class TestAlgorithmsExactlyEqualNoPathCompression(AlgorithmsExactlyEqualMixin):
    path_compression_enabled = False


class TestAlgorithmsExactlyEqualPathCompression(AlgorithmsExactlyEqualMixin):
    path_compression_enabled = True


class TestAlgorithmsExactlyEqualPrecision24(AlgorithmsExactlyEqualMixin):
    precision = 24


class TestAlgorithmsExactlyEqualPrecision6(AlgorithmsExactlyEqualMixin):
    precision = 6


class TestAlgorithmsExactlyEqualPrecision1(AlgorithmsExactlyEqualMixin):
    precision = 1


class TestAlgorithmsExactlyEqualPrecision0(AlgorithmsExactlyEqualMixin):
    precision = 0


class TestAlgorithmDebugOutput:
    """
    Test routines used to debug output from the algorithm
    """

    def sample_example(self, n_samples, n_sites):
        G, positions = get_random_data_example(n_samples, n_sites)
        sample_data = tsinfer.SampleData(sequence_length=n_sites)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        return sample_data

    def test_ancestor_builder_print_state(self):
        n_samples = 20
        n_sites = 50
        sample_data = self.sample_example(n_samples, n_sites)
        ancestor_builder = tsinfer.algorithm.AncestorBuilder(n_samples, n_sites)
        for variant in sample_data.variants():
            ancestor_builder.add_site(variant.site.time, variant.genotypes)
        with mock.patch("sys.stdout", new=io.StringIO()) as mock_output:
            ancestor_builder.print_state()
            # Simply check some text is output
            assert isinstance(mock_output.getvalue(), str)
            assert len(mock_output.getvalue()) > 0

    def test_ancestor_matcher_print_state(self):
        sample_data = self.sample_example(20, 50)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        matcher_container = tsinfer.AncestorMatcher(
            sample_data, ancestor_data, engine=tsinfer.PY_ENGINE
        )
        matcher_container.match_ancestors()
        with mock.patch("sys.stdout", new=io.StringIO()) as mockOutput:
            matcher_container.matcher[0].print_state()
            # Simply check some text is output
            assert isinstance(mockOutput.getvalue(), str)
            assert len(mockOutput.getvalue()) > 0

    def test_treeseq_builder_print_state(self):
        sample_data = self.sample_example(20, 50)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        matcher_container = tsinfer.AncestorMatcher(
            sample_data, ancestor_data, engine=tsinfer.PY_ENGINE
        )
        matcher_container.match_ancestors()
        with mock.patch("sys.stdout", new=io.StringIO()) as mockOutput:
            matcher_container.tree_sequence_builder.print_state()
            # Simply check some text is output
            assert isinstance(mockOutput.getvalue(), str)
            assert len(mockOutput.getvalue()) > 0


class TestPartialAncestorMatching:
    """
    Tests for copying process behaviour when we have partially
    defined ancestors.
    """

    def verify_edges(self, sample_data, ancestor_data, expected_edges):
        def edge_to_tuple(e):
            return (float(e.left), float(e.right), e.parent, e.child)

        for engine in [tsinfer.C_ENGINE, tsinfer.PY_ENGINE]:
            ts = tsinfer.match_ancestors(sample_data, ancestor_data, engine=engine)
            assert sorted([edge_to_tuple(e) for e in expected_edges]) == sorted(
                [edge_to_tuple(e) for e in ts.edges()]
            )

    def test_easy_case(self):
        num_sites = 6
        sample_data = tsinfer.SampleData()
        for j in range(num_sites):
            sample_data.add_site(j, [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData(sample_data)

        ancestor_data.add_ancestor(  # ID 0
            start=0, end=6, focal_sites=[], time=5, haplotype=[0, 0, 0, 0, 0, 0]
        )
        ancestor_data.add_ancestor(  # ID 1
            start=0, end=6, focal_sites=[], time=4, haplotype=[0, 0, 0, 0, 0, 0]
        )
        ancestor_data.add_ancestor(  # ID 2
            start=0,
            end=3,
            focal_sites=[2],
            time=3,
            haplotype=[0, 0, 1, -1, -1, -1][0:3],
        )
        ancestor_data.add_ancestor(  # ID 3
            start=3,
            end=6,
            focal_sites=[4],
            time=2,
            haplotype=[-1, -1, -1, 0, 1, 0][3:6],
        )
        ancestor_data.add_ancestor(  # ID 4
            start=0,
            end=6,
            focal_sites=[0, 1, 3, 5],
            time=1,
            haplotype=[1, 1, 1, 1, 1, 1],
        )
        ancestor_data.finalise()

        expected_edges = [
            tskit.Edge(0, 6, 0, 1),
            tskit.Edge(0, 3, 2, 4),
            tskit.Edge(3, 6, 3, 4),
            tskit.Edge(3, 6, 1, 3),
            tskit.Edge(0, 3, 1, 2),
        ]
        self.verify_edges(sample_data, ancestor_data, expected_edges)

    def test_partial_overlap(self):
        num_sites = 7
        sample_data = tsinfer.SampleData()
        for j in range(num_sites):
            sample_data.add_site(j, [0, 1, 1])
        sample_data.finalise()
        ancestor_data = tsinfer.AncestorData(sample_data)

        ancestor_data.add_ancestor(  # ID 0
            start=0, end=7, focal_sites=[], time=5, haplotype=[0, 0, 0, 0, 0, 0, 0]
        )
        ancestor_data.add_ancestor(  # ID 1
            start=0, end=7, focal_sites=[], time=4, haplotype=[0, 0, 0, 0, 0, 0, 0]
        )
        ancestor_data.add_ancestor(  # ID 2
            start=0,
            end=3,
            focal_sites=[2],
            time=3,
            haplotype=[0, 0, 1, 0, 0, 0, 0][0:3],
        )
        ancestor_data.add_ancestor(  # ID 3
            start=3,
            end=7,
            focal_sites=[4, 6],
            time=2,
            haplotype=[-1, -1, -1, 0, 1, 0, 1][3:7],
        )
        ancestor_data.add_ancestor(  # ID 4
            start=0,
            end=7,
            focal_sites=[0, 1, 3, 5],
            time=1,
            haplotype=[1, 1, 1, 1, 1, 1, 1],
        )
        ancestor_data.finalise()

        expected_edges = [
            tskit.Edge(0, 7, 0, 1),
            tskit.Edge(0, 3, 2, 4),
            tskit.Edge(3, 7, 3, 4),
            tskit.Edge(3, 7, 1, 3),
            tskit.Edge(0, 3, 1, 2),
        ]
        self.verify_edges(sample_data, ancestor_data, expected_edges)

    def test_edge_overlap_bug(self):
        num_sites = 12
        with tsinfer.SampleData() as sample_data:
            for j in range(num_sites):
                sample_data.add_site(j, [0, 1, 1])
        ancestor_data = tsinfer.AncestorData(sample_data)

        ancestor_data.add_ancestor(  # ID 0
            start=0,
            end=12,
            focal_sites=[],
            time=8,
            haplotype=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        ancestor_data.add_ancestor(  # ID 1
            start=0,
            end=12,
            focal_sites=[],
            time=7,
            haplotype=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        ancestor_data.add_ancestor(  # ID 2
            start=0,
            end=4,
            focal_sites=[],
            time=6,
            haplotype=[0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1][0:4],
        )
        ancestor_data.add_ancestor(  # ID 3
            start=4,
            end=12,
            focal_sites=[],
            time=5,
            haplotype=[-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0][4:12],
        )
        ancestor_data.add_ancestor(  # ID 4
            start=8,
            end=12,
            focal_sites=[9, 11],
            time=4,
            haplotype=[-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, 1][8:12],
        )
        ancestor_data.add_ancestor(  # ID 5
            start=4,
            end=8,
            focal_sites=[5, 7],
            time=3,
            haplotype=[-1, -1, -1, -1, 0, 1, 0, 1, -1, -1, -1, -1][4:8],
        )
        ancestor_data.add_ancestor(  # ID 6
            start=0,
            end=4,
            focal_sites=[1, 3],
            time=2,
            haplotype=[0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1][0:4],
        )
        ancestor_data.add_ancestor(  # ID 7
            start=0,
            end=12,
            focal_sites=[0, 2, 4, 6, 8, 10],
            time=1,
            haplotype=[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
        )
        ancestor_data.finalise()

        expected_edges = [
            tskit.Edge(0, 12, 0, 1),
            tskit.Edge(0, 4, 1, 2),
            tskit.Edge(4, 12, 1, 3),
            tskit.Edge(8, 12, 1, 4),
            tskit.Edge(4, 8, 1, 5),
            tskit.Edge(0, 4, 1, 6),
            tskit.Edge(0, 5, 1, 7),
            tskit.Edge(5, 8, 5, 7),
            tskit.Edge(8, 12, 1, 7),
        ]
        self.verify_edges(sample_data, ancestor_data, expected_edges)


class TestBadEngine:
    """
    Check that we catch bad engines parameters.
    """

    bad_engines = ["CCCC", "c", "p", "Py", "python"]

    def get_example(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=3)
        return tsinfer.SampleData.from_tree_sequence(ts)

    def test_infer(self):
        sample_data = self.get_example()
        for bad_engine in self.bad_engines:
            with pytest.raises(ValueError):
                tsinfer.infer(sample_data, engine=bad_engine)

    def test_generate_ancestors(self):
        sample_data = self.get_example()
        for bad_engine in self.bad_engines:
            with pytest.raises(ValueError):
                tsinfer.generate_ancestors(sample_data, engine=bad_engine)

    def test_match_ancestors(self):
        sample_data = self.get_example()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        for bad_engine in self.bad_engines:
            with pytest.raises(ValueError):
                tsinfer.match_ancestors(
                    sample_data,
                    ancestor_data,
                    engine=bad_engine,
                )

    def test_match_samples(self):
        sample_data = self.get_example()
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        for bad_engine in self.bad_engines:
            with pytest.raises(ValueError):
                tsinfer.match_samples(sample_data, ancestors_ts, engine=bad_engine)


class TestWrongAncestorsTreeSequence:
    """
    Tests covering what happens when we provide an incorrect tree sequence
    as the ancestrors_ts.
    Initial issue: https://github.com/tskit-dev/tsinfer/issues/53
    """

    def test_wrong_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        inferred_ts = tsinfer.infer(sample_data)
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, inferred_ts)
        # tsinfer.match_samples(sample_data, inferred_ts)

    def test_original_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        # This raises an error because we have non-inference sites in the
        # original ts.
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, sim)

    def test_zero_node_times(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        tables = ancestors_ts.dump_tables()
        tables.nodes.add_row(time=0, flags=0)
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, tables.tree_sequence())

    def test_different_ancestors_ts_match_samples(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)

        sim = msprime.simulate(sample_size=6, random_seed=2, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, ancestors_ts)

    def test_bad_edge_position(self):
        sim = msprime.simulate(sample_size=6, random_seed=1, mutation_rate=6)
        sample_data = tsinfer.SampleData.from_tree_sequence(sim)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)

        tables = ancestors_ts.dump_tables()
        # To make things easy, add a new node we can refer to without mucking
        # up the existing topology
        node = tables.nodes.add_row(flags=1)
        tables.edges.add_row(0.5, 1.0, node - 1, node)
        tables.sort()
        bad_ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, bad_ts)

        # Same thing for the right coordinate.
        tables = ancestors_ts.dump_tables()
        node = tables.nodes.add_row(flags=1)
        tables.edges.add_row(0, 0.5, node - 1, node)
        tables.sort()
        bad_ts = tables.tree_sequence()
        with pytest.raises(ValueError):
            tsinfer.match_samples(sample_data, bad_ts)


class TestSimplify:
    """
    Check that the simplify argument to infer is correctly invoked.
    """

    def verify(self, ts):
        n = ts.num_samples
        assert ts.num_sites > 2
        sd = tsinfer.SampleData.from_tree_sequence(ts)
        ts1 = tsinfer.infer(sd, simplify=True)
        # When simplify is true the samples should be zero to n.
        assert list(ts1.samples()) == list(range(n))
        for tree in ts1.trees():
            assert tree.num_samples() == len(list(tree.leaves()))

        # When simplify is true and there is no path compression,
        # the samples should be zero to N - n up to n
        ts2 = tsinfer.infer(sd, simplify=False, path_compression=False)
        assert list(ts2.samples()) == list(range(ts2.num_nodes - n, ts2.num_nodes))

        # Check that we're calling simplify with the correct arguments.
        ts2 = tsinfer.infer(sd, simplify=False).simplify(keep_unary=True)
        assert ts1.equals(ts2, ignore_provenance=True)

    def test_single_tree(self):
        ts = msprime.simulate(5, random_seed=1, mutation_rate=2)
        self.verify(ts)

    def test_many_trees(self):
        ts = msprime.simulate(5, random_seed=1, recombination_rate=2, mutation_rate=2)
        assert ts.num_trees > 2
        self.verify(ts)


def get_default_inference_sites(sample_data):
    """
    Returns the site positions that would be used for inference by
    default.
    """
    inference_sites = []
    for var in sample_data.variants():
        counts = tsinfer.allele_counts(var.genotypes)
        assert len(var.site.alleles) == 2
        if counts.derived > 1 and counts.derived < counts.known:
            inference_sites.append(var.site.position)
    return inference_sites


class TestMatchSiteSubsets:
    """
    Tests that we can successfully run the algorithm on data in which we have
    a subset of the original sites.
    """

    def subset_sites(self, ts, position):
        """
        Return a copy of the specified tree sequence with sites reduced to those
        with positions in the specified list.
        """
        tables = ts.dump_tables()
        lookup = set(position)
        tables.sites.clear()
        tables.mutations.clear()
        for site in ts.sites():
            if site.position in lookup:
                site_id = tables.sites.add_row(
                    site.position,
                    ancestral_state=site.ancestral_state,
                    metadata=site.metadata,
                )
                for mutation in site.mutations:
                    tables.mutations.add_row(
                        site_id,
                        node=mutation.node,
                        parent=mutation.parent,
                        derived_state=mutation.derived_state,
                        metadata=mutation.metadata,
                    )
        assert np.array_equal(tables.sites.position, position)
        return tables.tree_sequence()

    def verify(self, sample_data, position_subset):
        full_ts = tsinfer.infer(sample_data)
        subset_ts = self.subset_sites(full_ts, position_subset)
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestor_data)
        subset_ancestors_ts = tsinfer.minimise(
            self.subset_sites(ancestors_ts, position_subset)
        )
        subset_ancestors_ts = subset_ancestors_ts.simplify()
        subset_sample_data = tsinfer.SampleData.from_tree_sequence(subset_ts)
        output_ts = tsinfer.match_samples(subset_sample_data, subset_ancestors_ts)
        assert np.array_equal(output_ts.genotype_matrix(), subset_ts.genotype_matrix())

    def test_simple_case(self):
        ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=3)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        position = get_default_inference_sites(sample_data)
        self.verify(sample_data, position[:][::2])

    def test_one_sites(self):
        ts = msprime.simulate(15, mutation_rate=2, recombination_rate=2, random_seed=3)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        position = get_default_inference_sites(sample_data)
        self.verify(sample_data, position[:1])

    def test_no_recombination(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=4)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        position = get_default_inference_sites(sample_data)
        self.verify(sample_data, position[:][1::2])

    def test_random_data(self):
        n = 25
        m = 50
        G, positions = get_random_data_example(n, m)
        sample_data = tsinfer.SampleData(sequence_length=m)
        for genotypes, position in zip(G, positions):
            sample_data.add_site(position, genotypes)
        sample_data.finalise()
        position = get_default_inference_sites(sample_data)
        self.verify(sample_data, position[:][::2])


class PathCompressionMixin:
    """
    Common utilities for testing a tree sequence with path compression.
    """

    def verify_tree_sequence(self, ts):
        pc_nodes = [node for node in ts.nodes() if tsinfer.is_pc_ancestor(node.flags)]
        assert len(pc_nodes) > 0
        for node in pc_nodes:
            # print("Synthetic node", node)
            parent_edges = [edge for edge in ts.edges() if edge.parent == node.id]
            child_edges = [edge for edge in ts.edges() if edge.child == node.id]
            assert len(parent_edges) > 1
            assert len(child_edges) > 1
            child_edges.sort(key=lambda e: e.left)
            # print("parent edges")
            # for edge in parent_edges:
            #     print("\t", edge)
            # print("child edges")
            # Child edges should always be contiguous
            last_right = child_edges[0].left
            for edge in child_edges:
                # print("\t", edge)
                assert last_right == edge.left
                last_right = edge.right
            left = child_edges[0].left
            right = child_edges[-1].right
            original_matches = [
                e for e in parent_edges if e.left == left and e.right == right
            ]
            # We must have at least two initial edges that exactly span the
            # pc interval.
            assert len(original_matches) > 1

    def test_simple_case(self):
        ts = msprime.simulate(55, mutation_rate=5, random_seed=4, recombination_rate=8)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)

    def test_simulation_with_error(self):
        ts = msprime.simulate(
            50, mutation_rate=10, random_seed=4, recombination_rate=15
        )
        ts = eval_util.insert_errors(ts, 0.2, seed=32)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        self.verify(sample_data)

    def test_small_random_data(self):
        n = 25
        m = 20
        G, positions = get_random_data_example(n, m)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)

    def test_large_random_data(self):
        n = 100
        m = 30
        G, positions = get_random_data_example(n, m)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)


class PathCompressionAncestorsMixin(PathCompressionMixin):
    """
    Tests for the results of path compression on an ancestors tree sequence.
    """

    def verify(self, sample_data):
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        ts = tsinfer.match_ancestors(
            sample_data, ancestor_data, engine=self.engine, extended_checks=True
        )
        self.verify_tree_sequence(ts)


class TestPathCompressionAncestorsPyEngine(PathCompressionAncestorsMixin):
    engine = tsinfer.PY_ENGINE


class TestPathCompressionAncestorsCEngine(PathCompressionAncestorsMixin):
    engine = tsinfer.C_ENGINE

    def test_c_engine_fail_example(self):
        # Reproduce a failure that occured under the C engine.
        ts = msprime.simulate(
            20,
            Ne=10 ** 4,
            length=0.25 * 10 ** 6,
            recombination_rate=1e-8,
            mutation_rate=1e-8,
            random_seed=4,
        )
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        self.verify(sample_data)


class PathCompressionSamplesMixin(PathCompressionMixin):
    """
    Tests for the results of path compression just on samples.
    """

    def verify(self, sample_data):
        ancestor_data = tsinfer.generate_ancestors(sample_data)
        # Turn off path compression in the ancestors to make this as difficult
        # as possible.
        ancestors_ts = tsinfer.match_ancestors(
            sample_data, ancestor_data, path_compression=False
        )
        ts = tsinfer.match_samples(
            sample_data,
            ancestors_ts,
            path_compression=True,
            engine=self.engine,
            extended_checks=True,
        )
        self.verify_tree_sequence(ts)


class TestPathCompressionSamplesPyEngine(PathCompressionSamplesMixin):
    engine = tsinfer.PY_ENGINE


class TestPathCompressionSamplesCEngine(PathCompressionSamplesMixin):
    engine = tsinfer.C_ENGINE


class PathCompressionFullStackMixin(PathCompressionMixin):
    """
    Tests for the results of path compression just on samples.
    """

    def verify(self, sample_data):
        # We have to turn off simplify because it'll sometimes remove chunks
        # of pc ancestors, breaking out continguity requirements.
        ts = tsinfer.infer(
            sample_data, path_compression=True, engine=self.engine, simplify=False
        )
        self.verify_tree_sequence(ts)


class TestPathCompressionFullStackPyEngine(PathCompressionFullStackMixin):
    engine = tsinfer.PY_ENGINE


class TestPathCompressionFullStackCEngine(PathCompressionFullStackMixin):
    engine = tsinfer.C_ENGINE


class TestFlags:
    """
    Tests if we can set and detect the pc node flag correctly.
    """

    PC_BIT_POSITION = 16
    SRB_BIT_POSITION = 17

    def test_is_pc_ancestor(self):
        assert not tsinfer.is_pc_ancestor(0)
        assert not tsinfer.is_pc_ancestor(1)
        assert tsinfer.is_pc_ancestor(tsinfer.NODE_IS_PC_ANCESTOR)
        for bit in range(32):
            flags = 1 << bit
            if bit == self.PC_BIT_POSITION:
                assert tsinfer.is_pc_ancestor(flags)
            else:
                assert not tsinfer.is_pc_ancestor(flags)
        flags = tsinfer.NODE_IS_PC_ANCESTOR
        for bit in range(32):
            flags |= 1 << bit
            assert tsinfer.is_pc_ancestor(flags)
        flags = 0
        for bit in range(32):
            if bit != self.PC_BIT_POSITION:
                flags |= 1 << bit
            assert not tsinfer.is_pc_ancestor(flags)

    def test_count_pc_ancestors(self):
        assert tsinfer.count_pc_ancestors([0]) == 0
        assert tsinfer.count_pc_ancestors([tsinfer.NODE_IS_PC_ANCESTOR]) == 1
        assert tsinfer.count_pc_ancestors([0, 0]) == 0
        assert tsinfer.count_pc_ancestors([0, tsinfer.NODE_IS_PC_ANCESTOR]) == 1
        assert (
            tsinfer.count_pc_ancestors(
                [tsinfer.NODE_IS_PC_ANCESTOR, tsinfer.NODE_IS_PC_ANCESTOR]
            )
            == 2
        )
        assert tsinfer.count_pc_ancestors([1, tsinfer.NODE_IS_PC_ANCESTOR]) == 1
        assert (
            tsinfer.count_pc_ancestors(
                [1 | tsinfer.NODE_IS_PC_ANCESTOR, 1 | tsinfer.NODE_IS_PC_ANCESTOR]
            )
            == 2
        )

    def test_count_srb_ancestors_random(self):
        np.random.seed(42)
        flags = np.random.randint(0, high=2 ** 32, size=100, dtype=np.uint32)
        count = sum(map(tsinfer.is_srb_ancestor, flags))
        assert count == tsinfer.count_srb_ancestors(flags)

    def test_is_srb_ancestor(self):
        assert not tsinfer.is_srb_ancestor(0)
        assert not tsinfer.is_srb_ancestor(1)
        assert tsinfer.is_srb_ancestor(tsinfer.NODE_IS_SRB_ANCESTOR)
        for bit in range(32):
            flags = 1 << bit
            if bit == self.SRB_BIT_POSITION:
                assert tsinfer.is_srb_ancestor(flags)
            else:
                assert not tsinfer.is_srb_ancestor(flags)
        flags = tsinfer.NODE_IS_SRB_ANCESTOR
        for bit in range(32):
            flags |= 1 << bit
            assert tsinfer.is_srb_ancestor(flags)
        flags = 0
        for bit in range(32):
            if bit != self.SRB_BIT_POSITION:
                flags |= 1 << bit
            assert not tsinfer.is_srb_ancestor(flags)

    def test_count_srb_ancestors(self):
        assert tsinfer.count_srb_ancestors([0]) == 0
        assert tsinfer.count_srb_ancestors([tsinfer.NODE_IS_SRB_ANCESTOR]) == 1
        assert tsinfer.count_srb_ancestors([0, 0]) == 0
        assert tsinfer.count_srb_ancestors([0, tsinfer.NODE_IS_SRB_ANCESTOR]) == 1
        assert (
            tsinfer.count_srb_ancestors(
                [tsinfer.NODE_IS_SRB_ANCESTOR, tsinfer.NODE_IS_SRB_ANCESTOR]
            )
            == 2
        )
        assert tsinfer.count_srb_ancestors([1, tsinfer.NODE_IS_SRB_ANCESTOR]) == 1
        assert (
            tsinfer.count_srb_ancestors(
                [1 | tsinfer.NODE_IS_SRB_ANCESTOR, 1 | tsinfer.NODE_IS_SRB_ANCESTOR]
            )
            == 2
        )

    def test_count_pc_ancestors_random(self):
        np.random.seed(42)
        flags = np.random.randint(0, high=2 ** 32, size=100, dtype=np.uint32)
        count = sum(map(tsinfer.is_pc_ancestor, flags))
        assert count == tsinfer.count_pc_ancestors(flags)


class TestBugExamples:
    """
    Run tests on some examples that provoked bugs.
    """

    @pytest.mark.skip("Need to update example files")
    def test_path_compression_parent_child_identical_times(self):
        # This provoked a bug in which we created a pc ancestor
        # with the same time as its child, creating an invalid topology.
        sample_data = tsinfer.load("tests/data/bugs/invalid_pc_ancestor_time.samples")
        ts = tsinfer.infer(sample_data)
        for var, (_, genotypes) in zip(ts.variants(), sample_data.genotypes()):
            assert np.array_equal(var.genotypes, genotypes)

    @pytest.mark.skip("Need to solve https://github.com/tskit-dev/tsinfer/issues/210")
    def test_path_compression_parent_child_small_times(self):
        # If we allow the user to set variant times, they might create a pair of
        # parent & child ancestors that are separated by < PC_ANCESTOR_INCREMENT
        PC_ANCESTOR_INCREMENT = 1.0 / 65536  # From tree_sequence_builder.c
        ts = msprime.simulate(
            10, recombination_rate=10, mutation_rate=5, random_seed=123
        )
        # adjust the intermediate node times to squish them all together
        tables = ts.dump_tables()
        times = tables.nodes.time[:]
        adjust_times = np.logical_and(times != np.min(times), times != np.max(times))
        min_time = np.min(times[adjust_times])
        time_order = np.argsort(times[adjust_times])
        # Set < PC_ANCESTOR_INCREMENT apart
        times[adjust_times] = min_time + (time_order + 1) * PC_ANCESTOR_INCREMENT / 2
        tables.nodes.time = times
        new_ts = tables.tree_sequence()
        sample_data = tsinfer.SampleData.from_tree_sequence(new_ts)
        # Next line breaks with _tsinfer.LibraryError: Error occured: -5
        # see https://github.com/tskit-dev/tsinfer/issues/210
        tsinfer.infer(sample_data)


class TestVerify:
    """
    Checks that we correctly find problems with verify.
    """

    def test_nominal_case(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 0
        samples = tsinfer.SampleData.from_tree_sequence(ts)
        inferred_ts = tsinfer.infer(samples)

        tsinfer.verify(samples, inferred_ts)
        tsinfer.verify(samples, ts)

    def test_bad_num_sites(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData() as samples:
            samples.add_site(0, genotypes=[0, 1])

        with pytest.raises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_num_samples(self):
        n = 5
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData() as samples:
            for j in range(ts.num_sites):
                samples.add_site(j, genotypes=[0, 1])

        with pytest.raises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_sequence_length(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData(sequence_length=100) as samples:
            for j in range(ts.num_sites):
                samples.add_site(j, genotypes=[0, 1])

        with pytest.raises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_site_position(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position + 1e-6, genotypes=var.genotypes
                )

        with pytest.raises(ValueError):
            tsinfer.verify(samples, ts)

    def test_bad_ancestral_allele(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position,
                    alleles=["1", "0"],
                    genotypes=var.genotypes,
                )

        with pytest.raises(ValueError, match="Ancestral"):
            tsinfer.verify(samples, ts)

    def test_bad_alleles(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position,
                    alleles=["0", "T"],
                    genotypes=var.genotypes,
                )

        with pytest.raises(ValueError, match="Alleles"):
            tsinfer.verify(samples, ts)

    def test_bad_genotypes(self):
        n = 2
        ts = msprime.simulate(n, mutation_rate=5, random_seed=1)
        assert ts.num_sites > 1
        with tsinfer.SampleData(sequence_length=ts.sequence_length) as samples:
            for var in ts.variants():
                samples.add_site(
                    position=var.site.position, alleles=var.alleles, genotypes=[0, 0]
                )

        with pytest.raises(ValueError, match="Genotypes"):
            tsinfer.verify(samples, ts)

    def test_monomorphic_sites(self):
        ts = msprime.sim_ancestry(3, ploidy=1, sequence_length=10, random_seed=123)
        # A finite sites mutation model can create monomorphic sites by reversion etc.
        ts = msprime.sim_mutations(ts, rate=0.5, model="binary", random_seed=1)
        sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        is_monomorphic = np.all(np.diff(sd.sites_genotypes[:], axis=1) == 0, axis=1)
        assert len(is_monomorphic) == sd.num_sites
        assert np.any(is_monomorphic)
        ts_inf = tsinfer.infer(sd)
        tsinfer.verify(sd, ts_inf)

    def test_alternative_allele_encodings(self):
        ts = msprime.sim_ancestry(3, ploidy=1, sequence_length=10, random_seed=123)
        ts = msprime.sim_mutations(ts, rate=0.2, random_seed=1)
        sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        ts_inf = tsinfer.infer(sd)
        has_alt_order = False
        for v1, v2 in zip(sd.variants(), ts_inf.variants()):
            if set(v1.alleles) == set(v2.alleles) and v1.alleles != v2.alleles:
                has_alt_order = True
        assert has_alt_order
        tsinfer.verify(sd, ts_inf)


class TestExtractAncestors:
    """
    Checks whether the extract_ancestors function correctly returns an ancestors
    tree sequence with the required properties.
    """

    def verify(self, samples):
        ancestors = tsinfer.generate_ancestors(samples)
        # this ancestors TS has positions mapped only to inference sites
        ancestors_ts_1 = tsinfer.match_ancestors(samples, ancestors)
        ts = tsinfer.match_samples(
            samples, ancestors_ts_1, path_compression=False, simplify=False
        )
        t1 = ancestors_ts_1.dump_tables()

        t2, node_id_map = tsinfer.extract_ancestors(samples, ts)
        assert len(t2.provenances) == len(t1.provenances) + 2
        # Population isn't carried through in ancestors tree sequences
        # for now.
        t2.populations.clear()

        t1.assert_equals(t2, ignore_provenance=True, ignore_ts_metadata=True)

        for node in ts.nodes():
            if node_id_map[node.id] != -1:
                assert node.time == t1.nodes.time[node_id_map[node.id]]

    def test_simple_simulation(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=5, random_seed=2)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_non_zero_one_mutations(self):
        ts = msprime.simulate(10, recombination_rate=5, random_seed=2)
        ts = msprime.mutate(
            ts, rate=2, model=msprime.InfiniteSites(msprime.NUCLEOTIDES), random_seed=15
        )
        assert ts.num_mutations > 0
        self.verify(tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False))

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            G, positions = get_random_data_example(5, 10)
            with tsinfer.SampleData(sequence_length=G.shape[0]) as samples:
                for j in range(G.shape[0]):
                    samples.add_site(positions[j], G[j])
            self.verify(samples)


class TestInsertSrbAncestors:
    """
    Tests that the insert_srb_ancestors function behaves as expected.
    """

    def insert_srb_ancestors(self, samples, ts):

        srb_index = {}
        edges = sorted(ts.edges(), key=lambda e: (e.child, e.left))
        last_edge = edges[0]
        for edge in edges[1:]:
            condition = (
                ts.node(edge.child).is_sample()
                and edge.child == last_edge.child
                and edge.left == last_edge.right
            )
            if condition:
                key = edge.left, last_edge.parent, edge.parent
                if key in srb_index:
                    count, left_bound, right_bound = srb_index[key]
                    srb_index[key] = (
                        count + 1,
                        max(left_bound, last_edge.left),
                        min(right_bound, edge.right),
                    )
                else:
                    srb_index[key] = 1, last_edge.left, edge.right
            last_edge = edge

        tables, node_id_map = tsinfer.extract_ancestors(samples, ts)
        time = tables.nodes.time

        num_extra = 0
        for k, v in srb_index.items():
            if v[0] > 1:
                left, right = v[1:]
                x, pl, pr = k
                pl = node_id_map[pl]
                pr = node_id_map[pr]
                t = min(time[pl], time[pr]) - 1e-4
                node = tables.nodes.add_row(flags=tsinfer.NODE_IS_SRB_ANCESTOR, time=t)
                tables.edges.add_row(left, x, pl, node)
                tables.edges.add_row(x, right, pr, node)
                num_extra += 1

        tables.sort()
        ancestors_ts = tables.tree_sequence()
        return ancestors_ts

    def verify(self, samples):
        ts = tsinfer.infer(samples, simplify=False)
        ancestors_ts_1 = self.insert_srb_ancestors(samples, ts)
        ancestors_ts_2 = tsinfer.insert_srb_ancestors(samples, ts)
        t1 = ancestors_ts_1.dump_tables()
        t2 = ancestors_ts_2.dump_tables()
        t1.provenances.clear()
        t2.provenances.clear()
        assert t1 == t2

        tsinfer.check_ancestors_ts(ancestors_ts_1)
        ts2 = tsinfer.match_samples(samples, ancestors_ts_1)
        tsinfer.verify(samples, ts2)

    def test_simple_simulation(self):
        ts = msprime.simulate(10, mutation_rate=5, recombination_rate=15, random_seed=2)
        self.verify(tsinfer.SampleData.from_tree_sequence(ts))

    def test_random_data_small_examples(self):
        np.random.seed(4)
        num_random_tests = 10
        for _ in range(num_random_tests):
            G, positions = get_random_data_example(5, 10)
            with tsinfer.SampleData(sequence_length=G.shape[0]) as samples:
                for j in range(G.shape[0]):
                    samples.add_site(positions[j], G[j])
            self.verify(samples)

    def test_random_data_large_example(self):
        np.random.seed(5)
        G, positions = get_random_data_example(15, 100)
        with tsinfer.SampleData(sequence_length=G.shape[0]) as samples:
            for j in range(G.shape[0]):
                samples.add_site(positions[j], G[j])
        self.verify(samples)


class TestAugmentedAncestors:
    """
    Tests for augmenting an ancestors tree sequence with samples.
    """

    def verify_augmented_ancestors(
        self, subset, ancestors_ts, augmented_ancestors, path_compression
    ):

        t1 = ancestors_ts.dump_tables()
        t2 = augmented_ancestors.dump_tables()
        k = len(subset)
        m = len(t1.nodes)
        assert np.all(t2.nodes.flags[m : m + k] == tsinfer.NODE_IS_SAMPLE_ANCESTOR)
        assert np.all(t2.nodes.time[m : m + k] == 1)
        for j, node_id in enumerate(subset):
            node = t2.nodes[m + j]
            assert node.flags == tsinfer.NODE_IS_SAMPLE_ANCESTOR
            assert node.time == 1
            metadata = json.loads(node.metadata.decode())
            assert node_id == metadata["sample_data_id"]

        t2.nodes.truncate(len(t1.nodes))
        # Adding and subtracting 1 can lead to small diffs, so we compare
        # the time separately.
        t2.nodes.time -= 1.0
        assert np.allclose(t2.nodes.time, t1.nodes.time)
        t2.nodes.time = t1.nodes.time
        assert t1.nodes == t2.nodes
        if not path_compression:
            # If we have path compression it's possible that some older edges
            # will be compressed out.
            assert set(t2.edges) >= set(t1.edges)
        assert t1.sites == t2.sites
        # We can't compare the mutation tables easily because we can have new
        # mutations happening at sites.
        assert len(t1.mutations) <= len(t2.mutations)
        t2.provenances.truncate(len(t1.provenances))
        assert t1.provenances == t2.provenances
        assert t1.individuals == t2.individuals
        assert t1.populations == t2.populations

    def verify_example(self, subset, samples, ancestors, path_compression):
        ancestors_ts = tsinfer.match_ancestors(
            samples, ancestors, path_compression=path_compression
        )
        augmented_ancestors = tsinfer.augment_ancestors(
            samples, ancestors_ts, subset, path_compression=path_compression
        )
        self.verify_augmented_ancestors(
            subset, ancestors_ts, augmented_ancestors, path_compression
        )

        # Run the inference now
        final_ts = tsinfer.match_samples(samples, augmented_ancestors, simplify=False)
        t1 = ancestors_ts.dump_tables()
        tables = final_ts.tables
        for j, index in enumerate(subset):
            sample_id = final_ts.samples()[index]
            edges = [e for e in final_ts.edges() if e.child == sample_id]
            assert len(edges) == 1
            assert edges[0].left == 0
            assert edges[0].right == final_ts.sequence_length
            parent = edges[0].parent
            original_node = len(t1.nodes) + j
            assert tables.nodes.flags[original_node] == tsinfer.NODE_IS_SAMPLE_ANCESTOR
            # Most of the time the parent is the original node. However, in
            # simple cases it can be somewhere up the tree above it.
            if parent != original_node:
                for tree in final_ts.trees():
                    u = parent
                    while u != tskit.NULL:
                        siblings = tree.children(u)
                        if original_node in siblings:
                            break
                        u = tree.parent(u)
                    assert u != tskit.NULL

    def verify(self, samples):
        ancestors = tsinfer.generate_ancestors(samples)
        n = samples.num_samples
        subsets = [
            [0, 1],
            [n - 2, n - 1],
            [0, n // 2, n - 1],
            range(min(n, 5)),
        ]
        for subset in subsets:
            for path_compression in [True, False]:
                self.verify_example(subset, samples, ancestors, path_compression)

    def test_index_errors(self):
        ts = msprime.simulate(5, mutation_rate=5, random_seed=8, recombination_rate=1)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        ancestors = tsinfer.generate_ancestors(sample_data)
        ancestors_ts = tsinfer.match_ancestors(sample_data, ancestors)
        for bad_subset in [[], [-1], [0, 6]]:
            with pytest.raises(ValueError):
                tsinfer.augment_ancestors(sample_data, ancestors_ts, bad_subset)

    def test_simple_case(self):
        ts = msprime.simulate(55, mutation_rate=5, random_seed=8, recombination_rate=8)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        self.verify(sample_data)

    def test_simulation_with_error(self):
        ts = msprime.simulate(50, mutation_rate=5, random_seed=5, recombination_rate=8)
        ts = eval_util.insert_errors(ts, 0.1, seed=32)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        self.verify(sample_data)

    def test_intermediate_simulation_with_error(self):
        ts = msprime.simulate(10, mutation_rate=5, random_seed=78, recombination_rate=8)
        ts = eval_util.insert_errors(ts, 0.1, seed=32)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        self.verify(sample_data)

    def test_small_simulation_with_error(self):
        ts = msprime.simulate(5, mutation_rate=5, random_seed=5, recombination_rate=8)
        ts = eval_util.insert_errors(ts, 0.1, seed=32)
        sample_data = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        self.verify(sample_data)

    def test_small_random_data(self):
        n = 25
        m = 20
        G, positions = get_random_data_example(n, m, seed=1234)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)

    def test_large_random_data(self):
        n = 100
        m = 30
        G, positions = get_random_data_example(n, m, seed=1234)
        with tsinfer.SampleData(sequence_length=m) as sample_data:
            for genotypes, position in zip(G, positions):
                sample_data.add_site(position, genotypes)
        self.verify(sample_data)


class TestSequentialAugmentedAncestors(TestAugmentedAncestors):
    """
    Test that we can sequentially augment the ancestors.
    """

    def verify_example(self, full_subset, samples, ancestors, path_compression):
        ancestors_ts = tsinfer.match_ancestors(
            samples, ancestors, path_compression=path_compression
        )
        expected_sample_ancestors = 0
        for j in range(1, len(full_subset)):
            subset = full_subset[:j]
            expected_sample_ancestors += len(subset)
            augmented_ancestors = tsinfer.augment_ancestors(
                samples, ancestors_ts, subset, path_compression=path_compression
            )
            self.verify_augmented_ancestors(
                subset, ancestors_ts, augmented_ancestors, path_compression
            )
            # Run the inference now
            final_ts = tsinfer.match_samples(
                samples, augmented_ancestors, simplify=False
            )

            # Make sure metadata has been preserved in the final ts.
            num_sample_ancestors = 0
            for node in final_ts.nodes():
                if node.flags == tsinfer.NODE_IS_SAMPLE_ANCESTOR:
                    metadata = json.loads(node.metadata.decode())
                    assert metadata["sample_data_id"] in subset
                    num_sample_ancestors += 1
            assert expected_sample_ancestors == num_sample_ancestors
            tsinfer.verify(samples, final_ts.simplify())
            ancestors_ts = augmented_ancestors


class TestMismatchAndRecombination:
    """
    Various combinations of recombination & recombination_rate, mismatch & mismatch_rate
    are allowed in match_samples/match_ancestors. Others are meaningless. Test these.
    """

    def test_recombination_rate_with_one_site(self, small_sd_anc_fixture):
        """
        Where there is just one site, there is no recombination probability to use,
        so we default to an arbitrary mismatch rate
        """
        sd, anc = small_sd_anc_fixture
        # Delete all but the first inference site
        del_sites = np.isin(sd.sites_position[:], anc.sites_position[1:])
        sd = sd.subset(sites=np.where(np.logical_not(del_sites))[0])
        anc = tsinfer.generate_ancestors(sd)
        assert anc.num_sites == 1
        tsinfer.infer(sd, recombination_rate=0.1)

    def test_maximal_mismatch_ancestors(self, small_sd_anc_fixture):
        """
        Shouldn't be able to find a path on early part of match_ancestors if
        a mismatch is required (mismatch=1)
        """
        sd, anc = small_sd_anc_fixture
        num_loci = anc.num_sites
        r = np.full(num_loci - 1, 0.01)
        m = np.full(num_loci, 1)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            with pytest.raises(_tsinfer.MatchImpossible):
                tsinfer.match_ancestors(
                    sd,
                    anc,
                    recombination=r,
                    mismatch=m,
                    engine=engine,
                )

    def test_zero_recomb_mutation(self, small_sd_anc_fixture):
        sd, anc = small_sd_anc_fixture
        num_loci = anc.num_sites
        r = np.full(num_loci - 1, 0)
        m = np.full(num_loci, 0)
        for engine in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            with pytest.raises(_tsinfer.MatchImpossible):
                tsinfer.match_ancestors(
                    sd,
                    anc,
                    recombination=r,
                    mismatch=m,
                    engine=engine,
                )

    def test_maximal_mismatch_samples(self, small_sd_anc_fixture):
        """
        Although mismatch of 1 (required mismatch) not possible in match_ancestors,
        it should be in match_samples
        """
        sd, anc = small_sd_anc_fixture
        num_loci = anc.num_sites
        r = np.full(num_loci - 1, 0.01)
        m = np.full(num_loci, 1)
        for e in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            anc_ts = tsinfer.match_ancestors(sd, anc, engine=e)
            tsinfer.match_samples(sd, anc_ts, recombination=r, mismatch=m, engine=e)

    def test_extreme_parameters(self, small_sd_anc_fixture):
        sd, anc = small_sd_anc_fixture
        for e in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            tsinfer.match_ancestors(sd, anc, recombination_rate=1e20, engine=e)
            tsinfer.match_ancestors(
                sd, anc, recombination_rate=1, mismatch_ratio=1e20, engine=e
            )
            tsinfer.match_ancestors(
                sd, anc, recombination_rate=1e20, mismatch_ratio=1e20, engine=e
            )

    def test_recombination_mismatch_combos(self, small_sd_anc_fixture):
        sd, anc = small_sd_anc_fixture
        x = np.full(anc.num_sites, 0.1)
        with pytest.raises(ValueError, match="requires specifying both"):
            tsinfer.match_ancestors(sd, anc, recombination=x[1:])
        with pytest.raises(ValueError, match="requires specifying both"):
            tsinfer.match_ancestors(sd, anc, mismatch=x)
        with pytest.raises(ValueError, match="simultaneously"):
            tsinfer.match_ancestors(
                sd, anc, recombination=x[1:], recombination_rate=0.1, mismatch=x
            )
        with pytest.raises(ValueError, match="simultaneously"):
            tsinfer.match_ancestors(
                sd, anc, recombination=x[1:], mismatch_ratio=1, mismatch=x
            )

    def test_bad_recombination_rate(self, small_sd_fixture):
        sd = small_sd_fixture
        for bad_rate in [np.array([0.1, 0.2]), (0.1, 0.2), []]:
            with pytest.raises(ValueError):
                tsinfer.infer(sd, recombination_rate=bad_rate)

    def test_bad_recombination(self, small_sd_anc_fixture):
        sd, anc = small_sd_anc_fixture
        x = np.full(anc.num_sites, 0.1)
        # Check it normally works: requires array of size 1 less than num_sites
        _ = tsinfer.match_ancestors(sd, anc, mismatch=x, recombination=x[1:])
        for bad in [x, x[2:], []]:
            with pytest.raises(ValueError, match="Bad length"):
                tsinfer.match_ancestors(sd, anc, mismatch=x, recombination=bad)
        bad = x.copy()[1:]
        for bad_val in [1.1, -0.1, np.nan]:
            bad[-1] = bad_val
            with pytest.raises(ValueError, match="recombination.*between 0 & 1"):
                tsinfer.match_ancestors(sd, anc, mismatch=x, recombination=bad)

    def test_mismatch_no_recombination(self, small_sd_anc_fixture):
        sd, anc = small_sd_anc_fixture
        with pytest.raises(ValueError, match="Cannot use mismatch"):
            tsinfer.match_ancestors(sd, anc, mismatch_ratio=1)

    def test_bad_mismatch_ratio(self, small_sd_fixture):
        """Negative or otherwise bad ratios give nonsensical probabilities"""
        sd = small_sd_fixture
        for bad_ratio in [-1e-10, np.nan]:
            with pytest.raises(ValueError, match="mismatch.*between 0 & 1"):
                tsinfer.infer(sd, recombination_rate=0.1, mismatch_ratio=bad_ratio)

    def test_bad_mismatch_ratio_type(self, small_sd_fixture):
        sd = small_sd_fixture
        for bad_ratio in [np.array([0.1, 0.2])]:
            with pytest.raises(ValueError):
                tsinfer.infer(sd, recombination_rate=0.1, mismatch_ratio=bad_ratio)
        for bad_ratio in [(0.1, 0.2), []]:
            with pytest.raises(TypeError):
                tsinfer.infer(sd, recombination_rate=0.1, mismatch_ratio=bad_ratio)

    def test_bad_mismatch(self, small_sd_anc_fixture):
        sd, anc = small_sd_anc_fixture
        x = np.full(anc.num_sites + 1, 0.1)
        # Check it normally works
        _ = tsinfer.match_ancestors(sd, anc, recombination=x[2:], mismatch=x[1:])
        for bad in [x, x[2:], []]:
            with pytest.raises(ValueError, match="Bad length"):
                tsinfer.match_ancestors(sd, anc, recombination=x[2:], mismatch=bad)
        bad = x.copy()[1:]
        for bad_val in [1.1, -0.1, np.nan]:
            bad[-1] = bad_val
            with pytest.raises(ValueError, match="mismatch.*between 0 & 1"):
                tsinfer.match_ancestors(sd, anc, recombination=x[2:], mismatch=bad)

    def test_zero_recombination(self):
        """
        With zero recombination but a positive mismatch value, matching the oldest (root)
        ancestor should always be possible: issue #420
        """
        ts = msprime.simulate(
            10,
            length=1e4,
            Ne=10000,
            mutation_rate=1e-8,
            recombination_rate=1e-8,
            random_seed=50,
        )
        sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        anc = tsinfer.generate_ancestors(sd)
        # Need to be sure that mu is large here or the value associated with the
        # root haplotype can become less than precision, and we therefore
        # fail to find a match.
        m = np.full(anc.num_sites, 1e-2)
        r = np.full(anc.num_sites - 1, 0)  # Ban recombination
        for e in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            anc_ts = tsinfer.match_ancestors(
                sd,
                anc,
                recombination=r,
                mismatch=m,
                engine=e,
                path_compression=False,
                extended_checks=True,
            )
            ts = tsinfer.match_samples(
                sd,
                anc_ts,
                recombination=r,
                mismatch=m,
                engine=e,
                path_compression=False,
                extended_checks=True,
            )
            assert sd.num_sites == ts.num_sites
            for v1, v2 in zip(sd.variants(), ts.variants()):
                assert v1.site.position == v2.site.position
                assert np.all(v1.genotypes == v2.genotypes)

            # If we try this with a small precision value we fail.
            with pytest.raises(_tsinfer.MatchImpossible):
                tsinfer.match_samples(
                    sd,
                    anc_ts,
                    precision=3,
                    recombination=r,
                    mismatch=m,
                    engine=e,
                    path_compression=False,
                    extended_checks=True,
                )


class TestAlgorithmResults:
    """
    Some features of the algorithm have expected outcomes in simple cases. Test these.
    """

    def verify_single_recombination_position(self, positions, G, breakpoint_index):
        with tsinfer.SampleData() as sample_data:
            for pos, genotypes in zip(positions, G):
                sample_data.add_site(pos, genotypes)
        anc = tsinfer.generate_ancestors(sample_data)
        anc_ts = tsinfer.match_ancestors(
            sample_data, anc, recombination_rate=1, mismatch_ratio=1e-10
        )
        ts = tsinfer.match_samples(
            sample_data, anc_ts, recombination_rate=1, mismatch_ratio=1e-10
        )
        assert ts.num_trees == 2
        breakpoint_pos = set(ts.breakpoints()) - {0.0, ts.sequence_length}
        assert breakpoint_pos == {positions[breakpoint_index + 1]}

    def test_recombination_with_dist_high_freq_intermediate(self):
        G = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],  # could break either side of this
                [0, 1, 1, 0],
            ]
        )
        self.verify_single_recombination_position([0.0, 1.1, 2.0], G, 0)
        self.verify_single_recombination_position([0.0, 0.9, 2.0], G, 1)

    def test_recombination_with_dist_low_freq_intermediate(self):
        G = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],  # could break either side of this
                [1, 1, 0, 1],
            ]
        )
        self.verify_single_recombination_position([0.0, 1.1, 2.0], G, 0)
        self.verify_single_recombination_position([0.0, 0.9, 2.0], G, 1)

    @pytest.mark.skip("Should work once the ancestors TS contains non-inference sites")
    def test_recombination_with_dist_noninference_intermediate(self):
        G = np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],  # could break either side of this
                [1, 1, 0, 1],
            ]
        )
        self.verify_single_recombination_position([0.0, 1.1, 2.0], G, 0)
        self.verify_single_recombination_position([0.0, 0.9, 2.0], G, 1)


class TestMissingDataImputed:
    """
    Test that sites with tskit.MISSING_DATA are imputed, using both the PY and C engines
    """

    def test_missing_site(self):
        u = tskit.MISSING_DATA
        sites_by_samples = np.array(
            [
                [u, u, u, u, u],  # Site 0
                [1, 1, 1, 0, 1],  # Site 1
                [1, 0, 1, 1, 0],  # Site 2
                [0, 0, 0, 1, 0],  # Site 3
            ],
            dtype=np.int8,
        )
        expected = sites_by_samples.copy()
        expected[0, :] = [0, 0, 0, 0, 0]
        with tsinfer.SampleData() as sample_data:
            for row in range(sites_by_samples.shape[0]):
                sample_data.add_site(row, sites_by_samples[row, :])
        for e in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts = tsinfer.infer(sample_data, engine=e)
            assert ts.num_trees == 2
            assert np.all(expected == ts.genotype_matrix())

    def test_missing_haplotype(self):
        u = tskit.MISSING_DATA
        sites_by_samples = np.array(
            [
                [u, 1, 1, 1, 0],  # Site 0
                [u, 1, 1, 0, 0],  # Site 1
                [u, 0, 0, 1, 0],  # Site 2
                [u, 0, 1, 1, 0],  # Site 3
            ],
            dtype=np.int8,
        )
        expected = sites_by_samples.copy()
        expected[:, 0] = [0, 0, 0, 0]
        with tsinfer.SampleData() as sample_data:
            for row in range(sites_by_samples.shape[0]):
                sample_data.add_site(row, sites_by_samples[row, :])
        for e in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts = tsinfer.infer(sample_data, engine=e)
            assert ts.num_trees == 2
            assert np.all(expected == ts.genotype_matrix())

    def test_missing_inference_sites(self):
        u = tskit.MISSING_DATA
        sites_by_samples = np.array(
            # fmt:off
            [
                [u, 1, 1, 1, 0],  # Site 0
                [1, 1, u, 0, 0],  # Site 1
            ], dtype=np.int8
            # fmt: on
        )
        expected = sites_by_samples.copy()
        expected[:, 0] = [1, 1]
        expected[:, 2] = [1, 0]
        with tsinfer.SampleData() as sample_data:
            for row in range(sites_by_samples.shape[0]):
                sample_data.add_site(row, sites_by_samples[row, :])
        for e in [tsinfer.PY_ENGINE, tsinfer.C_ENGINE]:
            ts = tsinfer.infer(sample_data, engine=e)
            assert ts.num_trees == 1
            assert np.all(expected == ts.genotype_matrix())


class TestInferenceSites:
    """
    Tests where we expect some sites to be marked for inference and some not
    """

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
        with tsinfer.SampleData() as data:
            for col in range(sites_by_samples.shape[1]):
                data.add_site(col, sites_by_samples[:, col])
        assert data.sequence_length == 9.0
        assert data.num_sites == 9
        ts = tsinfer.infer(data)
        # First site is a entirely missing, second is singleton with missing data =>
        # neither should be marked for inference
        # inference_sites = data.sites_inference[:]
        inf_type = [json.loads(site.metadata)["inference_type"] for site in ts.sites()]
        assert inf_type[0] == tsinfer.INFERENCE_NONE
        assert inf_type[1] == tsinfer.INFERENCE_PARSIMONY
        for t in inf_type[2:]:
            assert t == tsinfer.INFERENCE_FULL

    def test_nan_sites(self):
        # Sites whose time is marked as NaN but are not tskit.UNKNOWN_TIME have
        # a meaningless concept of time and should not be marked for full inference
        with tsinfer.SampleData(1.0) as sample_data:
            sample_data.add_site(0.2, [1, 1, 0])
            sample_data.add_site(0.4, [1, 1, 0], time=np.nan)
            sample_data.add_site(0.6, [1, 1, 0])
        ts = tsinfer.infer(sample_data)
        assert ts.num_trees == 1
        inf_type = [json.loads(site.metadata)["inference_type"] for site in ts.sites()]
        assert inf_type[0] == tsinfer.INFERENCE_FULL
        assert inf_type[1] == tsinfer.INFERENCE_PARSIMONY
        assert inf_type[2] == tsinfer.INFERENCE_FULL


class TestInsertMissingSites:
    def test_bad_length(self):
        # Reduce the length by a tiny bit but keep the sites identical
        L = 2
        ts = msprime.simulate(
            8, length=L, recombination_rate=1, mutation_rate=2, random_seed=123
        )
        epsilon = L / 1e6
        last_site = ts.site(ts.num_sites - 1)
        sample_data = tsinfer.SampleData.from_tree_sequence(
            ts.keep_intervals([[0, last_site.position + epsilon]]).trim()
        )
        with pytest.raises(ValueError, match="sequence length"):
            tsinfer.insert_missing_sites(sample_data, ts)

    def test_bad_samples(self):
        ts = msprime.simulate(
            8, length=2, recombination_rate=1, mutation_rate=2, random_seed=123
        )
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        small_ts = ts.simplify(ts.samples()[1 : ts.num_samples])
        with pytest.raises(ValueError, match="number of samples"):
            tsinfer.insert_missing_sites(sample_data, small_ts)

    def test_simple_insert(self):
        ts = msprime.simulate(8, length=2, recombination_rate=1, random_seed=12)
        mutated_ts = msprime.mutate(ts, rate=1, random_seed=12)
        assert mutated_ts.num_sites > 0
        sample_data = tsinfer.SampleData.from_tree_sequence(mutated_ts)
        mapped_ts = tsinfer.insert_missing_sites(sample_data, ts)
        tsinfer.verify(sample_data, mutated_ts)
        for s in mapped_ts.sites():
            metadata = json.loads(s.metadata)
            assert "inference_type" in metadata
            assert metadata["inference_type"] == tsinfer.INFERENCE_PARSIMONY

    def test_insert_with_map(self):
        ts = msprime.simulate(8, length=1, recombination_rate=1, random_seed=123)
        mutated_ts = msprime.mutate(ts, rate=1, random_seed=123)
        mutated_ts = tsutil.mark_mutation_times_unknown(mutated_ts)
        assert mutated_ts.num_sites > 0
        sample_data = tsinfer.SampleData.from_tree_sequence(mutated_ts)
        reordered_sd = sample_data.subset(individuals=np.arange(ts.num_samples)[::-1])
        assert not sample_data.data_equal(reordered_sd)
        bad_mapped_ts = tsinfer.insert_missing_sites(reordered_sd, ts)
        well_mapped_ts = tsinfer.insert_missing_sites(
            reordered_sd, ts, sample_id_map=np.arange(ts.num_samples)[::-1]
        )
        assert mutated_ts.tables.mutations != bad_mapped_ts.tables.mutations
        assert mutated_ts.tables.mutations == well_mapped_ts.tables.mutations

    def test_mapping_with_inference(self):
        ts = msprime.simulate(
            8, length=2, mutation_rate=1, recombination_rate=1, random_seed=123
        )
        sample_data = tsinfer.SampleData.from_tree_sequence(ts)
        # only use the first half of the ts for inference
        keep = sample_data.sites_position[:] < 1
        assert np.sum(keep) > 0
        assert np.sum(np.logical_not(keep)) > 0
        truncated_sd = sample_data.subset(sites=np.where(keep)[0])
        half_ts = tsinfer.infer(truncated_sd)
        assert half_ts.num_sites < sample_data.num_sites
        full_ts = tsinfer.insert_missing_sites(sample_data, half_ts)
        assert full_ts.num_sites == sample_data.num_sites
        for v1, v2 in zip(sample_data.variants(), full_ts.variants()):
            assert np.array_equal(v1.genotypes, v2.genotypes)
            if v2.site.position >= 1:
                metadata = json.loads(v2.site.metadata)
                assert "inference_type" in metadata
                assert metadata["inference_type"] == tsinfer.INFERENCE_PARSIMONY

    def test_no_inference(self):
        ts = msprime.simulate(8, length=2, recombination_rate=1, random_seed=123)
        mutated_ts = msprime.mutate(ts, rate=1, random_seed=123)
        assert mutated_ts.num_sites > 2
        sd1 = tsinfer.SampleData.from_tree_sequence(mutated_ts)
        # New sample data file with first 3 sites zapped in different ways
        with tsinfer.SampleData(sequence_length=sd1.sequence_length) as sd2:
            for v in sd1.variants():
                genotypes = v.genotypes
                if v.site.id == 0:
                    # All ancestral
                    genotypes[:] = 0
                elif v.site.id == 1:
                    # Some ancestral, some missing
                    genotypes[:] = 0
                    genotypes[0:4] = -1
                elif v.site.id == 2:
                    # All missing
                    genotypes[:] = -1
                sd2.add_site(v.site.position, genotypes, v.alleles)
        full_ts = tsinfer.insert_missing_sites(sd2, ts)
        for v1, v2 in zip(mutated_ts.variants(), full_ts.variants()):
            assert len(v1.site.mutations) == 1
            metadata = json.loads(v2.site.metadata)
            assert "inference_type" in metadata
            if v2.site.id < 3:
                # First 3 sites have been changed so they shouldn't match
                assert not np.array_equal(v1.genotypes, v2.genotypes)
                assert len(v2.site.mutations) == 0
                assert metadata["inference_type"] == tsinfer.INFERENCE_NONE
            else:
                assert np.array_equal(v1.genotypes, v2.genotypes)
                assert len(v2.site.mutations) == 1
                assert metadata["inference_type"] == tsinfer.INFERENCE_PARSIMONY


class TestHistoricalSamples:
    def test_standard_pipeline(self):
        for sample_times in [
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1, 1.5),
            (1.0, 1.5, 0.0, 0.0),
            # (0.0, 0.0, 1.0, 1.5, 15), # see #328#issuecomment-674407970 - point 1
        ]:
            samples = [msprime.Sample(population=0, time=t) for t in sample_times]
            ts = msprime.simulate(
                samples=samples, recombination_rate=1, mutation_rate=10, random_seed=123
            )
            assert ts.num_sites > 0
            sd = tsinfer.SampleData.from_tree_sequence(
                ts, use_sites_time=True, use_individuals_time=True
            )
            generated_ancestors = tsinfer.generate_ancestors(sd)
            all_ancestors = generated_ancestors.insert_proxy_samples(sd)
            ancestors_ts = tsinfer.match_ancestors(sd, all_ancestors)
            inf_ts = tsinfer.match_samples(sd, ancestors_ts, force_sample_times=True)
            for t, u in zip(sample_times, inf_ts.samples()):
                sample = inf_ts.node(u)
                assert sample.time == t

    def test_sample_too_old(self):
        # If we use force_sample_times=True but can't force the sample old enough
        samples = [msprime.Sample(population=0, time=t) for t in (0.0, 0.0, 0.1, 1.5)]
        ts = msprime.simulate(
            samples=samples, recombination_rate=1, mutation_rate=10, random_seed=321
        )
        assert ts.num_sites > 0
        sd = tsinfer.SampleData.from_tree_sequence(
            ts, use_sites_time=True, use_individuals_time=True
        )
        sd = tsinfer.SampleData.from_tree_sequence(
            ts, use_sites_time=True, use_individuals_time=True
        )
        generated_ancestors = tsinfer.generate_ancestors(sd)
        all_ancestors = generated_ancestors.insert_proxy_samples(sd)
        ancestors_ts = tsinfer.match_ancestors(sd, all_ancestors)
        sd_copy = sd.copy()
        time = sd_copy.individuals_time[:]
        time[-1] = 100
        sd_copy.individuals_time[:] = time
        sd_copy.finalise()
        with pytest.raises(ValueError):
            tsinfer.match_samples(sd_copy, ancestors_ts, force_sample_times=True)


class TestAddToSchema:
    def test_is_copy(self):
        schema = tsinfer.permissive_json_schema()
        other = tsinfer.add_to_schema(schema, "name")
        assert schema is not other

    def test_name_collision(self):
        schema = tsinfer.permissive_json_schema()
        schema = tsinfer.add_to_schema(schema, "name")
        with pytest.raises(ValueError):
            tsinfer.add_to_schema(schema, "name")

    def test_defaults(self):
        schema = tsinfer.permissive_json_schema()
        schema = tsinfer.add_to_schema(schema, "name")
        assert schema["properties"]["name"] == {}
        assert schema["required"] == []

    def test_definition(self):
        schema = tsinfer.permissive_json_schema()
        definition = {"type": "number", "description": "sdf"}
        schema = tsinfer.add_to_schema(schema, "name", definition=definition)
        assert schema["properties"]["name"] == definition

    def test_many_keys(self):
        schema = tsinfer.permissive_json_schema()
        name_map = {}
        for j in range(20):
            name = f"x_{j}"
            definition = {"type": "number", "description": f"sdf{j}"}
            name_map[name] = definition
            schema = tsinfer.add_to_schema(schema, name=name, definition=definition)
        assert schema["properties"] == name_map
        assert schema["required"] == []

    def test_many_keys_required(self):
        schema = tsinfer.permissive_json_schema()
        name_map = {}
        names = []
        for j in range(10):
            name = f"x_{j}"
            definition = {"type": "number", "description": f"sdf{j}"}
            name_map[name] = definition
            names.append(name)
            schema = tsinfer.add_to_schema(
                schema, name=name, definition=definition, required=True
            )
        assert schema["properties"] == name_map
        assert schema["required"] == names
