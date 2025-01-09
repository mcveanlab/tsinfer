#
# Copyright (C) 2022 University of Oxford
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
import json
import logging
import sys
import tempfile
import warnings

import msprime
import numcodecs
import numpy as np
import pytest
import sgkit
import tskit
import tsutil
import zarr

import tsinfer
from tsinfer import formats


def ts_to_dataset(ts, chunks=None, samples=None, contigs=None):
    """
    # From https://github.com/sgkit-dev/sgkit/blob/main/sgkit/tests/test_popgen.py#L63
    Convert the specified tskit tree sequence into an sgkit dataset.
    Note this just generates haploids for now - see the note above
    in simulate_ts.
    """
    if samples is None:
        samples = ts.samples()
    tables = ts.dump_tables()
    alleles = []
    genotypes = []
    max_alleles = 0
    for var in ts.variants(samples=samples):
        alleles.append(var.alleles)
        max_alleles = max(max_alleles, len(var.alleles))
        genotypes.append(var.genotypes.astype(np.int8))
    padded_alleles = [
        list(site_alleles) + [""] * (max_alleles - len(site_alleles))
        for site_alleles in alleles
    ]
    alleles = np.array(padded_alleles).astype("S")
    genotypes = np.expand_dims(genotypes, axis=2)

    ds = sgkit.create_genotype_call_dataset(
        variant_contig_names=["1"] if contigs is None else contigs,
        variant_contig=np.zeros(len(tables.sites), dtype=int),
        variant_position=tables.sites.position.astype(int),
        variant_allele=alleles,
        sample_id=np.array([f"tsk_{u}" for u in samples]).astype("U"),
        call_genotype=genotypes,
    )
    if chunks is not None:
        ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    return ds


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_dataset_roundtrip(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    samples = tsinfer.VariantData(zarr_path, "variant_ancestral_allele")
    inf_ts = tsinfer.infer(samples)
    ds = sgkit.load_dataset(zarr_path)

    assert ts.num_individuals == inf_ts.num_individuals == ds.sizes["samples"]
    for ts_ind, sample_id in zip(inf_ts.individuals(), ds["sample_id"].values):
        assert ts_ind.metadata["variant_data_sample_id"] == sample_id

    assert (
        ts.num_samples == inf_ts.num_samples == ds.sizes["samples"] * ds.sizes["ploidy"]
    )
    assert ts.num_sites == inf_ts.num_sites == ds.sizes["variants"]
    assert ts.sequence_length == inf_ts.sequence_length == ds.attrs["contig_lengths"][0]
    for (
        v,
        inf_v,
    ) in zip(ts.variants(), inf_ts.variants()):
        mapping_dict = {a: i for i, a in enumerate(v.alleles)}
        mapping = np.array([mapping_dict[a] for a in inf_v.alleles])
        mapped_genotypes = mapping[inf_v.genotypes]
        assert np.array_equal(v.genotypes, mapped_genotypes)

    # Check that the trees are non-trivial (i.e. the sites have actually been used)
    assert inf_ts.num_trees > 10
    assert inf_ts.num_edges > 200


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_sgkit_individual_metadata_not_clobbered(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    # Load the zarr to add metadata for testing
    zarr_root = zarr.open(zarr_path)
    empty_obj = json.dumps({}).encode()
    indiv_metadata = np.array([empty_obj] * ts.num_individuals, dtype=object)
    indiv_metadata[42] = json.dumps({"variant_data_sample_id": "foobar"}).encode()
    zarr_root.create_dataset(
        "individuals_metadata", data=indiv_metadata, object_codec=numcodecs.VLenBytes()
    )
    zarr_root.attrs["individuals_metadata_schema"] = repr(
        tskit.MetadataSchema.permissive_json()
    )

    samples = tsinfer.VariantData(zarr_path, "variant_ancestral_allele")
    inf_ts = tsinfer.infer(samples)
    ds = sgkit.load_dataset(zarr_path)

    assert ts.num_individuals == inf_ts.num_individuals == ds.sizes["samples"]
    for i, (ts_ind, sample_id) in enumerate(
        zip(inf_ts.individuals(), ds["sample_id"].values)
    ):
        if i != 42:
            assert ts_ind.metadata["variant_data_sample_id"] == sample_id
        else:
            assert ts_ind.metadata["variant_data_sample_id"] == "foobar"


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
@pytest.mark.parametrize("in_mem", [True, False])
def test_variantdata_accessors(tmp_path, in_mem):
    path = None if in_mem else tmp_path
    ts, data = tsutil.make_ts_and_zarr(path, add_optional=True, shuffle_alleles=False)
    vd = tsinfer.VariantData(data, "variant_ancestral_allele", sites_time="sites_time")
    ds = data if in_mem else sgkit.load_dataset(data)

    assert vd.format_name == "tsinfer-variant-data"
    assert vd.format_version == (0, 1)
    assert vd.finalised
    assert vd.sequence_length == ts.sequence_length + 1337
    assert vd.num_sites == ts.num_sites
    assert vd.sites_metadata_schema == ts.tables.sites.metadata_schema.schema
    assert vd.sites_metadata == [site.metadata for site in ts.sites()]
    assert np.array_equal(vd.sites_time, np.arange(ts.num_sites) / ts.num_sites)
    assert np.array_equal(vd.sites_position, ts.tables.sites.position)
    for alleles, v in zip(vd.sites_alleles, ts.variants()):
        # sgkit alleles are padded to be rectangular
        assert np.all(alleles[: len(v.alleles)] == v.alleles)
        assert np.all(alleles[len(v.alleles) :] == "")
    assert np.array_equal(vd.sites_select, np.ones(ts.num_sites, dtype=bool))
    assert np.array_equal(
        vd.sites_ancestral_allele, np.zeros(ts.num_sites, dtype=np.int8)
    )
    assert np.array_equal(vd.sites_genotypes, ts.genotype_matrix())
    assert np.array_equal(
        vd.provenances_timestamp, ["2021-01-01T00:00:00", "2021-01-02T00:00:00"]
    )
    assert vd.provenances_record == [{"foo": 1}, {"foo": 2}]
    assert vd.num_samples == ts.num_samples
    assert np.array_equal(
        vd.samples_individual, np.repeat(np.arange(ts.num_samples // 3), 3)
    )
    assert vd.metadata_schema == tsutil.example_schema("example").schema
    assert vd.metadata == ts.tables.metadata
    assert (
        vd.populations_metadata_schema == ts.tables.populations.metadata_schema.schema
    )
    assert vd.populations_metadata == [pop.metadata for pop in ts.populations()]
    assert vd.num_individuals == ts.num_individuals
    assert np.array_equal(
        vd.individuals_time, np.arange(ts.num_individuals, dtype=np.float32)
    )
    assert (
        vd.individuals_metadata_schema == ts.tables.individuals.metadata_schema.schema
    )
    assert vd.individuals_metadata == [
        {"variant_data_sample_id": sample_id, **ind.metadata}
        for ind, sample_id in zip(ts.individuals(), ds.sample_id[:])
    ]
    assert np.array_equal(
        vd.individuals_location,
        np.tile(np.array([["0", "1"]], dtype="float32"), (ts.num_individuals, 1)),
    )
    assert np.array_equal(
        vd.individuals_population, np.zeros(ts.num_individuals, dtype="int32")
    )
    assert np.array_equal(
        vd.individuals_flags,
        np.random.RandomState(42).randint(
            0, 2_000_000, ts.num_individuals, dtype="int32"
        ),
    )

    # Need to shuffle for the ancestral allele test
    ts, data = tsutil.make_ts_and_zarr(path, add_optional=True)
    vd = tsinfer.VariantData(data, "variant_ancestral_allele")
    for i in range(ts.num_sites):
        assert (
            vd.sites_alleles[i][vd.sites_ancestral_allele[i]]
            == ts.site(i).ancestral_state
        )


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
@pytest.mark.parametrize("in_mem", [True, False])
def test_variantdata_accessors_defaults(tmp_path, in_mem):
    path = None if in_mem else tmp_path
    ts, data = tsutil.make_ts_and_zarr(path)
    vdata = tsinfer.VariantData(data, "variant_ancestral_allele")
    ds = data if in_mem else sgkit.load_dataset(data)

    default_schema = tskit.MetadataSchema.permissive_json().schema
    assert vdata.sequence_length == ts.sequence_length
    assert vdata.sites_metadata_schema == default_schema
    assert vdata.sites_metadata == [{} for _ in range(ts.num_sites)]
    for time in vdata.sites_time:
        assert tskit.is_unknown_time(time)
    assert np.array_equal(vdata.sites_select, np.ones(ts.num_sites, dtype=bool))
    assert np.array_equal(vdata.provenances_timestamp, [])
    assert np.array_equal(vdata.provenances_record, [])
    assert vdata.metadata_schema == default_schema
    assert vdata.metadata == {}
    assert vdata.populations_metadata_schema == default_schema
    assert vdata.populations_metadata == []
    assert vdata.individuals_metadata_schema == default_schema
    assert vdata.individuals_metadata == [
        {"variant_data_sample_id": sample_id} for sample_id in ds.sample_id[:]
    ]
    for time in vdata.individuals_time:
        assert tskit.is_unknown_time(time)
    assert np.array_equal(
        vdata.individuals_location, np.array([[]] * ts.num_individuals, dtype=float)
    )
    assert np.array_equal(
        vdata.individuals_population, np.full(ts.num_individuals, tskit.NULL)
    )
    assert np.array_equal(
        vdata.individuals_flags, np.zeros(ts.num_individuals, dtype=int)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_variantdata_sites_time_default():
    ts, data = tsutil.make_ts_and_zarr()
    vdata = tsinfer.VariantData(data, "variant_ancestral_allele")

    assert (
        np.all(np.isnan(vdata.sites_time)) and vdata.sites_time.size == vdata.num_sites
    )


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on windows")
def test_variantdata_sites_time_array():
    ts, data = tsutil.make_ts_and_zarr()
    sites_time = np.arange(ts.num_sites)
    vdata = tsinfer.VariantData(data, "variant_ancestral_allele", sites_time=sites_time)
    assert np.array_equal(vdata.sites_time, sites_time)
    wrong_length_sites_time = np.arange(ts.num_sites + 1)
    with pytest.raises(
        ValueError,
        match="Sites time array must be the same length as the number of selected sites",
    ):
        tsinfer.VariantData(
            data,
            "variant_ancestral_allele",
            sites_time=wrong_length_sites_time,
        )


def test_simulate_genotype_call_dataset(tmp_path):
    # Test that byte alleles are correctly converted to string
    ts = msprime.sim_ancestry(4, sequence_length=1000, random_seed=123)
    ts = msprime.sim_mutations(ts, rate=2e-3, random_seed=123)
    ds = ts_to_dataset(ts)
    ds.to_zarr(tmp_path, mode="w")
    vdata = tsinfer.VariantData(tmp_path, ds["variant_allele"][:, 0].values.astype(str))
    ts = tsinfer.infer(vdata)
    for v, ds_v, vd_v in zip(ts.variants(), ds.call_genotype, vdata.sites_genotypes):
        assert np.all(v.genotypes == ds_v.values.flatten())
        assert np.all(v.genotypes == vd_v)


def test_simulate_genotype_call_dataset_length(tmp_path):
    # create_genotype_call_dataset does not save contig lengths
    ts = msprime.sim_ancestry(4, sequence_length=1000, random_seed=123)
    ts = msprime.sim_mutations(ts, rate=2e-3, random_seed=123)
    ds = ts_to_dataset(ts)
    assert "contig_length" not in ds
    ds.to_zarr(tmp_path, mode="w")
    vdata = tsinfer.VariantData(tmp_path, ds["variant_allele"][:, 0].values.astype(str))
    assert vdata.sequence_length == ts.sites_position[-1] + 1


class TestMultiContig:
    def make_two_ts_dataset(self, path):
        # split ts into 2; put them as different contigs in the same dataset
        ts = msprime.sim_ancestry(4, sequence_length=1000, random_seed=123)
        ts = msprime.sim_mutations(ts, rate=2e-3, random_seed=123)
        split_at_site = 7
        assert ts.num_sites > 10
        site_break = ts.site(split_at_site).position
        ts1 = ts.keep_intervals([(0, site_break)]).rtrim()
        ts2 = ts.keep_intervals([(site_break, ts.sequence_length)]).ltrim()
        ds = ts_to_dataset(ts, contigs=["chr1", "chr2"])
        ds.update({"variant_ancestral_allele": ds["variant_allele"][:, 0]})
        variant_contig = ds["variant_contig"][:]
        variant_contig[split_at_site:] = 1
        ds.update({"variant_contig": variant_contig})
        variant_position = ds["variant_position"].values
        variant_position[split_at_site:] -= int(site_break)
        ds.update({"variant_position": ds["variant_position"]})
        ds.update(
            {"contig_length": np.array([ts1.sequence_length, ts2.sequence_length])}
        )
        ds.to_zarr(path, mode="w")
        return ts1, ts2

    def test_unmasked(self, tmp_path):
        self.make_two_ts_dataset(tmp_path)
        with pytest.raises(ValueError, match=r'multiple contigs \("chr1", "chr2"\)'):
            tsinfer.VariantData(tmp_path, "variant_ancestral_allele")

    def test_mask(self, tmp_path):
        ts1, ts2 = self.make_two_ts_dataset(tmp_path)
        vdata = tsinfer.VariantData(
            tmp_path,
            "variant_ancestral_allele",
            site_mask=np.array(ts1.num_sites * [True] + ts2.num_sites * [False]),
        )
        assert np.all(ts2.sites_position == vdata.sites_position)
        assert vdata.contig_id == "chr2"
        assert vdata.sequence_length == ts2.sequence_length

    @pytest.mark.parametrize("contig_id", ["chr1", "chr2"])
    def test_contig_id_param(self, contig_id, tmp_path):
        tree_seqs = {}
        tree_seqs["chr1"], tree_seqs["chr2"] = self.make_two_ts_dataset(tmp_path)
        vdata = tsinfer.VariantData(
            tmp_path, "variant_ancestral_allele", contig_id=contig_id
        )
        assert np.all(tree_seqs[contig_id].sites_position == vdata.sites_position)
        assert vdata.contig_id == contig_id
        assert vdata.sequence_length == tree_seqs[contig_id].sequence_length

    def test_contig_id_param_and_mask(self, tmp_path):
        ts1, ts2 = self.make_two_ts_dataset(tmp_path)
        vdata = tsinfer.VariantData(
            tmp_path,
            "variant_ancestral_allele",
            site_mask=np.array(
                (ts1.num_sites + 1) * [True] + (ts2.num_sites - 1) * [False]
            ),
            contig_id="chr2",
        )
        assert np.all(ts2.sites_position[1:] == vdata.sites_position)
        assert vdata.contig_id == "chr2"

    @pytest.mark.parametrize("contig_id", ["chr1", "chr2"])
    def test_contig_length(self, contig_id, tmp_path):
        tree_seqs = {}
        tree_seqs["chr1"], tree_seqs["chr2"] = self.make_two_ts_dataset(tmp_path)
        vdata = tsinfer.VariantData(
            tmp_path, "variant_ancestral_allele", contig_id=contig_id
        )
        assert vdata.sequence_length == tree_seqs[contig_id].sequence_length


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
class TestSgkitMask:
    @pytest.mark.parametrize("sites", [[1, 2, 3, 5, 9, 27], [0]])
    def test_sgkit_variant_mask(self, tmp_path, sites):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.ones_like(ds["variant_position"], dtype=bool)
        for i in sites:
            sites_mask[i] = False
        tsutil.add_array_to_dataset("variant_mask_42", sites_mask, zarr_path)
        vdata = tsinfer.VariantData(
            zarr_path,
            "variant_ancestral_allele",
            site_mask="variant_mask_42",
        )
        assert vdata.num_sites == len(sites)
        assert np.array_equal(vdata.sites_select, ~sites_mask)
        assert np.array_equal(
            vdata.sites_position, ts.tables.sites.position[~sites_mask]
        )
        inf_ts = tsinfer.infer(vdata)
        assert np.array_equal(
            ts.genotype_matrix()[~sites_mask], inf_ts.genotype_matrix()
        )
        assert np.array_equal(
            ts.tables.sites.position[~sites_mask], inf_ts.tables.sites.position
        )
        assert np.array_equal(
            ts.tables.sites.ancestral_state[~sites_mask],
            inf_ts.tables.sites.ancestral_state,
        )
        # TODO - site metadata needs merging not replacing
        # assert [site.metadata for site in ts.sites() if site.id in sites] == [
        #     site.metadata for site in inf_ts.sites()
        # ]

    def test_sgkit_variant_bad_mask_length(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_mask = np.zeros(ds.sizes["variants"] + 1, dtype=int)
        tsutil.add_array_to_dataset("variant_mask_foobar", sites_mask, zarr_path)
        tsinfer.VariantData(zarr_path, "variant_ancestral_allele")
        with pytest.raises(
            ValueError,
            match="Site mask array must be the same length as the number of"
            " unmasked sites",
        ):
            tsinfer.VariantData(
                zarr_path,
                "variant_ancestral_allele",
                site_mask="variant_mask_foobar",
            )

    def test_bad_select_length_at_iterator(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        sites_select = np.zeros(ds.sizes["variants"] + 1, dtype=int)
        from tsinfer.formats import chunk_iterator

        with pytest.raises(
            ValueError, match="Mask must be the same length as the array"
        ):
            for _ in chunk_iterator(ds.call_genotype, select=sites_select):
                pass

    @pytest.mark.parametrize("sample_list", [[1, 2, 3, 5, 9, 27], [0], []])
    def test_sgkit_sample_mask(self, tmp_path, sample_list):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path, add_optional=True)
        ds = sgkit.load_dataset(zarr_path)
        samples_mask = np.ones_like(ds["sample_id"], dtype=bool)
        for i in sample_list:
            samples_mask[i] = False
        tsutil.add_array_to_dataset("samples_mask_69", samples_mask, zarr_path)
        samples = tsinfer.VariantData(
            zarr_path,
            "variant_ancestral_allele",
            sample_mask="samples_mask_69",
        )
        assert samples.ploidy == 3
        assert samples.num_individuals == len(sample_list)
        assert samples.num_samples == len(sample_list) * samples.ploidy
        assert np.array_equal(samples.individuals_select, ~samples_mask)
        assert np.array_equal(samples.samples_select, np.repeat(~samples_mask, 3))
        assert np.array_equal(
            samples.individuals_time, ds.individuals_time.values[~samples_mask]
        )
        assert np.array_equal(
            samples.individuals_location, ds.individuals_location.values[~samples_mask]
        )
        assert np.array_equal(
            samples.individuals_population,
            ds.individuals_population.values[~samples_mask],
        )
        assert np.array_equal(
            samples.individuals_flags, ds.individuals_flags.values[~samples_mask]
        )
        assert np.array_equal(
            samples.samples_individual, np.repeat(np.arange(len(sample_list)), 3)
        )
        expected_gt = ds.call_genotype.values[:, ~samples_mask, :].reshape(
            samples.num_sites, len(sample_list) * 3
        )
        assert np.array_equal(samples.sites_genotypes, expected_gt)
        for v, gt in zip(samples.variants(), expected_gt):
            assert np.array_equal(v.genotypes, gt)

        for i, (id, haplo) in enumerate(samples.haplotypes()):
            assert id == i
            assert np.array_equal(haplo, expected_gt[:, i])

    def test_sgkit_sample_and_site_mask(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        ds = sgkit.load_dataset(zarr_path)
        # Mask out a random 1/3 of sites
        variant_mask = np.zeros(ts.num_sites, dtype=bool)
        random = np.random.RandomState(42)
        variant_mask[
            random.choice(ts.num_sites, ts.num_sites // 3, replace=False)
        ] = True
        # Mask out a random 1/3 of samples
        samples_mask = np.zeros(ts.num_individuals, dtype=bool)
        samples_mask[
            random.choice(ts.num_individuals, ts.num_individuals // 3, replace=False)
        ] = True
        tsutil.add_array_to_dataset(
            "variant_mask_foobar", variant_mask, zarr_path, dims=["variants"]
        )
        tsutil.add_array_to_dataset(
            "samples_mask_foobar", samples_mask, zarr_path, dims=["samples"]
        )
        samples = tsinfer.VariantData(
            zarr_path,
            "variant_ancestral_allele",
            site_mask="variant_mask_foobar",
            sample_mask="samples_mask_foobar",
        )
        genotypes = samples.sites_genotypes
        ds_genotypes = ds.call_genotype.values[~variant_mask][
            :, ~samples_mask, :
        ].reshape(samples.num_sites, samples.num_samples)
        assert np.array_equal(genotypes, ds_genotypes)

        # Check that variants and haplotypes give the same matrix
        for i, (id, haplo) in enumerate(samples.haplotypes()):
            assert id == i
            assert np.array_equal(haplo, genotypes[:, i])
            assert np.array_equal(haplo, ds_genotypes[:, i])

        for i, v in enumerate(samples.variants()):
            assert np.array_equal(v.genotypes, genotypes[i])
            assert np.array_equal(v.genotypes, ds_genotypes[i])

    def test_sgkit_missing_masks(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        samples = tsinfer.VariantData(zarr_path, "variant_ancestral_allele")
        samples.individuals_select
        samples.sites_select
        with pytest.raises(
            ValueError, match="The sites mask foobar was not found in the dataset."
        ):
            tsinfer.VariantData(
                zarr_path, "variant_ancestral_allele", site_mask="foobar"
            )
        with pytest.raises(
            ValueError,
            match="The samples mask foobar2 was not found in the dataset.",
        ):
            samples = tsinfer.VariantData(
                zarr_path,
                "variant_ancestral_allele",
                sample_mask="foobar2",
            )
            samples.individuals_select

    def test_variantdata_default_masks(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        samples = tsinfer.VariantData(zarr_path, "variant_ancestral_allele")
        assert np.array_equal(
            samples.sites_select, np.full(samples.num_sites, True, dtype=bool)
        )
        assert np.array_equal(
            samples.samples_select, np.full(samples.num_samples, True, dtype=bool)
        )

    def test_variantdata_mask_as_arrays(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        sample_mask = np.full(ts.num_individuals, False, dtype=bool)
        site_mask = np.full(ts.num_sites, False, dtype=bool)
        sample_mask[3] = True
        site_mask[5] = True

        samples = tsinfer.VariantData(
            zarr_path,
            "variant_ancestral_allele",
            sample_mask=sample_mask,
            site_mask=site_mask,
        )

        assert np.array_equal(samples.individuals_select, ~sample_mask)
        assert np.array_equal(samples.sites_select, ~site_mask)

    def test_variantdata_incorrect_mask_lengths(self, tmp_path):
        ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
        incorrect_sample_mask = np.array([True, False])  # Incorrect length
        incorrect_site_mask = np.array([True, False, True])  # Incorrect length

        with pytest.raises(
            ValueError,
            match="Samples mask array must be the same length as the"
            " number of individuals",
        ):
            tsinfer.VariantData(
                zarr_path,
                "variant_ancestral_allele",
                sample_mask=incorrect_sample_mask,
            )

        with pytest.raises(
            ValueError,
            match="Site mask array must be the same length as the number of"
            " unmasked sites",
        ):
            tsinfer.VariantData(
                zarr_path,
                "variant_ancestral_allele",
                site_mask=incorrect_site_mask,
            )

    def test_sgkit_subset_equivalence(self, tmp_path, tmpdir):
        (
            mat_sd,
            mask_sd,
            samples_mask,
            variant_mask,
        ) = tsutil.make_materialized_and_masked_sampledata(tmp_path, tmpdir)
        assert mat_sd.num_sites == mat_sd.num_sites
        assert mask_sd.num_samples == mat_sd.num_samples
        assert mask_sd.num_individuals == mat_sd.num_individuals
        assert np.array_equal(mask_sd.individuals_select, ~samples_mask)
        assert np.array_equal(mask_sd.samples_select, np.repeat(~samples_mask, 3))
        assert np.array_equal(mask_sd.sites_position, mat_sd.sites_position)
        assert np.array_equal(mask_sd.sites_alleles, mat_sd.sites_alleles)
        assert np.array_equal(mask_sd.sites_select, ~variant_mask)
        assert np.array_equal(
            mask_sd.sites_ancestral_allele, mat_sd.sites_ancestral_allele
        )
        assert np.array_equal(mask_sd.sites_genotypes, mat_sd.sites_genotypes)
        assert np.array_equal(mask_sd.samples_individual, mat_sd.samples_individual)
        assert np.array_equal(mask_sd.individuals_metadata, mat_sd.individuals_metadata)

        haplotypes = list(mask_sd.haplotypes())
        haplotypes_subset = list(mat_sd.haplotypes())
        assert len(haplotypes) == len(haplotypes_subset)
        for (id, haplo), (id_subset, haplo_subset) in zip(
            haplotypes, haplotypes_subset
        ):
            assert id == id_subset
            assert np.array_equal(haplo, haplo_subset)

        haplotypes = list(
            mask_sd._all_haplotypes(recode_ancestral=True, samples_slice=(9, 24))
        )
        haplotypes_subset = list(
            mat_sd._all_haplotypes(recode_ancestral=True, samples_slice=(9, 24))
        )
        assert len(haplotypes) == len(haplotypes_subset)
        for i, (id, haplo), (id_subset, haplo_subset) in zip(
            range(9, 24), haplotypes, haplotypes_subset
        ):
            assert id == i
            assert id == id_subset
            assert np.array_equal(haplo, haplo_subset)

        variants = list(mask_sd.variants())
        variants_subset = list(mat_sd.variants())
        assert len(variants) == len(variants_subset)
        for v, v_subset in zip(variants, variants_subset):
            assert np.array_equal(v.genotypes, v_subset.genotypes)

        ancestors_subset = tsinfer.generate_ancestors(mat_sd)
        ancestors = tsinfer.generate_ancestors(mask_sd)
        assert np.array_equal(ancestors_subset.sites_position, ancestors.sites_position)
        assert np.array_equal(
            ancestors_subset.ancestors_start, ancestors.ancestors_start
        )
        assert np.array_equal(ancestors_subset.ancestors_end, ancestors.ancestors_end)
        assert np.array_equal(ancestors_subset.ancestors_time, ancestors.ancestors_time)
        for ts_focal, sg_focal in zip(
            ancestors_subset.ancestors_focal_sites, ancestors.ancestors_focal_sites
        ):
            assert np.array_equal(ts_focal, sg_focal)
        assert np.array_equal(
            ancestors_subset.ancestors_full_haplotype,
            ancestors.ancestors_full_haplotype,
        )

        mat_anc_ts = tsinfer.match_ancestors(mat_sd, ancestors_subset)
        mask_anc_ts = tsinfer.match_ancestors(mask_sd, ancestors)
        mat_anc_ts.tables.assert_equals(mask_anc_ts.tables, ignore_timestamps=True)

        mat_ts = tsinfer.match_samples(mat_sd, mat_anc_ts)
        mask_ts = tsinfer.match_samples(mask_sd, mask_anc_ts)
        mat_ts.tables.assert_equals(mask_ts.tables, ignore_timestamps=True)


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_sgkit_ancestral_allele_same_ancestors(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ts_sampledata = tsinfer.SampleData.from_tree_sequence(ts)
    sg_sampledata = tsinfer.VariantData(zarr_path, "variant_ancestral_allele")
    ts_ancestors = tsinfer.generate_ancestors(ts_sampledata)
    sg_ancestors = tsinfer.generate_ancestors(sg_sampledata)
    assert np.array_equal(ts_ancestors.sites_position, sg_ancestors.sites_position)
    assert np.array_equal(ts_ancestors.ancestors_start, sg_ancestors.ancestors_start)
    assert np.array_equal(ts_ancestors.ancestors_end, sg_ancestors.ancestors_end)
    assert np.array_equal(ts_ancestors.ancestors_time, sg_ancestors.ancestors_time)
    for ts_focal, sg_focal in zip(
        ts_ancestors.ancestors_focal_sites, sg_ancestors.ancestors_focal_sites
    ):
        assert np.array_equal(ts_focal, sg_focal)
    assert np.array_equal(
        ts_ancestors.ancestors_full_haplotype, sg_ancestors.ancestors_full_haplotype
    )
    assert np.array_equal(
        ts_ancestors.ancestors_full_haplotype_mask,
        sg_ancestors.ancestors_full_haplotype_mask,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
def test_missing_ancestral_allele(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ds = ds.drop_vars(["variant_ancestral_allele"])
    sgkit.save_dataset(ds, str(zarr_path) + ".tmp")
    with pytest.raises(ValueError, match="variant_ancestral_allele was not found"):
        tsinfer.VariantData(str(zarr_path) + ".tmp", "variant_ancestral_allele")


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_deliberate_ancestral_missingness(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    ancestral_allele = ds.variant_ancestral_allele.values
    ancestral_allele[0] = "N"
    ancestral_allele[1] = "n"
    ds = ds.drop_vars(["variant_ancestral_allele"])
    sgkit.save_dataset(ds, str(zarr_path) + ".tmp")
    tsutil.add_array_to_dataset(
        "variant_ancestral_allele",
        ancestral_allele,
        str(zarr_path) + ".tmp",
        ["variants"],
    )
    ds = sgkit.load_dataset(str(zarr_path) + ".tmp")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # No warning raised if AA deliberately missing
        sd = tsinfer.VariantData(str(zarr_path) + ".tmp", "variant_ancestral_allele")
    inf_ts = tsinfer.infer(sd)
    for i, (inf_var, var) in enumerate(zip(inf_ts.variants(), ts.variants())):
        if i in [0, 1]:
            assert inf_var.site.metadata == {"inference_type": "parsimony"}
        else:
            assert inf_var.site.ancestral_state == var.site.ancestral_state


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_ancestral_missing_warning(tmp_path):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    anc_state = ds.variant_ancestral_allele.values
    anc_state[0] = "N"
    anc_state[11] = "-"
    anc_state[12] = "💩"
    anc_state[15] = "💩"
    with pytest.warns(
        UserWarning,
        match=r"not found in the variant_allele array for the 4 [\s\S]*'💩': 2",
    ):
        vdata = tsinfer.VariantData(zarr_path, anc_state)
    inf_ts = tsinfer.infer(vdata)
    for i, (inf_var, var) in enumerate(zip(inf_ts.variants(), ts.variants())):
        if i in [0, 11, 12, 15]:
            assert inf_var.site.metadata == {"inference_type": "parsimony"}
            assert inf_var.site.ancestral_state in var.site.alleles
        else:
            assert inf_var.site.ancestral_state == var.site.ancestral_state


@pytest.mark.skipif(sys.platform == "win32", reason="No cyvcf2 on Windows")
def test_ancestral_missing_info(tmp_path, caplog):
    ts, zarr_path = tsutil.make_ts_and_zarr(tmp_path)
    ds = sgkit.load_dataset(zarr_path)
    anc_state = ds.variant_ancestral_allele.values
    anc_state[0] = "N"
    anc_state[11] = "N"
    anc_state[12] = "n"
    anc_state[15] = "n"
    with caplog.at_level(logging.INFO):
        vdata = tsinfer.VariantData(zarr_path, anc_state)
    assert f"4 sites ({4/ts.num_sites * 100 :.2f}%) were deliberately " in caplog.text
    inf_ts = tsinfer.infer(vdata)
    for i, (inf_var, var) in enumerate(zip(inf_ts.variants(), ts.variants())):
        if i in [0, 11, 12, 15]:
            assert inf_var.site.metadata == {"inference_type": "parsimony"}
            assert inf_var.site.ancestral_state in var.site.alleles
        else:
            assert inf_var.site.ancestral_state == var.site.ancestral_state


@pytest.mark.skipif(sys.platform == "win32", reason="File permission errors on Windows")
def test_sgkit_ancestor(small_sd_fixture, tmp_path):
    with tempfile.TemporaryDirectory(prefix="tsi_eval") as tmpdir:
        f = f"{tmpdir}/test.ancestors"
        tsinfer.generate_ancestors(small_sd_fixture, path=f)
        store = formats.open_lmbd_readonly(f)
        ds = sgkit.load_dataset(store)
        ds = sgkit.variant_stats(ds, merge=True)
        ds = sgkit.sample_stats(ds, merge=True)
        sgkit.display_genotypes(ds)


class TestVariantDataErrors:
    @staticmethod
    def simulate_genotype_call_dataset(*args, **kwargs):
        # roll our own simulate_genotype_call_dataset to hack around bug in sgkit where
        # duplicate alleles are created. Doesn't need to be efficient: just for testing
        if "seed" not in kwargs:
            kwargs["seed"] = 123
        ds = sgkit.simulate_genotype_call_dataset(*args, **kwargs)
        variant_alleles = ds["variant_allele"].values
        allowed_alleles = np.array(
            ["A", "T", "C", "G", "N"], dtype=variant_alleles.dtype
        )
        for row in range(len(variant_alleles)):
            alleles = variant_alleles[row]
            if len(set(alleles)) != len(alleles):
                # Just use a set that we know is unique
                variant_alleles[row] = allowed_alleles[0 : len(alleles)]
        ds["variant_allele"] = ds["variant_allele"].dims, variant_alleles
        return ds

    def test_bad_zarr_spec(self):
        ds = zarr.group()
        ds["call_genotype"] = zarr.array(np.zeros(10, dtype=np.int8))
        with pytest.raises(
            ValueError, match="Expecting a VCF Zarr object with 3D call_genotype array"
        ):
            tsinfer.VariantData(ds, np.zeros(10, dtype="<U1"))

    def test_missing_phase(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match="The call_genotype_phased array is missing"
        ):
            tsinfer.VariantData(path, "variant_ancestral_allele")

    def test_phased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.ones(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        tsinfer.VariantData(path, ds["variant_allele"][:, 0].values.astype(str))

    def test_ploidy1_missing_phase(self, tmp_path):
        path = tmp_path / "data.zarr"
        # Ploidy==1 is always ok
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        sgkit.save_dataset(ds, path)
        tsinfer.VariantData(path, ds["variant_allele"][:, 0].values.astype(str))

    def test_ploidy1_unphased(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["call_genotype_phased"] = (
            ds["call_genotype"].dims,
            np.zeros(ds["call_genotype"].shape, dtype=bool),
        )
        sgkit.save_dataset(ds, path)
        tsinfer.VariantData(path, ds["variant_allele"][:, 0].values.astype(str))

    def test_duplicate_positions(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ds["variant_position"][2] = ds["variant_position"][1]
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="duplicate or out-of-order values"):
            tsinfer.VariantData(path, "variant_ancestral_allele")

    def test_bad_order_positions(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ds["variant_position"][0] = ds["variant_position"][2] - 0.5
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="duplicate or out-of-order values"):
            tsinfer.VariantData(path, "variant_ancestral_allele")

    def test_bad_ancestral_state(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ancestral_state = ds["variant_allele"][:, 0].values.astype(str)
        ancestral_state[1] = ""
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="cannot contain empty strings"):
            tsinfer.VariantData(path, ancestral_state)

    def test_ancestral_state_len_not_same_as_mask(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        sgkit.save_dataset(ds, path)
        ancestral_state = ds["variant_allele"][:, 0].values.astype(str)
        site_mask = np.zeros(ds.sizes["variants"], dtype=bool)
        site_mask[0] = True
        with pytest.raises(
            ValueError,
            match="Ancestral state array must be the same length as the number of"
            " selected sites",
        ):
            tsinfer.VariantData(path, ancestral_state, site_mask=site_mask)

    def test_empty_alleles_not_at_end(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["variant_allele"] = (
            ds["variant_allele"].dims,
            np.array([["A", "", "C"], ["A", "C", ""], ["A", "C", ""]], dtype="S1"),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match='Bad alleles: fill value "" in middle of list'
        ):
            tsinfer.VariantData(path, ds["variant_allele"][:, 0].values.astype(str))

    def test_unique_alleles(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = self.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_ploidy=1)
        ds["variant_allele"] = (
            ds["variant_allele"].dims,
            np.array([["A", "C", "T"], ["A", "C", ""], ["A", "A", ""]], dtype="S1"),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match="Duplicate allele values provided at site 2"
        ):
            tsinfer.VariantData(path, np.array(["A", "A", "A"], dtype="S1"))

    def test_unimplemented_from_tree_sequence(self):
        # NB we should reimplement something like this functionality.
        # Requires e.g. https://github.com/tskit-dev/tsinfer/issues/924
        with pytest.raises(NotImplementedError):
            tsinfer.VariantData.from_tree_sequence(None)

    def test_multiple_contigs(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ds["contig_id"] = (
            ds["contig_id"].dims,
            np.array(["c10", "c11"], dtype="<U3"),
        )
        ds["variant_contig"] = (
            ds["variant_contig"].dims,
            np.array([0, 0, 1], dtype=ds["variant_contig"].dtype),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match=r'Sites belong to multiple contigs \("c10", "c11"\)'
        ):
            tsinfer.VariantData(path, ds["variant_allele"][:, 0].astype(str))

    def test_all_masked(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match="All sites have been masked out"):
            tsinfer.VariantData(
                path, ds["variant_allele"][:, 0].astype(str), site_mask=np.ones(3, bool)
            )

    def test_bad_contig_param(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match='"XX" not found'):
            tsinfer.VariantData(
                path, ds["variant_allele"][:, 0].astype(str), contig_id="XX"
            )

    def test_multiple_contig_param(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        ds["contig_id"] = (
            ds["contig_id"].dims,
            np.array(["chr1", "chr1"], dtype="<U4"),
        )
        sgkit.save_dataset(ds, path)
        with pytest.raises(ValueError, match='Multiple contigs named "chr1"'):
            tsinfer.VariantData(
                path, ds["variant_allele"][:, 0].astype(str), contig_id="chr1"
            )

    def test_missing_sites_time(self, tmp_path):
        path = tmp_path / "data.zarr"
        ds = sgkit.simulate_genotype_call_dataset(n_variant=3, n_sample=3, phased=True)
        sgkit.save_dataset(ds, path)
        with pytest.raises(
            ValueError, match="The sites time array XX was not found in the dataset"
        ):
            tsinfer.VariantData(
                path, ds["variant_allele"][:, 0].astype(str), sites_time="XX"
            )
