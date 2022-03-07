#
# Copyright (C) 2020 University of Oxford
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
Extra utility functions used in several test files
"""
import json

import msprime
import numpy as np
import tskit

import tsinfer


def mark_mutation_times_unknown(ts):
    """
    Msprime now provides times for mutations, which cannot be estimated by tsinfer
    """
    tables = ts.dump_tables()
    tables.mutations.time = np.full_like(tables.mutations.time, tskit.UNKNOWN_TIME)
    return tables.tree_sequence()


def json_metadata_is_subset(metadata1, metadata2):
    return metadata1.items() <= metadata2.items()


def add_default_schemas(ts):
    """
    Returns a copy of the specified tree sequence with permissive JSON
    schemas on the tables that are used for round-tripping data in tsinfer.
    """
    tables = ts.dump_tables()
    schema = tskit.MetadataSchema(tsinfer.permissive_json_schema())
    assert len(tables.metadata) == 0
    tables.metadata_schema = schema
    tables.metadata = {}
    tables.populations.metadata_schema = schema
    # msprime 1.0 fills the population metadata, so put it back in here
    for pop in ts.populations():
        tables.populations[pop.id] = pop
    tables.individuals.metadata_schema = schema
    assert len(tables.individuals.metadata) == 0
    tables.individuals.packset_metadata([b"{}"] * ts.num_individuals)
    tables.sites.metadata_schema = schema
    assert len(tables.sites.metadata) == 0
    tables.sites.packset_metadata([b"{}"] * ts.num_sites)
    return tables.tree_sequence()


def get_example_ts(
    sample_size,
    sequence_length=10000,
    mutation_rate=0.0005,
    mutation_model=None,
    discrete_genome=True,
    random_seed=100,
):
    ts = msprime.sim_ancestry(
        sample_size,
        ploidy=1,
        sequence_length=sequence_length,
        recombination_rate=mutation_rate * 0.1,
        discrete_genome=discrete_genome,
        random_seed=random_seed,
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, model=mutation_model, random_seed=random_seed
    )
    return add_default_schemas(ts)


def get_example_individuals_ts_with_metadata(
    n,
    ploidy=2,
    sequence_length=10000,
    mutation_rate=0.0002,
    *,
    discrete_genome=True,
    skip_last=True,
):
    """
    For testing only, create a ts with lots of arbitrary metadata attached to sites,
    individuals & populations, and also set flags on individuals (*node* flags such as
    tskit.NODE_IS_SAMPLE are not expected to pass through the inference process, as
    they can be set during inference).

    For testing purposes, we can set ``skip_last`` to check what happens if we have
    some samples that are not associated with an individual in the tree sequence.
    """
    ts = msprime.sim_ancestry(
        n,
        ploidy=ploidy,
        recombination_rate=mutation_rate * 0.1,
        sequence_length=sequence_length,
        random_seed=100,
        discrete_genome=discrete_genome,
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, discrete_genome=discrete_genome, random_seed=100
    )
    ts = add_default_schemas(ts)
    tables = ts.dump_tables()
    tables.metadata = {f"a_{j}": j for j in range(n)}
    tables.populations.clear()
    tables.individuals.clear()
    for i in range(n):
        location = [i, i]
        individual_meta = {}
        pop_meta = {}
        if i % 2 == 0:
            # Add unicode metadata to every other individual: 8544+i = Roman numerals
            individual_meta = {"unicode id": chr(8544 + i)}
            individual_flags = np.random.randint(0, np.iinfo(np.uint32).max)
            # Also for populations: chr(127462) + chr(127462+i) give emoji flags
            pop_meta = {"utf": chr(127462) + chr(127462 + i)}
        tables.populations.add_row(metadata=pop_meta)  # One pop for each individual
        if i < n - 1 or not skip_last:
            tables.individuals.add_row(
                flags=individual_flags, location=location, metadata=individual_meta
            )

    node_populations = tables.nodes.population
    for node in ts.nodes():
        if node.is_sample():
            node_populations[node.id] = node.id // ploidy
    tables.nodes.population = node_populations

    # Manually encode the site metadata to avoid adding the rows one-by-one.
    site_metadata = []
    for site in ts.sites():
        if site.id % 4 == 0:  # Scatter metadata into sites: once every 4th row
            site_metadata.append(json.dumps({"id": f"site {site.id}"}).encode())
        else:
            site_metadata.append(b"{}")
    tables.sites.packset_metadata(site_metadata)

    nodes_individual = tables.nodes.individual  # Assign individuals to sample nodes
    sample_individuals = np.repeat(
        np.arange(n, dtype=tables.nodes.individual.dtype), ploidy
    )
    if skip_last:
        # Should work even if some sample nodes are not assigned to an individual
        sample_individuals[sample_individuals == n - 1] = tskit.NULL
    nodes_individual[ts.samples()] = sample_individuals
    tables.nodes.individual = nodes_individual
    return tables.tree_sequence()


def get_example_historical_sampled_ts(
    individual_times,
    ploidy=2,
    sequence_length=10000,
    mutation_rate=0.0002,
):
    samples = [
        msprime.SampleSet(1, population=0, time=t, ploidy=ploidy)
        for t in individual_times
    ]
    ts = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        recombination_rate=mutation_rate * 0.1,
        sequence_length=sequence_length,
        random_seed=100,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=100)
    ts = add_default_schemas(ts)
    tables = ts.dump_tables()
    # Add individuals
    tables.individuals.clear()
    nodes_individual = tables.nodes.individual
    individual_ids = []
    for _ in individual_times:
        individual_ids.append(tables.individuals.add_row(metadata={}))
    is_sample_node = (ts.tables.nodes.flags & tskit.NODE_IS_SAMPLE) != 0
    nodes_individual[is_sample_node] = np.repeat(individual_ids, ploidy)
    tables.nodes.individual = nodes_individual
    return tables.tree_sequence()
