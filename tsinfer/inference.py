#
# Copyright (C) 2018-2023 University of Oxford
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
Central module for high-level inference. The actual implementation of
of the core tasks like ancestor generation and matching are delegated
to other modules.
"""
import collections
import copy
import dataclasses
import heapq
import itertools
import json
import logging
import queue
import tempfile
import threading
import time

import humanize
import numpy as np
import tskit

import _tsinfer
import tsinfer.algorithm as algorithm
import tsinfer.constants as constants
import tsinfer.formats as formats
import tsinfer.progress as progress
import tsinfer.provenance as provenance
import tsinfer.threads as threads

logger = logging.getLogger(__name__)


sample_data_time_metadata_definition = {
    "description": "Time of an individual from the SampleData file.",
    "type": "number",
    # Defaults aren't currently used, see
    # https://github.com/tskit-dev/tskit/issues/1073
    "default": 0,
}

inference_type_metadata_definition = {
    "description": (
        "The type of inference used at this site. This can be one of the following: "
        f"'{constants.INFERENCE_FULL}' for sites which used the standard tsinfer "
        f"algorithm; '{constants.INFERENCE_NONE}' for sites containing only missing "
        f"data or the ancestral state; '{constants.INFERENCE_PARSIMONY}' for sites "
        "that used a parsimony algorithm to place mutations based on trees inferred "
        "from the remaining data."
    ),
    "type": "string",
    "enum": [
        constants.INFERENCE_NONE,
        constants.INFERENCE_FULL,
        constants.INFERENCE_PARSIMONY,
    ],
}


def add_to_schema(schema, name, definition=None, required=False):
    schema = copy.deepcopy(schema)
    if definition is None:
        definition = {}
    try:
        if name in schema["properties"]:
            raise ValueError(f"The metadata {name} is reserved for use by tsinfer")
    except KeyError:
        schema["properties"] = {}
    schema["properties"][name] = definition
    if required:
        if "required" not in schema:
            schema["required"] = []
        schema["required"].append(name)
    return schema


def is_pc_ancestor(flags):
    """
    Returns True if the path compression ancestor flag is set on the specified
    flags value.
    """
    return (flags & constants.NODE_IS_PC_ANCESTOR) != 0


def is_srb_ancestor(flags):
    """
    Returns True if the shared recombination breakpoint flag is set on the
    specified flags value.
    """
    return (flags & constants.NODE_IS_SRB_ANCESTOR) != 0


def count_pc_ancestors(flags):
    """
    Returns the number of values in the specified array which have the
    NODE_IS_PC_ANCESTOR set.
    """
    flags = np.array(flags, dtype=np.uint32, copy=False)
    return np.sum(is_pc_ancestor(flags))


def count_srb_ancestors(flags):
    """
    Returns the number of values in the specified array which have the
    NODE_IS_SRB_ANCESTOR set.
    """
    flags = np.array(flags, dtype=np.uint32, copy=False)
    return np.sum(np.bitwise_and(flags, constants.NODE_IS_SRB_ANCESTOR) != 0)


AlleleCounts = collections.namedtuple("AlleleCounts", "known ancestral derived")


def allele_counts(genotypes):
    """
    Return summary counts of the number of different allele types for a genotypes array
    """
    n_known = np.sum(genotypes != tskit.MISSING_DATA)
    n_ancestral = np.sum(genotypes == 0)
    return AlleleCounts(
        known=n_known, ancestral=n_ancestral, derived=n_known - n_ancestral
    )


def _get_progress_monitor(progress_monitor, **kwargs):
    """
    Check if this really is a ProgressMonitor, if not, return something usable as one
    """
    if isinstance(progress_monitor, progress.ProgressMonitor):
        return progress_monitor
    if progress_monitor:
        return progress.ProgressMonitor(**kwargs)
    return progress.DummyProgressMonitor()


def _encode_raw_metadata(obj):
    return json.dumps(obj).encode()


def _update_site_metadata(current_metadata, inference_type):
    return {"inference_type": inference_type, **current_metadata}


def verify(sample_data, tree_sequence, progress_monitor=None):
    """
    verify(samples, tree_sequence)

    Verifies that the specified sample data and tree sequence files encode the
    same data.

    :param SampleData samples: The input :class:`SampleData` instance
        representing the observed data that we wish to compare to.
    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`
        instance an encoding of the specified samples that we wish to verify.
    """
    progress_monitor = _get_progress_monitor(progress_monitor, verify=True)
    if sample_data.num_sites != tree_sequence.num_sites:
        raise ValueError("numbers of sites not equal")
    if sample_data.num_samples != tree_sequence.num_samples:
        raise ValueError("numbers of samples not equal")
    if sample_data.sequence_length != tree_sequence.sequence_length:
        raise ValueError("Sequence lengths not equal")
    progress = progress_monitor.get("verify", tree_sequence.num_sites)
    for var1, var2 in zip(
        sample_data.variants(recode_ancestral=True), tree_sequence.variants()
    ):
        if var1.site.position != var2.site.position:
            raise ValueError(
                "site positions not equal: {} != {}".format(
                    var1.site.position, var2.site.position
                )
            )
        # First (ancestral) allele should always be the same
        if var1.alleles[0] != var2.alleles[0]:
            raise ValueError(f"Ancestral allele not equal at site {var1.site.id}")
        if var1.alleles != var2.alleles:
            # Alleles may be in a different order, or even present/absent if not in the
            # genotype matrix so we need to explicitly compare the decoded values (slow)
            for i, (g1, g2) in enumerate(zip(var1.genotypes, var2.genotypes)):
                # We don't expect missingness in a tsinfer generated tree sequence
                assert g2 != tskit.NULL
                if g1 != tskit.NULL and var1.alleles[g1] != var2.alleles[g2]:
                    raise ValueError(
                        f"Alleles for sample {i} not equal at site {var1.site.id}"
                    )
        else:
            g1 = var1.genotypes
            g2 = np.copy(var2.genotypes)
            missing_mask = g1 == tskit.NULL
            g2[missing_mask] = tskit.NULL
            if not np.array_equal(g1, g2):
                raise ValueError(f"Genotypes not equal at site {var1.site.id}")
        progress.update()
    progress.close()


def check_sample_indexes(sample_data, indexes):
    """
    Checks that the specified sample indexes are valid for the specified
    sample data file.
    """
    if indexes is None:
        return np.arange(sample_data.num_samples, dtype=np.int32)
    indexes = np.array(indexes)
    if len(indexes) == 0:
        raise ValueError("Must supply at least one sample to match")
    if np.any(indexes < 0) or np.any(indexes >= sample_data.num_samples):
        raise ValueError("Sample index out of bounds")
    if np.any(indexes[:-1] >= indexes[1:]):
        raise ValueError("Sample indexes must be in increasing order")
    return indexes


def infer(
    sample_data,
    *,
    recombination_rate=None,
    mismatch_ratio=None,
    path_compression=True,
    exclude_positions=None,
    post_process=None,
    num_threads=0,
    # Deliberately undocumented parameters below
    precision=None,
    engine=constants.C_ENGINE,
    progress_monitor=None,
    time_units=None,
    simplify=None,  # Deprecated
    record_provenance=True,
):
    """
    infer(sample_data, *, recombination_rate=None, mismatch_ratio=None,\
            path_compression=True, exclude_positions=None, post_process=None,\
            num_threads=0)

    Runs the full :ref:`inference pipeline <sec_inference>` on the specified
    :class:`SampleData` instance and returns the inferred
    :class:`tskit.TreeSequence`.  See
    :ref:`matching ancestors & samples<sec_inference_match_ancestors_and_samples>`
    in the documentation for details of ``recombination_rate``, ``mismatch_ratio``
    and ``path_compression``.

    .. note::
        For finer grained control over inference, for example to set different mismatch
        ratios when matching ancestors versus samples, run
        :func:`tsinfer.generate_ancestors`, :func:`tsinfer.match_ancestors` and
        :func:`tsinfer.match_samples` separately.

    :param SampleData sample_data: The input :class:`SampleData` instance
        representing the observed data that we wish to make inferences from.
    :param recombination_rate: Either a floating point value giving a constant rate
        :math:`\\rho` per unit length of genome, or an :class:`msprime.RateMap`
        object. This is used to calculate the probability of recombination between
        adjacent sites. If ``None``, all matching conflicts are resolved by
        recombination and all inference sites will have a single mutation
        (equivalent to mismatch_ratio near zero)
    :type recombination_rate: float, msprime.RateMap
    :param float mismatch_ratio: The probability of a mismatch relative to the median
        probability of recombination between adjacent sites: can only be used if a
        recombination rate has been set (default: 1)
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param bool post_process: Whether to run the :func:`post_process` method on the
        the tree sequence which, among other things, removes ancestral material that
        does not end up in the current samples (if not specified, defaults to ``True``)
    :param array_like exclude_positions: A list of site positions to exclude
        for full inference. Sites with these positions will not be used to generate
        ancestors, and not used during the copying process. Any such sites that
        exist in the sample data file will be included in the trees after the
        main inference process using parsimony. The list does not need to be
        in to be in any particular order, and can include site positions that
        are not present in the sample data file.
    :param int num_threads: The number of worker threads to use in parallelised
        sections of the algorithm. If <= 0, do not spawn any threads and
        use simpler sequential algorithms (default).
    :param bool simplify: When post_processing, only simplify the tree sequence.
        deprecated but retained for backwards compatibility (default: ``None``).
    :return: The :class:`tskit.TreeSequence` object inferred from the
        input sample data.
    :rtype: tskit.TreeSequence
    """
    progress_monitor = _get_progress_monitor(
        progress_monitor,
        generate_ancestors=True,
        match_ancestors=True,
        match_samples=True,
    )
    ancestor_data = generate_ancestors(
        sample_data,
        num_threads=num_threads,
        exclude_positions=exclude_positions,
        engine=engine,
        progress_monitor=progress_monitor,
        record_provenance=False,
    )
    ancestors_ts = match_ancestors(
        sample_data,
        ancestor_data,
        engine=engine,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_ratio=mismatch_ratio,
        precision=precision,
        path_compression=path_compression,
        progress_monitor=progress_monitor,
        time_units=time_units,
        record_provenance=False,
    )
    inferred_ts = match_samples(
        sample_data,
        ancestors_ts,
        engine=engine,
        num_threads=num_threads,
        recombination_rate=recombination_rate,
        mismatch_ratio=mismatch_ratio,
        precision=precision,
        post_process=post_process,
        path_compression=path_compression,
        progress_monitor=progress_monitor,
        simplify=simplify,
        record_provenance=False,
    )
    if record_provenance:
        tables = inferred_ts.dump_tables()
        record = provenance.get_provenance_dict(command="infer")
        tables.provenances.add_row(record=json.dumps(record))
        inferred_ts = tables.tree_sequence()
    return inferred_ts


def generate_ancestors(
    sample_data,
    *,
    path=None,
    exclude_positions=None,
    num_threads=0,
    genotype_encoding=None,
    mmap_temp_dir=None,
    # Deliberately undocumented parameters below
    engine=constants.C_ENGINE,
    progress_monitor=None,
    record_provenance=True,
    **kwargs,
):
    """
    generate_ancestors(sample_data, *, path=None, exclude_positions=None,\
        num_threads=0, genotype_encoding=None, mmap_temp_dir=None, **kwargs)

    Runs the ancestor generation :ref:`algorithm <sec_inference_generate_ancestors>`
    on the specified :class:`SampleData` instance and returns the resulting
    :class:`AncestorData` instance. If you wish to store generated ancestors
    persistently on file you must pass the ``path`` keyword argument to this
    function. For example,

    .. code-block:: python

        ancestor_data = tsinfer.generate_ancestors(sample_data, path="mydata.ancestors")

    Other keyword arguments are passed to the :class:`AncestorData` constructor,
    which may be used to control the storage properties of the generated file.

    Ancestor generation involves loading the entire genotype matrix into
    memory, by default using one byte per haploid genotype, which can be
    prohibitively large when working with sample sizes of 100,000 or more.
    There are two options to help mitigate memory usage. The
    ``genotype_encoding`` parameter allows the user to specify a more compact
    encoding scheme, which reduces storage space for datasets with small
    numbers of alleles. Currently, the :attr:`.GenotypeEncoding.ONE_BIT`
    encoding is supported, which provides 8-fold compression of biallelic,
    non-missing data. An error is raised if an encoding that does not support
    the range of values present in a given dataset is provided.

    The second option for reducing the RAM footprint of this function is to
    use the ``mmap_temp_dir`` parameter. This allows the genotype data to be
    cached on file, transparently using the operating system's virtual memory
    subsystem to swap in and out the data. This can work well if the encoded
    genotype matrix *almost* fits in RAM and fast local storage is available.
    However, if the size of the encoded genotype matrix is more than, say,
    twice the available RAM it is unlikely that this function will complete
    in a reasonable time-frame. A temporary file is created in the specified
    ``mmap_temp_dir``, which is automatically deleted when the function
    completes.

    .. warning:: The ``mmap_temp_dir`` option is a silent no-op on Windows!

    :param SampleData sample_data: The :class:`SampleData` instance that we are
        genering putative ancestors from.
    :param str path: The path of the file to store the sample data. If None,
        the information is stored in memory and not persistent.
    :param array_like exclude_positions: A list of site positions to exclude
        for full inference. Sites with these positions will not be used to generate
        ancestors, and not used during the copying process. The list does not
        need be in any particular order.
    :param int num_threads: The number of worker threads to use. If < 1, use a
        simpler synchronous algorithm.
    :param int genotype_encoding: The encoding to use for genotype data internally
        when generating ancestors. See the :class:`.GenotypeEncoding` class for
        the available options. Defaults to one-byte per genotype.
    :param str mmap_temp_dir: The directory within which to create the
        temporary backing file when using mmaped memory for bulk genotype
        storage. If None (the default) allocate memory directly using the
        standard mechanism. This is an advanced option, usually only relevant
        when working with very large datasets (see above for more information).
    :return: The inferred ancestors stored in an :class:`AncestorData` instance.
    :rtype: AncestorData
    """
    sample_data._check_finalised()
    if np.any(np.isfinite(sample_data.sites_time[:])) and np.any(
        tskit.is_unknown_time(sample_data.sites_time[:])
    ):
        raise ValueError(
            "Cannot generate ancestors from a sample_data instance that mixes user-"
            "specified times with times-as-frequencies. To explicitly set an undefined"
            "time for a site, permanently excluding it from inference, set it to np.nan."
        )
    if genotype_encoding is None:
        # TODO should we provide some functionality to automatically figure
        # out what the minimum encoding is?
        genotype_encoding = constants.GenotypeEncoding.EIGHT_BIT
    generator = AncestorsGenerator(
        sample_data,
        ancestor_data_path=path,
        ancestor_data_kwargs=kwargs,
        num_threads=num_threads,
        engine=engine,
        genotype_encoding=genotype_encoding,
        mmap_temp_dir=mmap_temp_dir,
        progress_monitor=progress_monitor,
    )
    generator.add_sites(exclude_positions)
    ancestor_data = generator.run()
    for timestamp, record in sample_data.provenances():
        ancestor_data.add_provenance(timestamp, record)
    if record_provenance:
        ancestor_data.record_provenance("generate_ancestors")
    ancestor_data.finalise()
    return ancestor_data


def match_ancestors(
    sample_data,
    ancestor_data,
    *,
    recombination_rate=None,
    mismatch_ratio=None,
    path_compression=True,
    num_threads=0,
    # Deliberately undocumented parameters below
    recombination=None,  # See :class:`Matcher`
    mismatch=None,  # See :class:`Matcher`
    precision=None,
    engine=constants.C_ENGINE,
    progress_monitor=None,
    extended_checks=False,
    time_units=None,
    record_provenance=True,
):
    """
    match_ancestors(sample_data, ancestor_data, *, recombination_rate=None,\
        mismatch_ratio=None, path_compression=True, num_threads=0)

    Run the ancestor matching :ref:`algorithm <sec_inference_match_ancestors>`
    on the specified :class:`SampleData` and :class:`AncestorData` instances,
    returning the resulting :class:`tskit.TreeSequence` representing the
    complete ancestry of the putative ancestors. See
    :ref:`matching ancestors & samples<sec_inference_match_ancestors_and_samples>`
    in the documentation for details of ``recombination_rate``, ``mismatch_ratio``
    and ``path_compression``.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param AncestorData ancestor_data: The :class:`AncestorData` instance
        representing the set of ancestral haplotypes for which we are finding
        a history.
    :param recombination_rate: Either a floating point value giving a constant rate
        :math:`\\rho` per unit length of genome, or an :class:`msprime.RateMap`
        object. This is used to calculate the probability of recombination between
        adjacent sites. If ``None``, all matching conflicts are resolved by
        recombination and all inference sites will have a single mutation
        (equivalent to mismatch_ratio near zero)
    :type recombination_rate: float, msprime.RateMap
    :param float mismatch_ratio: The probability of a mismatch relative to the median
        probability of recombination between adjacent sites: can only be used if a
        recombination rate has been set (default: 1)
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :return: The ancestors tree sequence representing the inferred history
        of the set of ancestors.
    :rtype: tskit.TreeSequence
     """
    progress_monitor = _get_progress_monitor(progress_monitor, match_ancestors=True)
    sample_data._check_finalised()
    ancestor_data._check_finalised()

    matcher = AncestorMatcher(
        sample_data,
        ancestor_data,
        time_units=time_units,
        recombination_rate=recombination_rate,
        recombination=recombination,
        mismatch_ratio=mismatch_ratio,
        mismatch=mismatch,
        path_compression=path_compression,
        num_threads=num_threads,
        precision=precision,
        extended_checks=extended_checks,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    ts = matcher.match_ancestors()
    tables = ts.dump_tables()
    for timestamp, record in ancestor_data.provenances():
        tables.provenances.add_row(timestamp=timestamp, record=json.dumps(record))
    if record_provenance:
        record = provenance.get_provenance_dict(command="match_ancestors")
        tables.provenances.add_row(record=json.dumps(record))
    ts = tables.tree_sequence()
    return ts


def augment_ancestors(
    sample_data,
    ancestors_ts,
    indexes,
    *,
    recombination_rate=None,
    mismatch_ratio=None,
    path_compression=True,
    num_threads=0,
    # Deliberately undocumented parameters below
    recombination=None,  # See :class:`Matcher`
    mismatch=None,  # See :class:`Matcher`
    precision=None,
    extended_checks=False,
    engine=constants.C_ENGINE,
    progress_monitor=None,
    record_provenance=True,
):
    """
    augment_ancestors(sample_data, ancestors_ts, indexes, *, recombination_rate=None,\
        mismatch_ratio=None, path_compression=True, num_threads=0)

    Runs the sample matching :ref:`algorithm <sec_inference_match_samples>`
    on the specified :class:`SampleData` instance and ancestors tree sequence,
    for the specified subset of sample indexes, returning the
    :class:`tskit.TreeSequence` instance including these samples. This
    tree sequence can then be used as an ancestors tree sequence for subsequent
    matching against all samples.  See
    :ref:`matching ancestors & samples<sec_inference_match_ancestors_and_samples>`
    in the documentation for details of ``recombination_rate``, ``mismatch_ratio``
    and ``path_compression``.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param tskit.TreeSequence ancestors_ts: The
        :class:`tskit.TreeSequence` instance representing the inferred
        history among ancestral ancestral haplotypes.
    :param array indexes: The sample indexes to insert into the ancestors
        tree sequence, in increasing order.
    :param recombination_rate: Either a floating point value giving a constant rate
        :math:`\\rho` per unit length of genome, or an :class:`msprime.RateMap`
        object. This is used to calculate the probability of recombination between
        adjacent sites. If ``None``, all matching conflicts are resolved by
        recombination and all inference sites will have a single mutation
        (equivalent to mismatch_ratio near zero)
    :type recombination_rate: float, msprime.RateMap
    :param float mismatch_ratio: The probability of a mismatch relative to the median
        probability of recombination between adjacent sites: can only be used if a
        recombination rate has been set (default: 1)
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :return: The specified ancestors tree sequence augmented with copying
        paths for the specified sample.
    :rtype: tskit.TreeSequence
    """
    sample_data._check_finalised()
    progress_monitor = _get_progress_monitor(progress_monitor, augment_ancestors=True)
    manager = SampleMatcher(
        sample_data,
        ancestors_ts,
        recombination_rate=recombination_rate,
        mismatch_ratio=mismatch_ratio,
        recombination=recombination,
        mismatch=mismatch,
        path_compression=path_compression,
        num_threads=num_threads,
        precision=precision,
        extended_checks=extended_checks,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    sample_indexes = check_sample_indexes(sample_data, indexes)
    sample_times = np.zeros(
        len(sample_indexes), dtype=sample_data.individuals_time.dtype
    )
    manager.match_samples(sample_indexes, sample_times)
    ts = manager.get_augmented_ancestors_tree_sequence(sample_indexes)
    if record_provenance:
        tables = ts.dump_tables()
        record = provenance.get_provenance_dict(command="augment_ancestors")
        tables.provenances.add_row(record=json.dumps(record))
        ts = tables.tree_sequence()
    return ts


def match_samples(
    sample_data,
    ancestors_ts,
    *,
    recombination_rate=None,
    mismatch_ratio=None,
    path_compression=True,
    indexes=None,
    post_process=None,
    force_sample_times=False,
    num_threads=0,
    # Deliberately undocumented parameters below
    recombination=None,  # See :class:`Matcher`
    mismatch=None,  # See :class:`Matcher`
    precision=None,
    extended_checks=False,
    engine=constants.C_ENGINE,
    progress_monitor=None,
    simplify=None,  # deprecated
    record_provenance=True,
):
    """
    match_samples(sample_data, ancestors_ts, *, recombination_rate=None,\
        mismatch_ratio=None, path_compression=True, post_process=None,\
        indexes=None, force_sample_times=False, num_threads=0)

    Runs the sample matching :ref:`algorithm <sec_inference_match_samples>`
    on the specified :class:`SampleData` instance and ancestors tree sequence,
    returning the final :class:`tskit.TreeSequence` instance containing
    the full inferred history for all samples and sites. See
    :ref:`matching ancestors & samples<sec_inference_match_ancestors_and_samples>`
    in the documentation for details of ``recombination_rate``, ``mismatch_ratio``
    and ``path_compression``.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param tskit.TreeSequence ancestors_ts: The
        :class:`tskit.TreeSequence` instance representing the inferred
        history among ancestral ancestral haplotypes.
    :param recombination_rate: Either a floating point value giving a constant rate
        :math:`\\rho` per unit length of genome, or an :class:`msprime.RateMap`
        object. This is used to calculate the probability of recombination between
        adjacent sites. If ``None``, all matching conflicts are resolved by
        recombination and all inference sites will have a single mutation
        (equivalent to mismatch_ratio near zero)
    :type recombination_rate: float, msprime.RateMap
    :param float mismatch_ratio: The probability of a mismatch relative to the median
        probability of recombination between adjacent sites: can only be used if a
        recombination rate has been set (default: 1)
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param array_like indexes: An array of indexes into the sample_data file of
        the samples to match (in increasing order) or None for all samples.
    :param bool post_process: Whether to run the :func:`post_process` method on the
        the tree sequence which, among other things, removes ancestral material that
        does not end up in the current samples (if not specified, defaults to ``True``)
    :param bool force_sample_times: After matching, should an attempt be made to
        adjust the time of "historical samples" (those associated with an individual
        having a non-zero time) such that the sample nodes in the tree sequence
        appear at the time of the individual with which they are associated.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool simplify: Treated as an alias for ``post_process``, deprecated but
        currently retained for backwards compatibility if set to ``False``.

    :return: The tree sequence representing the inferred history
        of the sample.
    :rtype: tskit.TreeSequence
    """
    simplify_only = False  # if true, carry out "old" (deprecated) simplify behaviour
    if simplify is None:
        if post_process is None:
            post_process = True
    else:
        if post_process is not None:
            raise ValueError("Can't specify both `simplify` and `post_process`")
        else:
            if simplify:
                logger.warning(
                    "The `simplify` parameter is deprecated in favour of `post_process`"
                )
                simplify_only = True
                post_process = True
            else:
                post_process = False

    sample_data._check_finalised()
    progress_monitor = _get_progress_monitor(progress_monitor, match_samples=True)
    manager = SampleMatcher(
        sample_data,
        ancestors_ts,
        recombination_rate=recombination_rate,
        mismatch_ratio=mismatch_ratio,
        recombination=recombination,
        mismatch=mismatch,
        path_compression=path_compression,
        num_threads=num_threads,
        precision=precision,
        extended_checks=extended_checks,
        engine=engine,
        progress_monitor=progress_monitor,
    )
    sample_indexes = check_sample_indexes(sample_data, indexes)
    sample_times = np.zeros(
        len(sample_indexes), dtype=sample_data.individuals_time.dtype
    )
    if force_sample_times:
        individuals = sample_data.samples_individual[:][sample_indexes]
        # By construction all samples in an sd file have an individual: but check anyway
        assert np.all(individuals >= 0)
        sample_times = sample_data.individuals_time[:][individuals]

        # Here we might want to re-order sample_indexes and sample_times
        # so that any historical ones come first, any we bomb out early if they conflict
        # but that would mean re-ordering the sample nodes in the final ts, and
        # we sometimes assume they are in the same order as in the file

    manager.match_samples(sample_indexes, sample_times)
    ts = manager.finalise()
    if post_process:
        ts = _post_process(
            ts, warn_if_unexpected_format=True, simplify_only=simplify_only
        )
    if record_provenance:
        tables = ts.dump_tables()
        # We don't have a source here because tree sequence files don't have a UUID yet.
        record = provenance.get_provenance_dict(command="match_samples")
        tables.provenances.add_row(record=json.dumps(record))
        ts = tables.tree_sequence()
    return ts


def insert_missing_sites(
    sample_data, tree_sequence, *, sample_id_map=None, progress_monitor=None
):
    """
    Return a new tree sequence containing extra sites that are present in a
    :class:`SampleData` instance but are missing from a corresponding tree sequence.
    At each newly inserted site, mutations are overlaid parsimoneously, using
    :meth:`tskit.Tree.map_mutations`,
    such that the realised variation at that site corresponds to the allelic
    distribution seen in the sample_data file. Sites that have mutations overlaid
    in this way can be identified in the output tree sequence as their
    :ref:`metadata<tskit.sec_metadata_definition>` will contain a key named
    ``inference`` set to ``tsinfer.INFERENCE_PARSIMONY``. Newly inserted sites
    that do not require mutations will have this set to `tsinfer.INFERENCE_NONE`
    instead. Sites in ``sample_data`` that already exist in the tree sequence are
    left untouched.

    By default, sample 0 in ``sample_data`` is assumed to correspond to the first
    sample node in the input tree sequence (i.e. ``tree_sequence.samples()[0]``),
    sample 1 to the second sample node, and so on. If this is not the case, a map
    can be provided, which specifies the sample ids in ``sample_data`` that
    correspond to the sample nodes in the tree sequence. This also allows the use
    of :class:`SampleData` instances that contain samples in addition to those
    in the original tree sequence.

    .. note::
        Sample states observed as missing in the input ``sample_data`` need
        not correspond to samples whose nodes are actually "missing" (i.e.
        :ref:`isolated<tskit.sec_data_model_missing_data>`) in the input tree
        sequence. In this case, the allelic state of the sample in the returned
        tree sequence will be imputed to the most parsimonious state.

    .. note::
        If the ancestral state at a site is unknown (i.e. ``tskit.MISSING_DATA``),
        it will be inferred by parsimony. If it is unknown and all sample data
        is missing at that site, the site will be created with an ancestral state
        set to the empty string.

    :param SampleData sample_data: The :class:`SampleData` instance
        containing some sites that are not in the input tree sequence.
    :param tskit.TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`
        whose sample nodes correspond to a set of samples in the sample_data.
    :param sample_id_map array: An array of length `tree_sequence.num_samples`
        specifying the indexes of samples in the sample_data file that correspond
        to sample nodes ``0..(num_samples-1)`` in the tree sequence. If None,
        assume that all the samples in sample_data correspond to the sample nodes
        in the tree sequence, and are in the same order.
    :return: The input tree sequence with additional sites and mutations.
    :rtype: tskit.TreeSequence

    """
    if sample_data.sequence_length != tree_sequence.sequence_length:
        raise ValueError(
            "sample_data and tree_sequence must have the same sequence length"
        )
    if sample_id_map is None:
        sample_id_map = np.arange(sample_data.num_samples)
    if len(sample_id_map) != tree_sequence.num_samples:
        raise ValueError(
            "You must specify the same number of samples in sample_data "
            "as in the tree_sequence"
        )
    progress_monitor = _get_progress_monitor(progress_monitor)
    tables = tree_sequence.dump_tables()
    trees = tree_sequence.trees()
    tree = next(trees)
    positions = sample_data.sites_position[:]
    new_sd_sites = np.where(np.isin(positions, tables.sites.position) == 0)[0]
    schema = tables.sites.metadata_schema.schema

    # Create new sites and add the mutations
    progress = progress_monitor.get("ms_extra_sites", len(new_sd_sites))
    for variant in sample_data.variants(sites=new_sd_sites, recode_ancestral=True):
        site = variant.site
        pos = site.position
        anc_state = site.ancestral_state
        anc_value = 0  # variant(recode_ancestral=True) always has 0 as the anc index
        G = variant.genotypes[sample_id_map]
        # We can't perform parsimony inference if all sites are missing, and there's no
        # point if all non-missing sites are the ancestral state, so skip these cases
        if np.all(np.logical_or(G == tskit.MISSING_DATA, G == anc_value)):
            metadata = _update_site_metadata(
                site.metadata, inference_type=constants.INFERENCE_NONE
            )
            if schema is None:
                metadata = _encode_raw_metadata(metadata)
            tables.sites.add_row(
                position=pos,
                ancestral_state="" if anc_state is None else anc_state,
                metadata=metadata,
            )
        else:
            while tree.interval[1] <= pos:
                tree = next(trees)
            anc_state, mapped_mutations = tree.map_mutations(
                G, variant.alleles, ancestral_state=anc_state
            )
            metadata = _update_site_metadata(
                site.metadata, inference_type=constants.INFERENCE_PARSIMONY
            )
            if schema is None:
                metadata = _encode_raw_metadata(metadata)
            new_site_id = tables.sites.add_row(
                position=pos,
                ancestral_state=anc_state,
                metadata=metadata,
            )
            mut_map = {tskit.NULL: tskit.NULL}
            for i, mutation in enumerate(mapped_mutations):
                mut_map[i] = tables.mutations.add_row(
                    site=new_site_id,
                    node=mutation.node,
                    derived_state=mutation.derived_state,
                    parent=mut_map[mutation.parent],
                )
        progress.update()
    progress.close()

    tables.sort()
    return tables.tree_sequence()


class AncestorsGenerator:
    """
    Manages the process of building ancestors.
    """

    def __init__(
        self,
        sample_data,
        ancestor_data_path,
        ancestor_data_kwargs,
        num_threads=0,
        engine=constants.C_ENGINE,
        genotype_encoding=constants.GenotypeEncoding.EIGHT_BIT,
        mmap_temp_dir=None,
        progress_monitor=None,
    ):
        self.sample_data = sample_data
        self.ancestor_data_path = ancestor_data_path
        self.ancestor_data_kwargs = ancestor_data_kwargs
        self.progress_monitor = _get_progress_monitor(
            progress_monitor, generate_ancestors=True
        )
        self.max_sites = sample_data.num_sites
        self.num_sites = 0
        self.inference_site_ids = []
        self.num_samples = sample_data.num_samples
        self.num_threads = num_threads
        self.mmap_temp_file = None
        mmap_fd = -1

        genotype_matrix_size = self.max_sites * self.num_samples
        if genotype_encoding == constants.GenotypeEncoding.ONE_BIT:
            genotype_matrix_size /= 8
        genotype_mem = humanize.naturalsize(genotype_matrix_size, binary=True)
        logging.info(f"Max encoded genotype matrix size={genotype_mem}")
        if mmap_temp_dir is not None:
            self.mmap_temp_file = tempfile.NamedTemporaryFile(
                dir=mmap_temp_dir, prefix="tsinfer-mmap-genotypes-"
            )
            logging.info(f"Using mmapped {self.mmap_temp_file.name} for genotypes")
            mmap_fd = self.mmap_temp_file.fileno()
        if engine == constants.C_ENGINE:
            logger.debug("Using C AncestorBuilder implementation")
            self.ancestor_builder = _tsinfer.AncestorBuilder(
                self.num_samples,
                self.max_sites,
                genotype_encoding=genotype_encoding,
                mmap_fd=mmap_fd,
            )
        elif engine == constants.PY_ENGINE:
            logger.debug("Using Python AncestorBuilder implementation")
            self.ancestor_builder = algorithm.AncestorBuilder(
                self.num_samples,
                self.max_sites,
                genotype_encoding=genotype_encoding,
            )
        else:
            raise ValueError(f"Unknown engine:{engine}")

    def add_sites(self, exclude_positions=None):
        """
        Add all sites that are suitable for inference into the ancestor builder
        (and subsequent inference), unless they are held in the specified list of
        excluded site positions. Suitable sites have only 2 listed alleles, one of
        which is defined as the ancestral_state, and where at least two samples
        carry the derived allele and at least one sample carries the ancestral allele.

        Suitable sites will be added at the time given by site.time, unless
        site.time is  ``np.nan`` or ``tskit.UNKNOWN_TIME``. In the first case,
        the site will simply excluded as if it were in the list of
        ``excluded_positions``. In the second case, then the time associated with
        the site will be the frequency of the derived allele (i.e. the number
        of samples with the derived allele divided by the total number of samples
        with non-missing alleles).
        """
        if exclude_positions is None:
            exclude_positions = set()
        else:
            exclude_positions = np.array(exclude_positions, dtype=np.float64)
            if len(exclude_positions.shape) != 1:
                raise ValueError("exclude_positions must be a 1D array of numbers")
        exclude_positions = set(exclude_positions)

        logger.info(f"Starting addition of {self.max_sites} sites")
        progress = self.progress_monitor.get("ga_add_sites", self.max_sites)
        inference_site_id = []
        for variant in self.sample_data.variants(recode_ancestral=True):
            # If there's missing data the last allele is None
            num_alleles = len(variant.alleles) - int(variant.alleles[-1] is None)

            counts = allele_counts(variant.genotypes)
            use_site = False
            site = variant.site
            if (
                site.position not in exclude_positions
                and num_alleles == 2  # This will ensure that the derived state is "1"
                and 1 < counts.derived < counts.known
                and site.ancestral_state is not None
            ):
                use_site = True
                time = site.time
                if tskit.is_unknown_time(time):
                    # Non-variable sites have no obvious freq-as-time values
                    assert counts.known != counts.derived
                    assert counts.known != counts.ancestral
                    time = counts.derived / counts.known
                if np.isnan(time):
                    use_site = False  # Site with meaningless time value: skip inference
            if use_site:
                self.ancestor_builder.add_site(time, variant.genotypes)
                inference_site_id.append(site.id)
                self.num_sites += 1
            progress.update()
        progress.close()
        self.inference_site_ids = inference_site_id
        logger.info("Finished adding sites")

    def _run_synchronous(self, progress):
        a = np.zeros(self.num_sites, dtype=np.int8)
        for t, focal_sites in self.descriptors:
            before = time.perf_counter()
            start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
            duration = time.perf_counter() - before
            logger.debug(
                "Made ancestor in {:.2f}s at timepoint {} (epoch {}) "
                "from {} to {} (len={}) with {} focal sites ({})".format(
                    duration,
                    t,
                    self.timepoint_to_epoch[t],
                    start,
                    end,
                    end - start,
                    len(focal_sites),
                    focal_sites,
                )
            )
            self.ancestor_data.add_ancestor(
                start=start,
                end=end,
                time=t,
                focal_sites=focal_sites,
                haplotype=a[start:end],
            )
            progress.update()

    def _run_threaded(self, progress):
        # This works by pushing the ancestor descriptors onto the build_queue,
        # which the worker threads pop off and process. We need to add ancestors
        # in the the ancestor_data object in the correct order, so we maintain
        # a priority queue (add_queue) which allows us to track the next smallest
        # index of the generated ancestor. We add build ancestors to this queue
        # as they are built, and drain it when we can.
        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        build_queue = queue.Queue(queue_depth)
        add_lock = threading.Lock()
        next_add_index = 0
        add_queue = []

        def drain_add_queue():
            nonlocal next_add_index
            num_drained = 0
            while len(add_queue) > 0 and add_queue[0][0] == next_add_index:
                _, t, focal_sites, s, e, haplotype = heapq.heappop(add_queue)
                self.ancestor_data.add_ancestor(
                    start=s, end=e, time=t, focal_sites=focal_sites, haplotype=haplotype
                )
                progress.update()
                next_add_index += 1
                num_drained += 1
            logger.debug(f"Drained {num_drained} ancestors from add queue")

        def build_worker(thread_index):
            a = np.zeros(self.num_sites, dtype=np.int8)
            while True:
                work = build_queue.get()
                if work is None:
                    break
                index, t, focal_sites = work
                start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
                with add_lock:
                    haplotype = a[start:end].copy()
                    heapq.heappush(
                        add_queue, (index, t, focal_sites, start, end, haplotype)
                    )
                    drain_add_queue()
                build_queue.task_done()
            build_queue.task_done()

        build_threads = [
            threads.queue_consumer_thread(
                build_worker, build_queue, name=f"build-worker-{j}", index=j
            )
            for j in range(self.num_threads)
        ]
        logger.debug(f"Started {self.num_threads} build worker threads")

        for index, (t, focal_sites) in enumerate(self.descriptors):
            build_queue.put((index, t, focal_sites))

        # Stop the the worker threads.
        for _ in range(self.num_threads):
            build_queue.put(None)
        for j in range(self.num_threads):
            build_threads[j].join()
        drain_add_queue()

    def run(self):
        descriptors = self.ancestor_builder.ancestor_descriptors()
        peak_ram = humanize.naturalsize(self.ancestor_builder.mem_size, binary=True)
        logger.info(f"Ancestor builder peak RAM: {peak_ram}")

        # Sort the descriptors so that we deterministically create ancestors
        # in the same order across implementations
        d = [(t, tuple(focal_sites)) for t, focal_sites in descriptors]
        self.descriptors = sorted(d, reverse=True)
        self.num_ancestors = len(self.descriptors)
        # Maps epoch numbers to their corresponding ancestor times.
        self.timepoint_to_epoch = {}
        for t, _ in reversed(self.descriptors):
            if t not in self.timepoint_to_epoch:
                self.timepoint_to_epoch[t] = len(self.timepoint_to_epoch) + 1
        self.ancestor_data = formats.AncestorData(
            self.sample_data.sites_position[:][self.inference_site_ids],
            self.sample_data.sequence_length,
            path=self.ancestor_data_path,
            **self.ancestor_data_kwargs,
        )
        if self.num_ancestors > 0:
            logger.info(f"Starting build for {self.num_ancestors} ancestors")
            progress = self.progress_monitor.get("ga_generate", self.num_ancestors)
            a = np.zeros(self.num_sites, dtype=np.int8)
            root_time = max(self.timepoint_to_epoch.keys())
            av_timestep = root_time / len(self.timepoint_to_epoch)
            root_time += av_timestep  # Add a root a bit older than the oldest ancestor
            # Add an extra ancestor to act as a type of "virtual root" for the matching
            # algorithm: rather an awkward hack, but also allows the ancestor IDs to
            # line up. It's normally removed when processing the final tree sequence.
            self.ancestor_data.add_ancestor(
                start=0,
                end=self.num_sites,
                time=root_time + av_timestep,
                focal_sites=np.array([], dtype=np.int32),
                haplotype=a,
            )
            # This is the the "ultimate ancestor" of all zeros
            self.ancestor_data.add_ancestor(
                start=0,
                end=self.num_sites,
                time=root_time,
                focal_sites=np.array([], dtype=np.int32),
                haplotype=a,
            )
            if self.num_threads <= 0:
                self._run_synchronous(progress)
            else:
                self._run_threaded(progress)
            progress.close()
            logger.info("Finished building ancestors")
        if self.mmap_temp_file is not None:
            try:
                self.mmap_temp_file.close()
            except:  # noqa
                pass
        return self.ancestor_data


class Matcher:
    """
    A matching instance, used in both ``tsinfer.match_ancestors`` and
    ``tsinfer.match_samples``. For details of the ``path_compression``,
    `recombination_rate`` and ``mismatch_ratio`` parameters, see
    :ref:`matching ancestors & samples<sec_inference_match_ancestors_and_samples>`.

    Note that the ``recombination`` and ``mismatch`` parameters can be used in
    ``match_ancestors`` and ``match_samples`` and are passed directly to this
    function, but are deliberately not publicly documented in those methods.
    They are expected to be numpy arrays of length ``num_inference_sites - 1`` and
    ``num_inference_sites`` respectively, containing values between 0 and 1, and
    allow recombination and mismatch probabilities to be set directly. The
    ``recombination`` probabilities measure the probability of a recombination event
    between adjacent inference sites, used to calculate the HMM transition probabilities
    in the L&S-like matching algorithm. The ``mismatch`` probabilities are used
    to calculate the emission probabilities in the HMM. Note that values > 0.5 in
    the recombination and (particularly) the mutation arrays are likely to lead to
    pathological behaviour  - for example, a mismatch probability of 1 means that a
    mismatch is *required* at every site. For this reason, the probabilities
    created for recombination and mismatch when using the the public-facing
    ``recombination_rate`` and ``mismatch_ratio`` parameters are never > 0.5.
    TODO: include deliberately non-public details of precision here.
    """

    def __init__(
        self,
        sample_data,
        inference_site_position,
        num_threads=1,
        path_compression=True,
        recombination_rate=None,
        mismatch_ratio=None,
        recombination=None,
        mismatch=None,
        precision=None,
        extended_checks=False,
        engine=constants.C_ENGINE,
        progress_monitor=None,
        allow_multiallele=False,
    ):
        self.sample_data = sample_data
        self.num_threads = num_threads
        self.path_compression = path_compression
        self.num_samples = self.sample_data.num_samples
        self.num_sites = len(inference_site_position)
        if self.num_sites == 0:
            logger.warning("No sites used for inference")
        num_intervals = max(self.num_sites - 1, 0)
        self.progress_monitor = _get_progress_monitor(progress_monitor)
        self.match_progress = None  # Allocated by subclass
        self.extended_checks = extended_checks

        all_sites = self.sample_data.sites_position[:]
        index = np.searchsorted(all_sites, inference_site_position)
        num_alleles = sample_data.num_alleles()[index]
        if not np.all(all_sites[index] == inference_site_position):
            raise ValueError(
                "Site positions for inference must be a subset of those in "
                "the sample data file."
            )
        self.inference_site_id = index

        # Map of site index to tree sequence position. Bracketing
        # values of 0 and L are used for simplicity.
        self.position_map = np.hstack(
            [inference_site_position, [sample_data.sequence_length]]
        )
        self.position_map[0] = 0
        self.recombination = np.zeros(self.num_sites)  # TODO: reduce len by 1
        self.mismatch = np.zeros(self.num_sites)

        if recombination is not None or mismatch is not None:
            if recombination is None or mismatch is None:
                raise ValueError(
                    "Directly setting probabilities requires specifying "
                    "both 'recombination' and 'mismatch'"
                )
            if recombination_rate is not None or mismatch_ratio is not None:
                raise ValueError(
                    "Cannot simultaneously specify recombination & recombination_rate, "
                    "or mismatch and mismatch_ratio"
                )
            logger.info("Recombination and mismatch probabilities given by user")

        else:
            # Must set recombination and mismatch arrays
            if recombination_rate is None and mismatch_ratio is not None:
                raise ValueError("Cannot use mismatch without setting recombination")
            if (
                recombination_rate is None and mismatch_ratio is None
            ) or num_intervals == 0:
                # Special case: revert to tsinfer 0.1 behaviour with no mismatch allowed
                default_recombination_prob = 1e-2
                default_mismatch_prob = 1e-20  # Substantially < the value above
                recombination = np.full(num_intervals, default_recombination_prob)
                mismatch = np.full(self.num_sites, default_mismatch_prob)
                logger.info(
                    "Mismatch prevented by setting constant high recombination and "
                    + "low mismatch probabilities"
                )
            else:
                genetic_dists = self.recombination_rate_to_dist(
                    recombination_rate, inference_site_position
                )
                recombination = self.recombination_dist_to_prob(genetic_dists)
                if mismatch_ratio is None:
                    mismatch_ratio = 1.0
                mismatch = np.full(
                    self.num_sites,
                    self.mismatch_ratio_to_prob(
                        mismatch_ratio, np.median(genetic_dists), num_alleles
                    ),
                )
                logger.info(
                    "Recombination and mismatch probabilities calculated from "
                    + f"specified recomb rates with mismatch ratio = {mismatch_ratio}"
                )

        if len(recombination) != num_intervals:
            raise ValueError("Bad length for recombination array")
        if len(mismatch) != self.num_sites:
            raise ValueError("Bad length for mismatch array")
        if not (np.all(recombination >= 0) and np.all(recombination <= 1)):
            raise ValueError("Underlying recombination probabilities not between 0 & 1")
        if not (np.all(mismatch >= 0) and np.all(mismatch <= 1)):
            raise ValueError("Underlying mismatch probabilities not between 0 & 1")

        if precision is None:
            precision = 13
        self.recombination[1:] = recombination
        self.mismatch[:] = mismatch
        self.precision = precision

        if len(recombination) == 0:
            logger.info("Fewer than two inference sites: no recombination possible")
        else:
            logger.info(
                "Summary of recombination probabilities between sites: "
                f"min={np.min(recombination):.5g}; "
                f"max={np.max(recombination):.5g}; "
                f"median={np.median(recombination):.5g}; "
                f"mean={np.mean(recombination):.5g}"
            )

        if len(mismatch) == 0:
            logger.info("No inference sites: no mismatch possible")
        else:
            logger.info(
                "Summary of mismatch probabilities over sites: "
                f"min={np.min(mismatch):.5g}; "
                f"max={np.max(mismatch):.5g}; "
                f"median={np.median(mismatch):.5g}; "
                f"mean={np.mean(mismatch):.5g}"
            )
        logger.info(
            f"Matching using {precision} digits of precision in likelihood calcs"
        )

        if engine == constants.C_ENGINE:
            logger.debug("Using C matcher implementation")
            self.tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
            self.ancestor_matcher_class = _tsinfer.AncestorMatcher
        elif engine == constants.PY_ENGINE:
            logger.debug("Using Python matcher implementation")
            self.tree_sequence_builder_class = algorithm.TreeSequenceBuilder
            self.ancestor_matcher_class = algorithm.AncestorMatcher
        else:
            raise ValueError(f"Unknown engine:{engine}")
        self.tree_sequence_builder = None

        # Allocate 64K nodes and edges initially. This will double as needed and will
        # quickly be big enough even for very large instances.
        max_edges = 64 * 1024
        max_nodes = 64 * 1024
        if np.any(num_alleles > 2) and not allow_multiallele:
            # Currently only used for unsupported extend operation. We can
            # remove in future versions.
            raise ValueError("Cannot currently match with > 2 alleles.")
        self.tree_sequence_builder = self.tree_sequence_builder_class(
            num_alleles=num_alleles, max_nodes=max_nodes, max_edges=max_edges
        )
        logger.debug(f"Allocated tree sequence builder with max_nodes={max_nodes}")

    @staticmethod
    def find_path(matcher, child_id, haplotype, start, end):
        """
        Finds the path of the specified haplotype and returns the MatchResult object
        """
        missing = haplotype == tskit.MISSING_DATA
        match = np.full(len(haplotype), tskit.MISSING_DATA, dtype=np.int8)
        left, right, parent = matcher.find_path(haplotype, start, end, match)
        match[missing] = tskit.MISSING_DATA
        diffs = start + np.where(haplotype[start:end] != match[start:end])[0]
        derived_state = haplotype[diffs]

        result = MatchResult(
            node=child_id,
            path=Path(left=left, right=right, parent=parent),
            mutations_site=diffs.astype(np.int32),
            mutations_derived_state=derived_state,
            mean_traceback_size=matcher.mean_traceback_size,
        )

        logger.debug(
            "Matched node {}; "
            "num_edges={} tb_size={:.2f} match_mem={}".format(
                child_id,
                left.shape[0],
                matcher.mean_traceback_size,
                humanize.naturalsize(matcher.total_memory, binary=True),
            )
        )
        return result

    @staticmethod
    def recombination_rate_to_dist(rho, positions):
        """
        Return the mean number of recombinations between adjacent positions (i.e.
        the genetic distance in Morgans) given either a fixed rate or a RateMap
        """
        try:
            return np.diff(rho.get_cumulative_mass(positions))
        except AttributeError:
            return np.diff(positions) * rho

    @staticmethod
    def recombination_dist_to_prob(genetic_distances):
        """
        Convert genetic distances (in Morgans) to a probability of recombination,
        (i.e. an odd number of events) assuming a Poisson distribution,
        see Haldane, 1919 J. Genetics 8: 299-309. This maxes out at 0.5 as dist -> inf
        """
        return (1 - np.exp(-genetic_distances * 2)) / 2

    @staticmethod
    def mismatch_ratio_to_prob(ratio, genetic_distances, num_alleles=2):
        """
        Convert a mismatch ratio, relative to a genetic distance, to a probability
        of mismatch. A mismatch probability of 1 means that the emitted allele has a
        100% probability of being different from the allele implied by the hidden
        state. For all allele types to be emitted with equal probability, regardless
        of the copying haplotype, the mismatch probability should be set to
        1/num_alleles.

        For a small genetic_distance d, setting a ratio of X should give a
        probability of approximately X * r, where r is the recombination probability
        given by recombination_dist_to_prob(d)
        """
        return (1 - np.exp(-genetic_distances * ratio * num_alleles)) / num_alleles

    def create_matcher_instance(self):
        return self.ancestor_matcher_class(
            self.tree_sequence_builder,
            recombination=self.recombination,
            mismatch=self.mismatch,
            precision=self.precision,
            extended_checks=self.extended_checks,
        )

    def convert_inference_mutations(self, tables):
        """
        Convert the mutations stored in the tree sequence builder into the output
        format.
        """
        mut_site, node, derived_state, _ = self.tree_sequence_builder.dump_mutations()
        mutation_id = 0
        num_mutations = len(mut_site)
        progress = self.progress_monitor.get(
            "ms_full_mutations", len(self.inference_site_id)
        )
        schema = tables.sites.metadata_schema.schema
        for site in self.sample_data.sites(self.inference_site_id):
            metadata = _update_site_metadata(site.metadata, constants.INFERENCE_FULL)
            if schema is None:
                metadata = _encode_raw_metadata(metadata)
            site_id = tables.sites.add_row(
                site.position,
                ancestral_state=site.ancestral_state,
                metadata=metadata,
            )
            while mutation_id < num_mutations and mut_site[mutation_id] == site_id:
                tables.mutations.add_row(
                    site_id,
                    node=node[mutation_id],
                    derived_state=site.reorder_alleles()[derived_state[mutation_id]],
                )
                mutation_id += 1
            progress.update()
        progress.close()


class AncestorMatcher(Matcher):
    def __init__(self, sample_data, ancestor_data, time_units=None, **kwargs):
        super().__init__(sample_data, ancestor_data.sites_position[:], **kwargs)
        self.ancestor_data = ancestor_data
        if time_units is None:
            time_units = tskit.TIME_UNITS_UNCALIBRATED
        self.time_units = time_units
        self.num_ancestors = self.ancestor_data.num_ancestors
        self.epoch = self.ancestor_data.ancestors_time[:]

        # Add nodes for all the ancestors so that the ancestor IDs are equal
        # to the node IDs.
        for ancestor_id in range(self.num_ancestors):
            self.tree_sequence_builder.add_node(self.epoch[ancestor_id])
        self.ancestors = self.ancestor_data.ancestors()
        # Consume the "virtual root" ancestor
        virtual_root_ancestor = next(self.ancestors, None)
        self.num_epochs = 0
        if virtual_root_ancestor is not None:
            # assert np.array_equal(a.haplotype, np.zeros(self.num_sites, dtype=np.int8))
            # Create a list of all ID ranges in each epoch.
            breaks = np.where(self.epoch[1:] != self.epoch[:-1])[0]
            start = np.hstack([[0], breaks + 1])
            end = np.hstack([breaks + 1, [self.num_ancestors]])
            self.epoch_slices = np.vstack([start, end]).T
            self.num_epochs = self.epoch_slices.shape[0]
        self.start_epoch = 1

    def __epoch_info_dict(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        return collections.OrderedDict(
            [("epoch", str(self.epoch[start])), ("nanc", str(end - start))]
        )

    def __start_epoch(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        info = collections.OrderedDict(
            [("epoch", str(self.epoch[start])), ("nanc", str(end - start))]
        )
        self.progress_monitor.set_detail(info)
        self.tree_sequence_builder.freeze_indexes()

    def __complete_epoch(self, epoch_index, results):
        start, end = map(int, self.epoch_slices[epoch_index])
        num_ancestors_in_epoch = end - start
        nodes_before = self.tree_sequence_builder.num_nodes
        match_nodes_before = self.tree_sequence_builder.num_match_nodes

        for child_id, result in zip(range(start, end), results):
            assert result.node == child_id
            self.tree_sequence_builder.add_path(
                child_id,
                result.path.left,
                result.path.right,
                result.path.parent,
                compress=self.path_compression,
                extended_checks=self.extended_checks,
            )
            self.tree_sequence_builder.add_mutations(
                child_id, result.mutations_site, result.mutations_derived_state
            )

        extra_nodes = self.tree_sequence_builder.num_nodes - nodes_before
        assert (
            self.tree_sequence_builder.num_match_nodes
            == match_nodes_before + extra_nodes + num_ancestors_in_epoch
        )
        logger.debug(
            "Finished epoch index={} with {} ancestors; {} extra nodes inserted; "
            "mean_tb_size={:.2f} edges={};".format(
                epoch_index,
                num_ancestors_in_epoch,
                extra_nodes,
                sum(result.mean_traceback_size for result in results) / len(results),
                self.tree_sequence_builder.num_edges,
            )
        )

    def match_ancestors(self):
        logger.info(f"Starting ancestor matching for {self.num_epochs} epochs")
        self.match_progress = self.progress_monitor.get("ma_match", self.num_ancestors)

        local_data = threading.local()

        def worker_function(ancestor_ancestor_id):
            ancestor, ancestor_id = ancestor_ancestor_id
            assert ancestor.id == ancestor_id
            if not hasattr(local_data, "matcher"):
                local_data.matcher = self.create_matcher_instance()
            result = self.find_path(
                matcher=local_data.matcher,
                child_id=ancestor.id,
                haplotype=ancestor.full_haplotype,
                start=ancestor.start,
                end=ancestor.end,
            )
            return result

        for j in range(self.start_epoch, self.num_epochs):
            self.__start_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            if self.num_threads > 0:
                mapper = lambda func, args: threads.threaded_map(  # noqa E731
                    func, args, self.num_threads
                )
            else:
                mapper = map
            # We have to consume the iterator here, otherwise we'll match against
            # ancestors in this epoch
            results = []
            work_iter = (
                (ancestor, i + start)
                for i, ancestor in enumerate(
                    itertools.islice(self.ancestors, end - start)
                )
            )
            for result in mapper(worker_function, work_iter):
                results.append(result)
                self.match_progress.update()

            self.__complete_epoch(j, results)

        ts = self.store_output()
        self.match_progress.close()
        logger.info("Finished ancestor matching")
        return ts

    def get_ancestors_tables(self):
        """
        Return the ancestors tree sequence tables. Only inference sites are included in
        this tree sequence. All nodes have the sample flag bit set, and if a node
        corresponds to an ancestor in the ancestors file, it is indicated via metadata.
        """
        logger.debug("Building ancestors tree sequence")
        tsb = self.tree_sequence_builder

        tables = tskit.TableCollection(
            sequence_length=self.ancestor_data.sequence_length
        )

        flags, times = tsb.dump_nodes()
        pc_ancestors = is_pc_ancestor(flags)
        tables.nodes.set_columns(flags=flags, time=times)

        # # FIXME we should do this as a struct codec?
        # dict_schema = permissive_json_schema()
        # dict_schema = add_to_schema(dict_schema, "ancestor_data_id",
        #         {"type": "integer"})
        # schema = tskit.MetadataSchema(dict_schema)
        # tables.nodes.schema = schema

        # Add metadata for any non-PC node, pointing to the original ancestor
        metadata = []
        ancestor = 0
        for is_pc in pc_ancestors:
            if is_pc:
                metadata.append(b"")
            else:
                metadata.append(_encode_raw_metadata({"ancestor_data_id": ancestor}))
                ancestor += 1
        tables.nodes.packset_metadata(metadata)
        left, right, parent, child = tsb.dump_edges()
        tables.edges.set_columns(
            left=self.position_map[left],
            right=self.position_map[right],
            parent=parent,
            child=child,
        )

        self.convert_inference_mutations(tables)

        logger.debug("Sorting ancestors tree sequence")
        tables.sort()
        # Note: it's probably possible to compute the mutation parents from the
        # tsb data structures but we're not doing it for now.
        tables.build_index()
        tables.compute_mutation_parents()
        logger.debug("Sorting ancestors tree sequence done")
        logger.info(
            "Built ancestors tree sequence: {} nodes ({} pc ancestors); {} edges; "
            "{} sites; {} mutations".format(
                len(tables.nodes),
                np.sum(pc_ancestors),
                len(tables.edges),
                len(tables.mutations),
                len(tables.sites),
            )
        )
        return tables

    def store_output(self):
        if self.num_ancestors > 0:
            tables = self.get_ancestors_tables()
        else:
            # Allocate an empty tree sequence.
            tables = tskit.TableCollection(
                sequence_length=self.ancestor_data.sequence_length
            )
        tables.time_units = self.time_units
        return tables.tree_sequence()


class SampleMatcher(Matcher):
    def __init__(self, sample_data, ancestors_ts, **kwargs):
        self.ancestors_ts_tables = ancestors_ts.dump_tables()
        super().__init__(sample_data, self.ancestors_ts_tables.sites.position, **kwargs)
        self.restore_tree_sequence_builder()
        # Map from input sample indexes (IDs in the SampleData file) to the
        # node ID in the tree sequence.
        self.sample_id_map = {}

    def restore_tree_sequence_builder(self):
        tables = self.ancestors_ts_tables
        if self.sample_data.sequence_length != tables.sequence_length:
            raise ValueError(
                "Ancestors tree sequence not compatible: sequence length is different to"
                " sample data file."
            )
        if np.any(tables.nodes.time <= 0):
            raise ValueError("All nodes must have time > 0")

        edges = tables.edges
        # Get the indexes into the position array.
        left = np.searchsorted(self.position_map, edges.left)
        if np.any(self.position_map[left] != edges.left):
            raise ValueError("Invalid left coordinates")
        right = np.searchsorted(self.position_map, edges.right)
        if np.any(self.position_map[right] != edges.right):
            raise ValueError("Invalid right coordinates")

        # Need to sort by child ID here and left so that we can efficiently
        # insert the child paths.
        index = np.lexsort((left, edges.child))
        nodes = tables.nodes
        self.tree_sequence_builder.restore_nodes(nodes.time, nodes.flags)
        self.tree_sequence_builder.restore_edges(
            left[index].astype(np.int32),
            right[index].astype(np.int32),
            edges.parent[index],
            edges.child[index],
        )
        assert self.tree_sequence_builder.num_match_nodes == 1 + len(
            np.unique(edges.child)
        )

        mutations = tables.mutations
        derived_state = np.zeros(len(mutations), dtype=np.int8)
        mutation_site = mutations.site
        site_id = 0
        mutation_id = 0
        for site in self.sample_data.sites(self.inference_site_id):
            while (
                mutation_id < len(mutations) and mutation_site[mutation_id] == site_id
            ):
                allele = mutations[mutation_id].derived_state
                derived_state[mutation_id] = site.reorder_alleles().index(allele)
                mutation_id += 1
            site_id += 1
        self.tree_sequence_builder.restore_mutations(
            mutation_site, mutations.node, derived_state, mutations.parent
        )
        logger.info(
            "Loaded {} samples {} nodes; {} edges; {} sites; {} mutations".format(
                self.num_samples,
                len(nodes),
                len(edges),
                self.num_sites,
                len(mutations),
            )
        )

    def _match_samples(self, sample_indexes):
        num_samples = len(sample_indexes)
        builder = self.tree_sequence_builder
        _, times = builder.dump_nodes()
        logger.info(f"Started matching for {num_samples} samples")
        if self.num_sites > 0:
            self.match_progress = self.progress_monitor.get("ms_match", num_samples)
            local_data = threading.local()

            def worker_function(j_haplotype):
                j, haplotype = j_haplotype
                assert len(haplotype) == self.num_sites
                if not hasattr(local_data, "matcher"):
                    local_data.matcher = self.create_matcher_instance()
                result = self.find_path(
                    matcher=local_data.matcher,
                    child_id=self.sample_id_map[j],
                    haplotype=haplotype,
                    start=0,
                    end=self.num_sites,
                )
                self.match_progress.update()
                return result

            sample_haplotypes = self.sample_data.haplotypes(
                sample_indexes,
                sites=self.inference_site_id,
                recode_ancestral=True,
            )
            if self.num_threads > 0:
                mapper = lambda func, args: threads.threaded_map(  # noqa E731
                    func, args, self.num_threads
                )
            else:
                mapper = map
            # We have to consume the iterator here, otherwise we'll match against
            # ancestors in this epoch
            results = []
            for result in mapper(worker_function, sample_haplotypes):
                results.append(result)
                self.match_progress.update()

            self.match_progress.close()
            logger.info(
                "Inserting sample paths: {} edges in total".format(
                    sum(len(r.path.left) for r in results)
                )
            )
            progress_monitor = self.progress_monitor.get("ms_paths", num_samples)
            for j, result in zip(sample_indexes, results):
                node_id = int(self.sample_id_map[j])
                assert node_id == result.node
                if np.any(times[node_id] > times[result.path.parent]):
                    p = result.path.parent[np.argmin(times[result.path.parent])]
                    raise ValueError(
                        f"Failed to put sample {j} (node {node_id}) at time "
                        f"{times[node_id]} as it has a younger parent (node {p})."
                    )
                builder.add_path(
                    result.node,
                    result.path.left,
                    result.path.right,
                    result.path.parent,
                    compress=self.path_compression,
                )
                builder.add_mutations(
                    result.node,
                    result.mutations_site,
                    result.mutations_derived_state,
                )
                progress_monitor.update()
            progress_monitor.close()

    def match_samples(self, sample_indexes, sample_times=None):
        if sample_times is None:
            sample_times = np.zeros(len(sample_indexes))
        builder = self.tree_sequence_builder
        for j, t in zip(sample_indexes, sample_times):
            self.sample_id_map[j] = builder.add_node(t)

        self._match_samples(sample_indexes)

    def finalise(self):
        logger.info("Finalising tree sequence")
        ts = self.get_samples_tree_sequence()
        # Check that there are the same number of samples as expected
        assert len(self.sample_id_map) == ts.num_samples
        return ts

    def get_samples_tree_sequence(self, map_additional_sites=True):
        """
        Returns the current state of the build tree sequence. Sample nodes will have the
        sample node flag set and be in the same order as passed the order of
        sample_indexes passed to match_samples. For correct sample reconstruction,
        the non-inference sites also need to be placed into the resulting tree sequence.
        """
        tsb = self.tree_sequence_builder
        tables = self.ancestors_ts_tables.copy()

        schema = self.sample_data.metadata_schema
        tables.metadata_schema = tskit.MetadataSchema(schema)
        tables.metadata = self.sample_data.metadata

        schema = self.sample_data.populations_metadata_schema
        if schema is not None:
            tables.populations.metadata_schema = tskit.MetadataSchema(schema)
        for metadata in self.sample_data.populations_metadata[:]:
            if schema is None:
                # Use the default json encoding to avoid breaking old code.
                tables.populations.add_row(_encode_raw_metadata(metadata))
            else:
                tables.populations.add_row(metadata)

        schema = self.sample_data.individuals_metadata_schema
        if schema is not None:
            schema = add_to_schema(
                schema,
                "sample_data_time",
                definition=sample_data_time_metadata_definition,
            )
            tables.individuals.metadata_schema = tskit.MetadataSchema(schema)

        num_ancestral_individuals = len(tables.individuals)
        for ind in self.sample_data.individuals():
            metadata = ind.metadata
            if ind.time != 0:
                metadata["sample_data_time"] = ind.time
            if schema is None:
                metadata = _encode_raw_metadata(ind.metadata)
            tables.individuals.add_row(
                location=ind.location,
                metadata=metadata,
                flags=ind.flags,
            )

        logger.debug("Adding tree sequence nodes")
        flags, times = tsb.dump_nodes()
        num_pc_ancestors = count_pc_ancestors(flags)

        # All true ancestors are samples in the ancestors tree sequence. We unset
        # the SAMPLE flag but keep other flags intact.
        new_flags = np.bitwise_and(tables.nodes.flags, ~tskit.NODE_IS_SAMPLE)
        tables.nodes.flags = new_flags.astype(np.uint32)
        sample_ids = list(self.sample_id_map.values())
        assert len(tables.nodes) == sample_ids[0]
        individuals_population = self.sample_data.individuals_population[:]
        samples_individual = self.sample_data.samples_individual[:]
        individuals_time = self.sample_data.individuals_time[:]
        for index, sample_id in self.sample_id_map.items():
            individual = samples_individual[index]
            if individuals_time[individual] != 0:
                flags[sample_id] = np.bitwise_or(
                    flags[sample_id], constants.NODE_IS_HISTORICAL_SAMPLE
                )
            population = individuals_population[individual]
            tables.nodes.add_row(
                flags=flags[sample_id],
                time=times[sample_id],
                population=population,
                individual=num_ancestral_individuals + individual,
            )
        # Add in the remaining non-sample nodes.
        for u in range(sample_ids[-1] + 1, tsb.num_nodes):
            tables.nodes.add_row(flags=flags[u], time=times[u])

        logger.debug("Adding tree sequence edges")
        tables.edges.clear()
        left, right, parent, child = tsb.dump_edges()
        if self.num_sites == 0:
            # We have no inference sites, so no edges have been estimated. To ensure
            # we have a rooted tree, we add in edges for each sample to an artificial
            # root.
            assert left.shape[0] == 0
            max_node_time = tables.nodes.time.max()
            root = tables.nodes.add_row(flags=0, time=max_node_time + 1)
            ultimate = tables.nodes.add_row(flags=0, time=max_node_time + 2)
            tables.edges.add_row(0, tables.sequence_length, ultimate, root)
            for sample_id in sample_ids:
                tables.edges.add_row(0, tables.sequence_length, root, sample_id)
        else:
            tables.edges.set_columns(
                left=self.position_map[left],
                right=self.position_map[right],
                parent=parent,
                child=child,
            )

        logger.debug("Sorting and building intermediate tree sequence.")
        tables.sites.clear()
        tables.mutations.clear()
        tables.sort()

        schema = self.sample_data.sites_metadata_schema
        if schema is not None:
            schema = add_to_schema(
                schema,
                "inference_type",
                definition=inference_type_metadata_definition,
            )
            tables.sites.metadata_schema = tskit.MetadataSchema(schema)
        self.convert_inference_mutations(tables)

        # FIXME this is a shortcut. We should be computing the mutation parent above
        # during insertion (probably)
        tables.build_index()
        tables.compute_mutation_parents()

        logger.info(
            "Built samples tree sequence: {} nodes ({} pc); {} edges; "
            "{} sites; {} mutations".format(
                len(tables.nodes),
                num_pc_ancestors,
                len(tables.edges),
                len(tables.sites),
                len(tables.mutations),
            )
        )

        ts = tables.tree_sequence()
        num_additional_sites = self.sample_data.num_sites - self.num_sites
        if map_additional_sites and num_additional_sites > 0:
            logger.info("Mapping additional sites")
            assert np.array_equal(ts.samples(), list(self.sample_id_map.values()))
            ts = insert_missing_sites(
                self.sample_data,
                ts,
                sample_id_map=np.array(list(self.sample_id_map.keys())),
                progress_monitor=self.progress_monitor,
            )
        else:
            logger.info("Skipping additional site mapping")

        return ts

    def get_augmented_ancestors_tree_sequence(self, sample_indexes):
        """
        Return the ancestors tree sequence augmented with samples as extra ancestors.
        """
        logger.debug("Building augmented ancestors tree sequence")
        tsb = self.tree_sequence_builder
        tables = self.ancestors_ts_tables.copy()

        flags, times = tsb.dump_nodes()
        s = 0
        num_pc_ancestors = 0
        for j in range(len(tables.nodes), len(flags)):
            if times[j] == 0.0:
                # This is an augmented ancestor node.
                tables.nodes.add_row(
                    flags=constants.NODE_IS_SAMPLE_ANCESTOR,
                    time=times[j],
                    metadata=_encode_raw_metadata(
                        {"sample_data_id": int(sample_indexes[s])}
                    ),
                )
                s += 1
            else:
                # This is a path compressed node
                tables.nodes.add_row(flags=flags[j], time=times[j])
                assert is_pc_ancestor(flags[j])
                num_pc_ancestors += 1
        assert s == len(sample_indexes)
        assert len(tables.nodes) == len(flags)

        # Increment the time for all nodes so the augmented samples are no longer
        # at timepoint 0.
        tables.nodes.time = tables.nodes.time + 1

        # TODO - check this works for augmented ancestors with missing data
        left, right, parent, child = tsb.dump_edges()
        tables.edges.set_columns(
            left=self.position_map[left],
            right=self.position_map[right],
            parent=parent,
            child=child,
        )

        tables.sites.clear()
        tables.mutations.clear()
        self.convert_inference_mutations(tables)

        logger.debug("Sorting ancestors tree sequence")
        tables.sort()
        logger.debug("Sorting ancestors tree sequence done")
        logger.info(
            "Augmented ancestors tree sequence: {} nodes ({} extra pc ancestors); "
            "{} edges; {} sites; {} mutations".format(
                len(tables.nodes),
                num_pc_ancestors,
                len(tables.edges),
                len(tables.mutations),
                len(tables.sites),
            )
        )
        return tables.tree_sequence()


@dataclasses.dataclass
class Path:
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray


@dataclasses.dataclass
class MatchResult:
    node: int
    path: Path
    mutations_site: list
    mutations_derived_state: list
    mean_traceback_size: int


def has_single_edge_over_grand_root(ts):
    # Internal function to check if this is a "raw" inferred tree sequence.
    if ts.num_edges < 2:
        # must have edge to grand root and above grand root
        return False
    last_edge = ts.edge(-1)
    if last_edge.left != 0 or last_edge.right != ts.sequence_length:
        return False  # Not a single edge spanning the entire genome
    if ts.edge(-2).parent == last_edge.parent:
        return False  # other edges point to the oldest node => not a virtual-like root
    return True


def has_same_root_everywhere(ts):
    roots = set()
    for tree in ts.trees():
        if not tree.has_single_root:
            return False
        roots.add(tree.root)
        if len(roots) > 1:
            return False
    return True


def post_process(
    ts,
    *,
    split_ultimate=None,
    erase_flanks=None,
    # Parameters below deliberately undocumented
    warn_if_unexpected_format=None,
    simplify_only=None,
):
    """
    post_process(ts, *, split_ultimate=None, erase_flanks=None)

    Post-process a tsinferred tree sequence into a more conventional form. This is
    the function run by default on the final tree sequence output by
    :func:`match_samples`. It involves the following 4 steps:

    #. If the oldest node is connected to a single child via an edge that spans the
       entire tree sequence, this oldest node is removed, so that its child becomes
       the new root (this step is undertaken to remove the "virtual-root-like node"
       which is added to ancestor tree sequences to enable matching).
    #. If the oldest node is removed in the first step and the new root spans the
       entire genome, it is treated as the "ultimate ancestor" and (unless
       ``split_ultimate`` is ``False``) the node is split into multiple coexisiting
       nodes with the splits occurring whenever the children of the ultimate ancestor
       change. The rationale is that tsinfer creates a single ancestral haplotype with
       all inference sites in the ancestral state: this is, however, unlikely to
       represent a single ancestor in the past. If the tree sequence is then dated,
       the fact that ultimate ancestor is split into separate nodes allows these nodes
       to be dated to different times.
    #. Often, extensive regions of genome exist before the first defined site and after
       the last defined site. Since no data exists in these sections of the genome, post
       processing by default erases the inferred topology in these regions. However,
       if ``erase_flanks`` is False, the flanking regions at the start and end will be
       assigned the same topology as inferred at the first and last site respectively.
    #. The sample nodes are reordered such that they are the first nodes listed in the
       node table,  removing tree nodes and edges that are not on a path between the
       root and any of the samples (by applying the :meth:`~tskit.TreeSequence.simplify`
       method with ``keep_unary`` set to True but ``filter_sites``,
       ``filter_populations`` and ``filter_individuals`` set to False).

    :param bool split_ultimate: If ``True`` (default) and the oldest node is the only
        parent to a single "ultimate ancestor" node, attempt to split this node into
        many separate nodes (see above). If ``False`` do not attempt to identify or
        split an ultimate ancestor node.
    :param bool erase_flanks: If ``True`` (default), keep only the
        inferred topology between the first and last sites. If ``False``,
        output regions of topology inferred before the first site and after
        the last site.
    :return: The post-processed tree sequence.
    :rtype: tskit.TreeSequence
    """
    if split_ultimate is None:
        split_ultimate = True
    if erase_flanks is None:
        erase_flanks = True

    tables = ts.dump_tables()

    if not simplify_only:
        if has_single_edge_over_grand_root(ts):
            logger.info(
                "Removing the oldest edge to detach the virtual-root-like ancestor"
            )
            last_edge = ts.edge(-1)  # Edge with oldest parent is last in the edge table
            tables.edges.truncate(tables.edges.num_rows - 1)

            # move any mutations above the virtual-root-like ancestor to above the
            # ultimate ancestor instead (these will be mutations placed by parsimony)
            mutations_node = tables.mutations.node
            mutations_node[mutations_node == last_edge.parent] = last_edge.child
            tables.mutations.node = mutations_node

            if split_ultimate:
                split_ultimate_ancestor(tables, warn_if_unexpected_format)
        elif warn_if_unexpected_format:
            logger.warning(
                "Cannot find a virtual-root-like ancestor during preprocessing"
            )

        if erase_flanks and ts.num_sites > 0:
            # So that the last site falls within a tree, we must add one to the
            # site position (or simply extend to the end of the ts)
            keep_max = min(ts.sites_position[-1] + 1, ts.sequence_length)
            tables.keep_intervals(
                [[ts.sites_position[0], keep_max]],
                simplify=False,
                record_provenance=False,
            )
            erased = ts.sites_position[0] + ts.sequence_length - keep_max
            erased *= 100 / ts.sequence_length
            logger.info(
                f"Erased flanks covering {erased}% of the genome: "
                f"{ts.sites_position[0]} units at the start and "
                f"{ts.sequence_length - keep_max} units at the end"
            )

    logger.info(
        "Simplifying with filter_sites=False, filter_populations=False, "
        "filter_individuals=False, and keep_unary=True on "
        f"{tables.nodes.num_rows} nodes and {tables.edges.num_rows} edges"
    )
    # NB: if this is an inferred TS, match_samples is guaranteed to produce samples
    # in the same order as passed in to sample_indexes, and simplification will
    # simply stick all those at the start but keep their order the same
    tables.simplify(
        filter_sites=False,
        filter_populations=False,
        filter_individuals=False,
        keep_unary=True,
        record_provenance=False,
    )
    logger.info(
        "Finished simplify; now have {} nodes and {} edges".format(
            tables.nodes.num_rows, tables.edges.num_rows
        )
    )
    return tables.tree_sequence()


def _post_process(*args, **kwargs):
    return post_process(*args, **kwargs)


def split_ultimate_ancestor(tables, warn_if_unexpected_format=None):
    # Internal function: if a single oldest node is a root across the entire genome,
    # split it up into a set of contemporaneous nodes whenever the node children change

    ts = tables.tree_sequence()
    if not has_same_root_everywhere(ts):
        if warn_if_unexpected_format:
            logger.warning("Cannot find a single contiguous ultimate ancestor to split")
        return

    # Split into multiple contemporaneous nodes whenever the node children change
    genomewide_ultimate_ancestor_id = ts.edge(-1).parent
    genomewide_ultimate_ancestor = ts.node(genomewide_ultimate_ancestor_id)
    logger.info("Located the all zeros ultimate ancestor")
    root_breaks = set()
    edges = tables.edges
    j = len(edges) - 1  # the last edges are the ones connecting to the genomewide UA
    while j >= 0 and edges[j].parent == genomewide_ultimate_ancestor_id:
        root_breaks |= {edges[j].left, edges[j].right}
        j -= 1
    root_breaks = sorted(root_breaks)
    assert root_breaks[0] == 0
    if root_breaks[1] == tables.sequence_length:
        # Only a single edge: no splitting needed
        return

    logger.info(f"Splitting ultimate ancestor into {len(root_breaks) - 1} nodes")
    # detach the ultimate ancestor from all its children, so it can be simplified out
    tables.edges.truncate(j + 1)

    # Move the mutations above the ultimate ancestor to the new nodes
    mutation_ids = np.where(tables.mutations.node == genomewide_ultimate_ancestor_id)[0]
    mutation_positions = tables.sites.position[tables.mutations.site[mutation_ids]]
    mut_iter = zip(mutation_ids, mutation_positions)
    mutation_id, mutation_pos = next(mut_iter, (None, ts.sequence_length))

    # Go through the trees, making a new root node whereever we hit a root_break
    # and recreating the edges to the children each time
    trees_iter = ts.trees()
    tree = next(trees_iter)
    left = root_breaks[0]
    for right in root_breaks[1:]:
        while tree.interval.right != right:
            tree = next(trees_iter)
        new_root = tables.nodes.append(genomewide_ultimate_ancestor)
        for c in tree.children(genomewide_ultimate_ancestor_id):
            tables.edges.add_row(parent=new_root, child=c, left=left, right=right)
        while mutation_pos < right:
            tables.mutations[mutation_id] = tables.mutations[mutation_id].replace(
                node=new_root
            )
            mutation_id, mutation_pos = next(mut_iter, (None, ts.sequence_length))
        left = right
    tables.sort()


def minimise(ts):
    """
    Returns a tree sequence with the minimal information required to represent
    the tree topologies at its sites.

    This is a convenience function used when we wish to use a subset of the
    sites in a tree sequence for ancestor matching. It is a thin-wrapper
    over the simplify method.
    """
    return ts.simplify(
        reduce_to_site_topology=True,
        filter_sites=False,
        filter_individuals=False,
        filter_populations=False,
    )
