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
Central module for high-level inference. The actual implementation of
of the core tasks like ancestor generation and matching are delegated
to other modules.
"""
import collections
import queue
import time
import logging
import threading
import json
import heapq

import numpy as np
import humanize
import tskit

import _tsinfer
import tsinfer.formats as formats
import tsinfer.algorithm as algorithm
import tsinfer.threads as threads
import tsinfer.provenance as provenance
import tsinfer.constants as constants

logger = logging.getLogger(__name__)


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
    return np.sum(np.bitwise_and(flags, constants.NODE_IS_PC_ANCESTOR) != 0)


def count_srb_ancestors(flags):
    """
    Returns the number of values in the specified array which have the
    NODE_IS_SRB_ANCESTOR set.
    """
    flags = np.array(flags, dtype=np.uint32, copy=False)
    return np.sum(np.bitwise_and(flags, constants.NODE_IS_SRB_ANCESTOR) != 0)


class DummyProgress(object):
    """
    Class that mimics the subset of the tqdm API that we use in this module.
    """
    def update(self):
        pass

    def close(self):
        pass


class DummyProgressMonitor(object):
    """
    Simple class to mimic the interface of the real progress monitor.
    """
    def get(self, key, total):
        return DummyProgress()

    def set_detail(self, info):
        pass


def _get_progress_monitor(progress_monitor):
    if progress_monitor is None:
        progress_monitor = DummyProgressMonitor()
    return progress_monitor


def verify(samples, tree_sequence, progress_monitor=None):
    """
    verify(samples, tree_sequence)

    Verifies that the specified sample data and tree sequence files encode the
    same data.

    :param SampleData samples: The input :class:`SampleData` instance
        representing the observed data that we wish to compare to.
    :param TreeSequence tree_sequence: The input :class:`tskit.TreeSequence`
        instance an encoding of the specified samples that we wish to verify.
    """
    progress_monitor = _get_progress_monitor(progress_monitor)
    if samples.num_sites != tree_sequence.num_sites:
        raise ValueError("numbers of sites not equal")
    if samples.num_samples != tree_sequence.num_samples:
        raise ValueError("numbers of samples not equal")
    if samples.sequence_length != tree_sequence.sequence_length:
        raise ValueError("Sequence lengths not equal")
    progress = progress_monitor.get("verify", tree_sequence.num_sites)
    for var1, var2 in zip(samples.variants(), tree_sequence.variants()):
        if var1.site.position != var2.site.position:
            raise ValueError("site positions not equal: {} != {}".format(
                var1.site.position, var2.site.position))
        if var1.alleles != var2.alleles:
            raise ValueError("alleles not equal: {} != {}".format(
                var1.alleles, var2.alleles))
        if not np.array_equal(var1.genotypes, var2.genotypes):
            raise ValueError("Genotypes not equal at site {}".format(var1.site.id))
        progress.update()
    progress.close()


def infer(
        sample_data, num_threads=0, path_compression=True, simplify=True,
        recombination_rate=None, mutation_rate=None, precision=None,
        engine=constants.C_ENGINE, progress_monitor=None):
    """
    infer(sample_data, num_threads=0, path_compression=True, simplify=True)

    Runs the full :ref:`inference pipeline <sec_inference>` on the specified
    :class:`SampleData` instance and returns the inferred
    :class:`tskit.TreeSequence`.

    :param SampleData sample_data: The input :class:`SampleData` instance
        representing the observed data that we wish to make inferences from.
    :param int num_threads: The number of worker threads to use in parallelised
        sections of the algorithm. If <= 0, do not spawn any threads and
        use simpler sequential algorithms (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param bool simplify: Whether to remove extra tree nodes and edges that are not
        on a path between the root and any of the samples. To do so, the final tree
        sequence is simplified by appling the :meth:`tskit.TreeSequence.simplify` method
        with ``keep_unary`` set to True (default = ``True``).
    :returns: The :class:`tskit.TreeSequence` object inferred from the
        input sample data.
    :rtype: tskit.TreeSequence
    """
    ancestor_data = generate_ancestors(
        sample_data, num_threads=num_threads,
        engine=engine, progress_monitor=progress_monitor)
    # NOTE: we set mutation rate = 0 here when matching ancestors, since we're
    # assuming that the ancestors we estimate are probably good. Seems likely
    # that we don't want to use the same mu for ancestor matching and
    # sample matching anyway --- something to nail down for the final interface.
    ancestors_ts = match_ancestors(
        sample_data, ancestor_data, engine=engine, num_threads=num_threads,
        recombination_rate=recombination_rate, mutation_rate=0,
        precision=precision,
        path_compression=path_compression, progress_monitor=progress_monitor)
    inferred_ts = match_samples(
        sample_data, ancestors_ts, engine=engine, num_threads=num_threads,
        recombination_rate=recombination_rate, mutation_rate=mutation_rate,
        precision=precision,
        path_compression=path_compression, simplify=simplify,
        progress_monitor=progress_monitor)
    return inferred_ts


def generate_ancestors(
        sample_data, num_threads=0, path=None,
        engine=constants.C_ENGINE, progress_monitor=None, **kwargs):
    """
    generate_ancestors(sample_data, num_threads=0, path=None, **kwargs)

    Runs the ancestor generation :ref:`algorithm <sec_inference_generate_ancestors>`
    on the specified :class:`SampleData` instance and returns the resulting
    :class:`AncestorData` instance. If you wish to store generated ancestors
    persistently on file you must pass the ``path`` keyword argument to this
    function. For example,

    .. code-block:: python

        ancestor_data = tsinfer.generate_ancestors(sample_data, path="mydata.ancestors")

    Other keyword arguments are passed to the :class:`AncestorData` constructor,
    which may be used to control the storage properties of the generated file.

    :param SampleData sample_data: The :class:`SampleData` instance that we are
        genering putative ancestors from.
    :param int num_threads: The number of worker threads to use. If < 1, use a
        simpler synchronous algorithm.
    :param str path: The path of the file to store the sample data. If None,
        the information is stored in memory and not persistent.
    :rtype: AncestorData
    :returns: The inferred ancestors stored in an :class:`AncestorData` instance.
    """
    sample_data._check_finalised()
    progress_monitor = _get_progress_monitor(progress_monitor)
    with formats.AncestorData(sample_data, path=path, **kwargs) as ancestor_data:
        generator = AncestorsGenerator(
            sample_data, ancestor_data, num_threads=num_threads,
            engine=engine, progress_monitor=progress_monitor)
        generator.add_sites()
        generator.run()
        ancestor_data.record_provenance("generate-ancestors")
    return ancestor_data


def match_ancestors(
        sample_data, ancestor_data, num_threads=0, path_compression=True,
        recombination_rate=None, mutation_rate=None, precision=None,
        extended_checks=False, engine=constants.C_ENGINE, progress_monitor=None):
    """
    match_ancestors(sample_data, ancestor_data, num_threads=0, path_compression=True)

    Runs the ancestor matching :ref:`algorithm <sec_inference_match_ancestors>`
    on the specified :class:`SampleData` and :class:`AncestorData` instances,
    returning the resulting :class:`tskit.TreeSequence` representing the
    complete ancestry of the putative ancestors.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param AncestorData ancestor_data: The :class:`AncestorData` instance
        representing the set of ancestral haplotypes that we are finding
        a history for.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :return: The ancestors tree sequence representing the inferred history
        of the set of ancestors.
    :rtype: tskit.TreeSequence
    """
    sample_data._check_finalised()
    ancestor_data._check_finalised()
    matcher = AncestorMatcher(
        sample_data, ancestor_data, num_threads=num_threads,
        recombination_rate=recombination_rate, mutation_rate=mutation_rate,
        precision=precision,
        path_compression=path_compression, extended_checks=extended_checks,
        engine=engine, progress_monitor=progress_monitor)
    return matcher.match_ancestors()


def augment_ancestors(
        sample_data, ancestors_ts, indexes, num_threads=0, path_compression=True,
        recombination_rate=None, mutation_rate=None, precision=None,
        extended_checks=False, engine=constants.C_ENGINE, progress_monitor=None):
    """
    augment_ancestors(sample_data, ancestors_ts, indexes, num_threads=0,\
        path_compression=True)

    Runs the sample matching :ref:`algorithm <sec_inference_match_samples>`
    on the specified :class:`SampleData` instance and ancestors tree sequence,
    for the specified subset of sample indexes, returning the
    :class:`tskit.TreeSequence` instance including these samples. This
    tree sequence can then be used as an ancestors tree sequence for subsequent
    matching against all samples.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param tskit.TreeSequence ancestors_ts: The
        :class:`tskit.TreeSequence` instance representing the inferred
        history among ancestral ancestral haplotypes.
    :param array indexes: The sample indexes to insert into the ancestors
        tree sequence.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :return: The specified ancestors tree sequence augmented with copying
        paths for the specified sample.
    :rtype: tskit.TreeSequence
    """
    sample_data._check_finalised()
    manager = SampleMatcher(
        sample_data, ancestors_ts, num_threads=num_threads,
        recombination_rate=recombination_rate, mutation_rate=mutation_rate,
        precision=precision,
        path_compression=path_compression, extended_checks=extended_checks,
        engine=engine, progress_monitor=progress_monitor
        )
    manager.match_samples(indexes)
    ts = manager.get_augmented_ancestors_tree_sequence(indexes)
    return ts


def match_samples(
        sample_data, ancestors_ts, num_threads=0, path_compression=True, simplify=True,
        recombination_rate=None, mutation_rate=None, precision=None,
        extended_checks=False, stabilise_node_ordering=False, engine=constants.C_ENGINE,
        progress_monitor=None):
    """
    match_samples(sample_data, ancestors_ts, num_threads=0, path_compression=True,\
        simplify=True)

    Runs the sample matching :ref:`algorithm <sec_inference_match_samples>`
    on the specified :class:`SampleData` instance and ancestors tree sequence,
    returning the final :class:`tskit.TreeSequence` instance containing
    the full inferred history for all samples and sites.

    :param SampleData sample_data: The :class:`SampleData` instance
        representing the input data.
    :param tskit.TreeSequence ancestors_ts: The
        :class:`tskit.TreeSequence` instance representing the inferred
        history among ancestral ancestral haplotypes.
    :param int num_threads: The number of match worker threads to use. If
        this is <= 0 then a simpler sequential algorithm is used (default).
    :param bool path_compression: Whether to merge edges that share identical
        paths (essentially taking advantage of shared recombination breakpoints).
    :param bool simplify: Whether to remove extra tree nodes and edges that are not
        on a path between the root and any of the samples. To do so, the final tree
        sequence is simplified by appling the :meth:`tskit.TreeSequence.simplify` method
        with ``keep_unary`` set to True (default = ``True``).
    :return: The tree sequence representing the inferred history
        of the sample.
    :rtype: tskit.TreeSequence
    """
    sample_data._check_finalised()
    manager = SampleMatcher(
        sample_data, ancestors_ts, num_threads=num_threads,
        recombination_rate=recombination_rate, mutation_rate=mutation_rate,
        precision=precision,
        path_compression=path_compression, extended_checks=extended_checks,
        engine=engine, progress_monitor=progress_monitor)
    manager.match_samples()
    ts = manager.finalise(
        simplify=simplify, stabilise_node_ordering=stabilise_node_ordering)
    return ts


class AncestorsGenerator(object):
    """
    Manages the process of building ancestors.
    """
    def __init__(
            self, sample_data, ancestor_data, num_threads=0,
            engine=constants.C_ENGINE, progress_monitor=None):
        self.sample_data = sample_data
        self.ancestor_data = ancestor_data
        self.progress_monitor = progress_monitor
        self.num_sites = sample_data.num_inference_sites
        self.num_samples = sample_data.num_samples
        self.num_threads = num_threads
        if engine == constants.C_ENGINE:
            logger.debug("Using C AncestorBuilder implementation")
            self.ancestor_builder = _tsinfer.AncestorBuilder(
                self.num_samples, self.num_sites)
        elif engine == constants.PY_ENGINE:
            logger.debug("Using Python AncestorBuilder implementation")
            self.ancestor_builder = algorithm.AncestorBuilder(
                self.num_samples, self.num_sites)
        else:
            raise ValueError("Unknown engine:{}".format(engine))

    def add_sites(self):
        """
        Add all sites from the sample_data object into the ancestor builder.
        """
        logger.info("Starting addition of {} sites".format(self.num_sites))
        progress = self.progress_monitor.get("ga_add_sites", self.num_sites)
        for j, variant in enumerate(self.sample_data.variants(inference_sites=True)):
            self.ancestor_builder.add_site(j, variant.site.time, variant.genotypes)
            progress.update()
        progress.close()
        logger.info("Finished adding sites")

    def _run_synchronous(self, progress):
        a = np.zeros(self.num_sites, dtype=np.int8)
        for t, focal_sites in self.descriptors:
            before = time.perf_counter()
            s, e = self.ancestor_builder.make_ancestor(focal_sites, a)
            duration = time.perf_counter() - before
            logger.debug(
                "Made ancestor in {:.2f}s at timepoint {} (epoch {}) "
                "from {} to {} (len={}) with {} focal sites ({})".format(
                    duration, t, self.timepoint_to_epoch[t], s, e, e - s,
                    focal_sites.shape[0], focal_sites))
            self.ancestor_data.add_ancestor(
                start=s, end=e, time=t, focal_sites=focal_sites,
                haplotype=a[s:e])
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
                    start=s, end=e, time=t, focal_sites=focal_sites, haplotype=haplotype)
                progress.update()
                next_add_index += 1
                num_drained += 1
            logger.debug("Drained {} ancestors from add queue".format(num_drained))

        def build_worker(thread_index):
            a = np.zeros(self.num_sites, dtype=np.int8)
            while True:
                work = build_queue.get()
                if work is None:
                    break
                index, t, focal_sites = work
                start, end = self.ancestor_builder.make_ancestor(focal_sites, a)
                with add_lock:
                    haplotype = a[start: end].copy()
                    heapq.heappush(
                        add_queue,
                        (index, t, focal_sites, start, end, haplotype))
                    drain_add_queue()
                build_queue.task_done()
            build_queue.task_done()

        build_threads = [
            threads.queue_consumer_thread(
                build_worker, build_queue, name="build-worker-{}".format(j),
                index=j)
            for j in range(self.num_threads)]
        logger.debug("Started {} build worker threads".format(self.num_threads))

        for index, (t, focal_sites) in enumerate(self.descriptors):
            build_queue.put((index, t, focal_sites))

        # Stop the the worker threads.
        for j in range(self.num_threads):
            build_queue.put(None)
        for j in range(self.num_threads):
            build_threads[j].join()
        drain_add_queue()

    def run(self):
        self.descriptors = self.ancestor_builder.ancestor_descriptors()
        self.num_ancestors = len(self.descriptors)
        # Maps epoch numbers to their corresponding ancestor times.
        self.timepoint_to_epoch = {}
        for t, _ in reversed(self.descriptors):
            if t not in self.timepoint_to_epoch:
                self.timepoint_to_epoch[t] = len(self.timepoint_to_epoch) + 1
        if self.num_ancestors > 0:
            logger.info("Starting build for {} ancestors".format(self.num_ancestors))
            progress = self.progress_monitor.get("ga_generate", self.num_ancestors)
            a = np.zeros(self.num_sites, dtype=np.int8)
            root_time = max(self.timepoint_to_epoch.keys()) + 1
            ultimate_ancestor_time = root_time + 1
            # Add the ultimate ancestor. This is an awkward hack really; we don't
            # ever insert this ancestor. The only reason to add it here is that
            # it makes sure that the ancestor IDs we have in the ancestor file are
            # the same as in the ancestor tree sequence. This seems worthwhile.
            self.ancestor_data.add_ancestor(
                start=0, end=self.num_sites, time=ultimate_ancestor_time,
                focal_sites=[], haplotype=a)
            # Hack to ensure we always have a root with zeros at every position.
            self.ancestor_data.add_ancestor(
                start=0, end=self.num_sites, time=root_time,
                focal_sites=np.array([], dtype=np.int32), haplotype=a)
            if self.num_threads <= 0:
                self._run_synchronous(progress)
            else:
                self._run_threaded(progress)
            progress.close()
            logger.info("Finished building ancestors")


class Matcher(object):

    def __init__(
            self, sample_data, tables, num_threads=1, path_compression=True,
            recombination_rate=None, mutation_rate=None, precision=None,
            extended_checks=False, engine=constants.C_ENGINE, progress_monitor=None):
        self.sample_data = sample_data
        self.num_threads = num_threads
        self.path_compression = path_compression
        self.num_samples = self.sample_data.num_samples
        self.num_sites = self.sample_data.num_inference_sites
        self.progress_monitor = _get_progress_monitor(progress_monitor)
        self.match_progress = None  # Allocated by subclass
        self.extended_checks = extended_checks

        if precision is None:
            # For now, default to high precision where round does nothing.
            precision = 100

        if recombination_rate is None:
            # TODO is this a good value? I guess it depends on the default precision
            # as well. Will need to tune
            recombination_rate = 1e-8

        self.recombination_rate = np.zeros(self.num_sites)
        # FIXME not quite right: we should check the rho[0] = 0
        self.recombination_rate[:] = recombination_rate
        if mutation_rate is None:
            mutation_rate = 0
        self.mutation_rate = np.zeros(self.num_sites)
        self.mutation_rate[:] = mutation_rate
        self.precision = precision

        max_edges = 64 * 1024
        max_nodes = 64 * 1024
        if engine == constants.C_ENGINE:
            logger.debug("Using C matcher implementation")
            # self.tree_sequence_builder_class = _tsinfer.TreeSequenceBuilder
            self.ancestor_matcher_class = _tsinfer.AncestorMatcher
            # Using Python always for now.
            self.tables = tables
            self.tree_sequence_builder_class = algorithm.TreeSequenceBuilder
            self.ancestor_matcher_class = algorithm.AncestorMatcher
        elif engine == constants.PY_ENGINE:
            logger.debug("Using Python matcher implementation")
            self.tables = tables
            self.tree_sequence_builder_class = algorithm.TreeSequenceBuilder
            self.ancestor_matcher_class = algorithm.AncestorMatcher
        else:
            raise ValueError("Unknown engine:{}".format(engine))
        self.tree_sequence_builder = None

        # Allocate 64K nodes and edges initially. This will double as needed and will
        # quickly be big enough even for very large instances.
        # Use the alleles list to signal inference and non-inference sites.
        alleles = self.sample_data.sites_alleles[:]
        alleles[self.sample_data.sites_inference[:] == 0] = None

        # NOTE We should probably require that the edges are sorted by
        # child ID and left so we can insert efficiently. Easier than doing
        # this in C. e.g.:
        # index = np.lexsort((left, edges.child))
        # # edges.set_columns(
        # #     left=left[index],
        # #     right=right[index],
        # #     parent=edges.parent[index],
        # #     child=edges.child[index])

        # Might as well allocate the tsb in the conditional  above. Might need
        # to use different classes for the table collection
        self.tree_sequence_builder = self.tree_sequence_builder_class(
                self.tables, alleles, max_nodes, max_edges)

        logger.debug("Allocated tree sequence builder with max_nodes={}".format(
            max_nodes))

        # self.tree_sequence_builder.print_state()

        # Allocate the matchers and statistics arrays.
        num_threads = max(1, self.num_threads)
        self.match = [np.zeros(self.num_sites, np.int8) for _ in range(num_threads)]
        self.results = ResultBuffer()
        self.mean_traceback_size = np.zeros(num_threads)
        self.num_matches = np.zeros(num_threads)
        self.matcher = [
            self.ancestor_matcher_class(
                self.tree_sequence_builder,
                recombination_rate=self.recombination_rate,
                mutation_rate=self.mutation_rate,
                precision=precision,
                extended_checks=self.extended_checks)
            for _ in range(num_threads)]

    def _find_path(self, child_id, haplotype, start, end, thread_index=0):
        """
        Finds the path of the specified haplotype and upates the results
        for the specified thread_index.
        """
        matcher = self.matcher[thread_index]
        match = self.match[thread_index]
        left, right, parent = matcher.find_path(haplotype, start, end, match)
        self.results.set_path(child_id, left, right, parent)
        self.match_progress.update()
        self.mean_traceback_size[thread_index] += matcher.mean_traceback_size
        self.num_matches[thread_index] += 1
        logger.debug("matched node {}; num_edges={} tb_size={:.2f} match_mem={}".format(
            child_id, left.shape[0], matcher.mean_traceback_size,
            humanize.naturalsize(matcher.total_memory, binary=True)))
        return left, right, parent

    # def get_ancestors_tree_sequence(self):
    #     """
    #     Return the ancestors tree sequence. In this tree sequence the coordinates
    #     are measured in units of site indexes and all ancestral and derived states
    #     are 0/1. All nodes have the sample flag bit set.
    #     """
    #     logger.debug("Building ancestors tree sequence")
    #     tsb = self.tree_sequence_builder
    #     lwt = _tsinfer.LightweightTableCollection()
    #     tsb.dump(lwt)
    #     tables_dict = lwt.asdict()
    #     tables_dict["sequence_length"] = self.ancestor_data.sequence_length
    #     tables = tskit.TableCollection.fromdict(tables_dict)
    #     num_pc_ancestors = count_pc_ancestors(tables.nodes.flags)

    #     position = self.ancestor_data.sites_position
    #     pos_map = np.hstack([position, [tables.sequence_length]])
    #     pos_map[0] = 0
    #     tables.edges.set_columns(
    #         left=pos_map[tables.edges.left.astype(int)],
    #         right=pos_map[tables.edges.right.astype(int)],
    #         parent=tables.edges.parent, child=tables.edges.child)

    #     tables.sites.set_columns(
    #         position=position,
    #         ancestral_state=tables.sites.ancestral_state,
    #         ancestral_state_offset=tables.sites.ancestral_state_offset)
    #     tables.mutations.set_columns(
    #         site=tables.mutations.site, node=tables.mutations.node,
    #         derived_state=tables.mutations.derived_state,
    #         derived_state_offset=tables.mutations.derived_state_offset,
    #         parent=tables.mutations.parent)
    #     for timestamp, record in self.ancestor_data.provenances():
    #         tables.provenances.add_row(timestamp=timestamp, record=json.dumps(record))
    #     record = provenance.get_provenance_dict(
    #         command="match_ancestors",
    #         source={"uuid": self.ancestor_data.uuid})
    #     tables.provenances.add_row(record=json.dumps(record))
    #     logger.debug("Sorting ancestors tree sequence")
    #     tables.sort()
    #     logger.debug("Sorting ancestors tree sequence done")
    #     logger.info(
    #         "Built ancestors tree sequence: {} nodes ({} pc ancestors); {} edges; "
    #         "{} sites; {} mutations".format(
    #             len(tables.nodes), num_pc_ancestors, len(tables.edges),
    #             len(tables.mutations), len(tables.sites)))
    #     tables = tskit.TableCollection.fromdict(
    #         self.tree_sequence_builder.tables.asdict())
    #     tables.sort()
    #     return tables.tree_sequence()

    def encode_metadata(self, value):
        return json.dumps(value).encode()

    # def insert_sites(self, tables, tsb_tables):
    #     """
    #     Insert the sites in the sample data that were not marked for inference,
    #     updating the specified site and mutation tables. This is done by
    #     iterating over the trees
    #     """
    #     # NOTE: This is all quite confusing, but there's not much point in cleaning
    #     # it up, when we'll be using tskit to work more directly with the
    #     # tables.
    #     num_sites = self.sample_data.num_sites
    #     num_non_inference_sites = self.sample_data.num_non_inference_sites
    #     progress_monitor = self.progress_monitor.get("ms_sites", num_sites)

    #     # NOTE: we're passing in the tsb_tables as an intermediate step towards
    #     # removing the different table representations. The code is still making
    #     # assumptions about the derived state, etc, which we need to change.
    #     site_id = tsb_tables.mutations.site
    #     node = tsb_tables.mutations.node
    #     derived_state = tsb_tables.mutations.derived_state - ord("0")
    #     ts = tables.tree_sequence()
    #     if num_non_inference_sites > 0:
    #         assert ts.num_edges > 0
    #         logger.info(
    #             "Starting mutation positioning for {} non inference sites".format(
    #                 num_non_inference_sites))
    #         inferred_mutation = 0
    #         inferred_site = 0
    #         trees = ts.trees()
    #         tree = next(trees)
    #         for variant in self.sample_data.variants():
    #             site = variant.site
    #             predefined_anc_state = site.ancestral_state
    #             while tree.interval[1] <= site.position:
    #                 tree = next(trees)
    #             assert tree.interval[0] <= site.position < tree.interval[1]
    #             tables.sites.add_row(
    #                 position=site.position,
    #                 ancestral_state=predefined_anc_state,
    #                 metadata=self.encode_metadata(site.metadata))
    #             if site.inference == 1:
    #                 while (
    #                         inferred_mutation < len(site_id) and
    #                         site_id[inferred_mutation] == inferred_site):
    #                     tables.mutations.add_row(
    #                         site=site.id, node=node[inferred_mutation],
    #                         derived_state=variant.alleles[
    #                             derived_state[inferred_mutation]])
    #                     inferred_mutation += 1
    #                 inferred_site += 1
    #             else:
    #                 inferred_anc_state, mapped_mutations = tree.map_mutations(
    #                     variant.genotypes, variant.alleles)
    #                 if inferred_anc_state != predefined_anc_state:
    #                     # We need to set the ancestral state to that defined in the
    #                     # original file
    #                     for root_node in tree.roots:
    #                         # Add a transition at each root to the mapped value
    #                         tables.mutations.add_row(
    #                             site=site.id, node=root_node,
    #                             derived_state=inferred_anc_state)
    #                 for mutation in mapped_mutations:
    #                     tables.mutations.add_row(
    #                         site=site.id, node=mutation.node,
    #                         derived_state=mutation.derived_state)
    #             progress_monitor.update()
    #     else:
    #         # Simple case where all sites are inference sites. We save a lot of
    #         # time here by not decoding the genotypes.
    #         logger.info("Inserting detailed site information")
    #         position = self.sample_data.sites_position[:]
    #         alleles = self.sample_data.sites_alleles[:]
    #         metadata = self.sample_data.sites_metadata[:]
    #         k = 0
    #         for j in range(self.num_sites):
    #             tables.sites.add_row(
    #                 position=position[j],
    #                 ancestral_state=alleles[j][0],
    #                 metadata=self.encode_metadata(metadata[j]))
    #             while k < len(site_id) and site_id[k] == j:
    #                 tables.mutations.add_row(
    #                     site=j, node=node[k], derived_state=alleles[j][
    #                           derived_state[k]])
    #                 k += 1
    #             progress_monitor.update()
    #     progress_monitor.close()

    def get_augmented_ancestors_tree_sequence(self, sample_indexes):
        """
        Return the ancestors tree sequence augmented with samples as extra ancestors.
        """
        logger.debug("Building augmented ancestors tree sequence")

        # tsb = self.tree_sequence_builder
        # tables = self.ancestors_ts.dump_tables()
        # num_pc_ancestors = count_pc_ancestors(tables.nodes.flags)

        # lwt = _tsinfer.LightweightTableCollection()
        # tsb.dump(lwt)
        # tsb_tables = tskit.TableCollection.fromdict(lwt.asdict())

        # flags = tsb_tables.nodes.flags
        # times = tsb_tables.nodes.time
        # s = 0
        # for j in range(len(tables.nodes), len(flags)):
        #     if times[j] == 0.0:
        #         # This is an augmented ancestor node.
        #         tables.nodes.add_row(
        #             flags=constants.NODE_IS_SAMPLE_ANCESTOR,
        #             time=times[j],
        #             metadata=self.encode_metadata({"sample": int(sample_indexes[s])}))
        #         s += 1
        #     else:
        #         tables.nodes.add_row(flags=flags[j], time=times[j])
        # assert s == len(sample_indexes)
        # assert len(tables.nodes) == len(flags)

        # # Increment the time for all nodes so the augmented samples are no longer
        # # at timepoint 0.
        # tables.nodes.set_columns(
        #     flags=tables.nodes.flags,
        #     time=tables.nodes.time + 1,
        #     population=tables.nodes.population,
        #     individual=tables.nodes.individual,
        #     metadata=tables.nodes.metadata,
        #     metadata_offset=tables.nodes.metadata_offset)
        # num_pc_ancestors = count_pc_ancestors(tables.nodes.flags) - num_pc_ancestors

        # position = tables.sites.position
        # pos_map = np.hstack([position, [tables.sequence_length]])
        # pos_map[0] = 0
        # tables.edges.set_columns(
        #     left=pos_map[tsb_tables.edges.left.astype(int)],
        #     right=pos_map[tsb_tables.edges.right.astype(int)],
        #     parent=tsb_tables.edges.parent,
        #     child=tsb_tables.edges.child)

        # tables.mutations.set_columns(
        #     site=tsb_tables.mutations.site,
        #     node=tsb_tables.mutations.node,
        #     derived_state=tsb_tables.mutations.derived_state,
        #     derived_state_offset=tsb_tables.mutations.derived_state_offset,
        #     parent=tsb_tables.mutations.parent)
        record = provenance.get_provenance_dict(command="augment_ancestors")
        self.tables.provenances.add_row(record=json.dumps(record))
        logger.debug("Sorting ancestors tree sequence")
        self.tables.sort()
        logger.debug("Sorting ancestors tree sequence done")
        # logger.info(
        #     "Augmented ancestors tree sequence: {} nodes ({} extra pc ancestors); "
        #     "{} edges; {} sites; {} mutations".format(
        #         len(tables.nodes), num_pc_ancestors, len(tables.edges),
        #         len(tables.mutations), len(tables.sites)))
        return self.tables.tree_sequence()

    def add_non_inference_sites(self, ts):
        """
        Adds mutations for non-inference sites to the tables.
        """
        tree = ts.first()
        for variant in self.sample_data.variants(inference_sites=False):
            while tree.interval[1] < variant.site.position:
                tree.next()
            site = ts.site(variant.site.id)
            inferred_anc_state, mapped_mutations = tree.map_mutations(
                variant.genotypes, variant.alleles)
            if inferred_anc_state != site.ancestral_state:
                for root_node in tree.roots:
                    # Add a transition at each root to the mapped value so that
                    self.tables.mutations.add_row(
                        site=site.id, node=root_node,
                        derived_state=inferred_anc_state)
            for mutation in mapped_mutations:
                self.tables.mutations.add_row(
                    site=site.id, node=mutation.node,
                    derived_state=mutation.derived_state)
        # TODO we should be iterating over the full list of mutations, keeping
        # track of the parent values.
        self.tables.sort()

    def get_samples_tree_sequence(self):
        """
        Returns the current state of the build tree sequence. All samples and
        ancestors will have the sample node flag set.
        """

        non_inference_sites = ~self.sample_data.sites_inference[:]

        flags = self.tables.nodes.flags
        num_pc_ancestors = count_pc_ancestors(flags)

        # All true ancestors are samples in the ancestors tree sequence. We unset
        # the SAMPLE flag but keep other flags intact.
        new_flags = np.bitwise_and(flags, ~tskit.NODE_IS_SAMPLE)
        new_flags[self.sample_ids] = tskit.NODE_IS_SAMPLE
        self.tables.nodes.flags = new_flags.astype(np.uint32)

        # logger.debug("Adding tree sequence edges")
        if np.all(non_inference_sites):
            # We have no inference sites, so no edges have been estimated. To ensure
            # we have a rooted tree, we add in edges for each sample to an artificial
            # root.
            assert len(self.tables.edges) == 0
            root = self.tables.nodes.add_row(
                flags=0, time=self.tables.nodes.time.max() + 1)
            for sample_id in self.sample_ids:
                self.tables.edges.add_row(
                    0, self.tables.sequence_length, root, sample_id)

        logger.debug("Sorting and building intermediate tree sequence.")
        self.tables.sort()
        # print(self.tables.populations())
        ts = self.tables.tree_sequence()
        if np.any(non_inference_sites):
            self.add_non_inference_sites(ts)

        # We don't have a source here because tree sequence files don't have a
        # UUID yet.
        record = provenance.get_provenance_dict(command="match-samples")
        self.tables.provenances.add_row(record=json.dumps(record))

        logger.info(
            "Built samples tree sequence: {} nodes ({} pc); {} edges; "
            "{} sites; {} mutations".format(
                len(self.tables.nodes),
                num_pc_ancestors,
                len(self.tables.edges),
                len(self.tables.sites),
                len(self.tables.mutations)))
        return self.tables.tree_sequence()


def initial_ancestor_tables(sample_data, ancestor_data):
    tables = ancestor_data.as_tables(sample_data)

    ancestor_time = ancestor_data.ancestors_time[:]
    for ancestor_id in range(ancestor_data.num_ancestors):
        node_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE, time=ancestor_time[ancestor_id])
        assert ancestor_id == node_id

    return tables

class AncestorMatcher(Matcher):

    def __init__(self, sample_data, ancestor_data, **kwargs):
        tables = initial_ancestor_tables(sample_data, ancestor_data)
        super().__init__(sample_data, tables, **kwargs)
        if not np.all(self.mutation_rate == 0):
            raise ValueError("mutations not supported for ancestor matching yet")

        self.num_ancestors = ancestor_data.num_ancestors
        self.epoch = ancestor_data.ancestors_time[:]

        self.ancestors = ancestor_data.ancestors()
        # Consume the first ancestor.
        a = next(self.ancestors, None)
        self.num_epochs = 0
        if a is not None:
            assert np.array_equal(a.haplotype, np.zeros(self.num_sites, dtype=np.int8))
            # Create a list of all ID ranges in each epoch.
            breaks = np.where(self.epoch[1:] != self.epoch[:-1])[0]
            start = np.hstack([[0], breaks + 1])
            end = np.hstack([breaks + 1, [self.num_ancestors]])
            self.epoch_slices = np.vstack([start, end]).T
            self.num_epochs = self.epoch_slices.shape[0]
        self.start_epoch = 1

    def __epoch_info_dict(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        return collections.OrderedDict([
            ("epoch", str(self.epoch[start])),
            ("nanc", str(end - start))
        ])

    def __ancestor_find_path(self, ancestor, thread_index=0):
        haplotype = np.full(self.num_sites, tskit.MISSING_DATA, dtype=np.int8)
        focal_sites = ancestor.focal_sites
        start = ancestor.start
        end = ancestor.end
        self.results.set_mutations(ancestor.id, focal_sites)
        assert ancestor.haplotype.shape[0] == (end - start)
        haplotype[start: end] = ancestor.haplotype
        assert np.all(haplotype[focal_sites] == 1)
        logger.debug(
            "Finding path for ancestor {}; start={} end={} num_focal_sites={}".format(
                ancestor.id, start, end, focal_sites.shape[0]))
        haplotype[focal_sites] = 0
        left, right, parent = self._find_path(
                ancestor.id, haplotype, start, end, thread_index)
        assert np.all(self.match[thread_index][start: end] == haplotype[start: end])

    def __start_epoch(self, epoch_index):
        start, end = self.epoch_slices[epoch_index]
        info = collections.OrderedDict([
            ("epoch", str(self.epoch[start])),
            ("nanc", str(end - start))
        ])
        self.progress_monitor.set_detail(info)
        self.tree_sequence_builder.freeze_indexes()

    def __complete_epoch(self, epoch_index):
        start, end = map(int, self.epoch_slices[epoch_index])
        num_ancestors_in_epoch = end - start
        current_time = self.epoch[start]
        nodes_before = self.tree_sequence_builder.num_nodes

        for child_id in range(start, end):
            left, right, parent = self.results.get_path(child_id)
            self.tree_sequence_builder.add_path(
                child_id, left, right, parent,
                compress=self.path_compression,
                extended_checks=self.extended_checks)
            site, derived_state = self.results.get_mutations(child_id)
            self.tree_sequence_builder.add_mutations(child_id, site, derived_state)

        extra_nodes = self.tree_sequence_builder.num_nodes - nodes_before
        mean_memory = np.mean([matcher.total_memory for matcher in self.matcher])
        logger.debug(
            "Finished epoch {} with {} ancestors; {} extra nodes inserted; "
            "mean_tb_size={:.2f} edges={}; mean_matcher_mem={}".format(
                current_time, num_ancestors_in_epoch, extra_nodes,
                np.sum(self.mean_traceback_size) / np.sum(self.num_matches),
                self.tree_sequence_builder.num_edges,
                humanize.naturalsize(mean_memory, binary=True)))
        self.mean_traceback_size[:] = 0
        self.num_matches[:] = 0
        self.results.clear()

    def __match_ancestors_single_threaded(self):
        for j in range(self.start_epoch, self.num_epochs):
            self.__start_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            for ancestor_id in range(start, end):
                a = next(self.ancestors)
                assert ancestor_id == a.id
                self.__ancestor_find_path(a)
            self.__complete_epoch(j)

    def __match_ancestors_multi_threaded(self, start_epoch=1):
        # See note on match samples multithreaded below. Should combine these
        # into a single function. Possibly when trying to make the thread
        # error handling more robust.
        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        match_queue = queue.Queue(queue_depth)

        def match_worker(thread_index):
            while True:
                work = match_queue.get()
                if work is None:
                    break
                self.__ancestor_find_path(work, thread_index)
                match_queue.task_done()
            match_queue.task_done()

        match_threads = [
            threads.queue_consumer_thread(
                match_worker, match_queue, name="match-worker-{}".format(j),
                index=j)
            for j in range(self.num_threads)]
        logger.debug("Started {} match worker threads".format(self.num_threads))

        for j in range(self.start_epoch, self.num_epochs):
            self.__start_epoch(j)
            start, end = map(int, self.epoch_slices[j])
            for ancestor_id in range(start, end):
                a = next(self.ancestors)
                assert a.id == ancestor_id
                match_queue.put(a)
            # Block until all matches have completed.
            match_queue.join()
            self.__complete_epoch(j)

        # Stop the the worker threads.
        for j in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def match_ancestors(self):
        logger.info("Starting ancestor matching for {} epochs".format(self.num_epochs))
        self.match_progress = self.progress_monitor.get("ma_match", self.num_ancestors)
        if self.num_threads <= 0:
            self.__match_ancestors_single_threaded()
        else:
            self.__match_ancestors_multi_threaded()
        ts = self.store_output()
        self.match_progress.close()
        logger.info("Finished ancestor matching")
        return ts

    def store_output(self):
        self.tree_sequence_builder.flush_edges()
        record = provenance.get_provenance_dict(command="match-ancestors")
        self.tables.provenances.add_row(record=json.dumps(record))

        self.tables.sort()
        ts = self.tables.tree_sequence()
        return ts


class SampleMatcher(Matcher):

    def __init__(self, sample_data, ancestors_ts, **kwargs):
        tables = ancestors_ts.dump_tables()
        self.sample_ids = sample_data.add_samples(tables)
        super().__init__(sample_data, tables, **kwargs)
        self.ancestors_ts = ancestors_ts

    def __process_sample(self, sample_id, haplotype, thread_index=0):
        self._find_path(sample_id, haplotype, 0, self.num_sites, thread_index)
        match = self.match[thread_index]
        diffs = np.where(haplotype != match)[0]
        derived_state = haplotype[diffs]
        self.results.set_mutations(sample_id, diffs.astype(np.int32), derived_state)

    def __match_samples_single_threaded(self, indexes):
        sample_haplotypes = self.sample_data.haplotypes(indexes, inference_sites=True)
        for j, a in sample_haplotypes:
            self.__process_sample(self.sample_ids[j], a)

    def __match_samples_multi_threaded(self, indexes):
        # Note that this function is not almost identical to the match_ancestors
        # multithreaded function above. All we need to do is provide a function
        # to do the matching and some producer for the actual items and we
        # can bring this into a single function.

        queue_depth = 8 * self.num_threads  # Seems like a reasonable limit
        match_queue = queue.Queue(queue_depth)

        def match_worker(thread_index):
            while True:
                work = match_queue.get()
                if work is None:
                    break
                sample_id, a = work
                self.__process_sample(sample_id, a, thread_index)
                match_queue.task_done()
            match_queue.task_done()

        match_threads = [
            threads.queue_consumer_thread(
                match_worker, match_queue, name="match-worker-{}".format(j), index=j)
            for j in range(self.num_threads)]
        logger.debug("Started {} match worker threads".format(self.num_threads))

        sample_haplotypes = self.sample_data.haplotypes(indexes, inference_sites=True)
        for j, a in sample_haplotypes:
            match_queue.put((self.sample_ids[j], a))

        # Stop the the worker threads.
        for j in range(self.num_threads):
            match_queue.put(None)
        for j in range(self.num_threads):
            match_threads[j].join()

    def match_samples(self, indexes=None):
        if indexes is None:
            indexes = np.arange(self.num_samples)
        # # Add in sample nodes.
        # for j in indexes:
        #     self.sample_ids[j] = self.tree_sequence_builder.add_node(0)

        logger.info("Started matching for {} samples".format(len(indexes)))
        if self.sample_data.num_inference_sites > 0:
            self.match_progress = self.progress_monitor.get("ms_match", len(indexes))
            if self.num_threads <= 0:
                self.__match_samples_single_threaded(indexes)
            else:
                self.__match_samples_multi_threaded(indexes)
            self.match_progress.close()
            logger.info("Inserting sample paths: {} edges in total".format(
                self.results.total_edges))
            progress_monitor = self.progress_monitor.get("ms_paths", len(indexes))
            for j in indexes:
                sample_id = int(self.sample_ids[j])
                left, right, parent = self.results.get_path(sample_id)
                self.tree_sequence_builder.add_path(
                    sample_id, left, right, parent, compress=self.path_compression)
                site, derived_state = self.results.get_mutations(sample_id)
                self.tree_sequence_builder.add_mutations(sample_id, site, derived_state)
                progress_monitor.update()
            progress_monitor.close()

    def finalise(self, simplify=True, stabilise_node_ordering=False):
        logger.info("Finalising tree sequence")
        # self.tables.sort()
        # ts = self.tables.tree_sequence()
        self.tree_sequence_builder.flush_edges()
        ts = self.get_samples_tree_sequence()
        # print(ts.tables)
        if simplify:
            logger.info(
                "Running simplify(keep_unary=True) on {} nodes and {} edges".format(
                    ts.num_nodes, ts.num_edges))
            if stabilise_node_ordering:
                # Ensure all the node times are distinct so that they will have
                # stable IDs after simplifying. This could possibly also be done
                # by reversing the IDs within a time slice. This is used for comparing
                # tree sequences produced by perfect inference.
                tables = ts.dump_tables()
                times = tables.nodes.time
                for t in range(1, int(times[0])):
                    index = np.where(times == t)[0]
                    k = index.shape[0]
                    times[index] += np.arange(k)[::-1] / k
                tables.nodes.set_columns(flags=tables.nodes.flags, time=times)
                tables.sort()
                ts = tables.tree_sequence()
            ts = ts.simplify(
                samples=self.sample_ids, filter_sites=False, keep_unary=True)
            logger.info("Finished simplify; now have {} nodes and {} edges".format(
                ts.num_nodes, ts.num_edges))
        return ts


class ResultBuffer(object):
    """
    A wrapper for numpy arrays representing the results of a copying operations.
    """
    def __init__(self):
        self.paths = {}
        self.mutations = {}
        self.lock = threading.Lock()
        self.total_edges = 0

    def clear(self):
        """
        Clears this result buffer.
        """
        self.paths.clear()
        self.mutations.clear()
        self.total_edges = 0

    def set_path(self, node_id, left, right, parent):
        with self.lock:
            assert node_id not in self.paths
            self.paths[node_id] = left, right, parent
            self.total_edges += len(left)

    def set_mutations(self, node_id, site, derived_state=None):
        if derived_state is None:
            derived_state = np.ones(site.shape[0], dtype=np.int8)
        with self.lock:
            self.mutations[node_id] = site, derived_state

    def get_path(self, node_id):
        return self.paths[node_id]

    def get_mutations(self, node_id):
        return self.mutations[node_id]


def minimise(ts):
    """
    Returns a tree sequence with the minimal information required to represent
    the tree topologies at its sites.

    This is a convenience function used when we wish to use a subset of the
    sites in a tree sequence for ancestor matching. It is a thin-wrapper
    over the simplify method.
    """
    return ts.simplify(reduce_to_site_topology=True, filter_sites=False)
