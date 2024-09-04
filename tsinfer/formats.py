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
Manage tsinfer's various file formats.
"""
import collections
import collections.abc as abc
import datetime
import functools
import itertools
import json
import logging
import os.path
import queue
import sys
import threading
import warnings
from typing import Union  # noqa: F401

import attr
import humanize
import lmdb
import numcodecs
import numpy as np
import tskit
import zarr
from tskit import MISSING_DATA

import tsinfer.exceptions as exceptions
import tsinfer.provenance as provenance
import tsinfer.threads as threads


logger = logging.getLogger(__name__)


FORMAT_NAME_KEY = "format_name"
FORMAT_VERSION_KEY = "format_version"
FINALISED_KEY = "finalised"

# We use the zstd compressor because it allows for compression of buffers
# bigger than 2GB, which can occur in a larger instances.
DEFAULT_COMPRESSOR = numcodecs.Zstd()

# Lmdb on windows allocates the entire file size rather than
# growing dynamically (see https://github.com/mozilla/lmdb-rs/issues/40).
# For the default setting on windows, we therefore hard code a smaller
# map_size of 1GiB to avoid filling up disk space. On other platforms where
# sparse files are supported, we default to 1TiB.
DEFAULT_MAX_FILE_SIZE = 2**30 if sys.platform == "win32" else 2**40


def np_obj_equal(np_obj_array1, np_obj_array2):
    """
    A replacement for np.array_equal to test equality of numpy arrays that
    contain objects, as used e.g. for metadata, location, alleles, etc.
    """
    if np_obj_array1.shape != np_obj_array2.shape:
        return False
    return all(itertools.starmap(np.array_equal, zip(np_obj_array1, np_obj_array2)))


def exclude_id(attribute, value):
    """
    Used to filter out the id field from attrs objects such as Ancestor
    """
    return attribute.name != "id"


def exclude_id_and_full_haplotype(attribute, value):
    """
    Used to filter out the id field from attrs objects such as Ancestor
    """
    return attribute.name not in ["id", "full_haplotype"]


def open_lmbd_readonly(path):
    # We set the mapsize here because LMBD will map 1TB of virtual memory if
    # we don't, making it hard to figure out how much memory we're actually
    # using.
    map_size = None
    try:
        map_size = os.path.getsize(path)
    except OSError as e:
        raise exceptions.FileFormatError(str(e)) from e
    try:
        store = zarr.LMDBStore(
            path, map_size=map_size, readonly=True, subdir=False, lock=False
        )
    except lmdb.InvalidError as e:
        raise exceptions.FileFormatError(f"Unknown file format:{str(e)}") from e
    except lmdb.Error as e:
        raise exceptions.FileFormatError(str(e)) from e
    return store


def remove_lmdb_lockfile(lmdb_file):
    lockfile = lmdb_file + "-lock"
    if os.path.exists(lockfile):
        os.unlink(lockfile)


class BufferedItemWriter:
    """
    Class that writes items sequentially into a set of zarr arrays,
    buffering writes and flushing them to the destination arrays
    asynchronosly using threads.
    """

    def __init__(self, array_map, num_threads=0):
        self.chunk_size = -1
        for key, array in array_map.items():
            chunked_dimension = 1 if "full_haplotype" in key else 0
            if self.chunk_size == -1:
                self.chunk_size = array.chunks[chunked_dimension]
            else:
                if array.chunks[chunked_dimension] != self.chunk_size:
                    raise ValueError("Chunk sizes must be equal")

        self.arrays = array_map
        if num_threads <= 0:
            # Use a syncronous algorithm.
            self.num_threads = 0
            self.num_buffers = 1
        else:
            # One buffer for each thread. Buffers are referred to by their indexes.
            self.num_buffers = num_threads
            self.num_threads = num_threads
        self.buffers = {}
        self.current_size = 0
        self.total_items = 0
        for key, array in self.arrays.items():
            self.buffers[key] = [None for _ in range(self.num_buffers)]
            np_array = array[:]
            shape = list(array.shape)
            chunked_dimension = 1 if "full_haplotype" in key else 0
            shape[chunked_dimension] = self.chunk_size
            for j in range(self.num_buffers):
                self.buffers[key][j] = np.empty_like(np_array)
                self.buffers[key][j].resize(*shape)
                # We need to initialise the buffers for the arrays where only the extent
                # of the ancestor is written
                if key == "full_haplotype":
                    self.buffers[key][j][...] = MISSING_DATA
                elif key == "full_haplotype_mask":
                    self.buffers[key][j][...] = True

            # Make sure the destination array is zero sized at the start.
            shape[chunked_dimension] = 0
            array.resize(*shape)

        self.start_offset = [0 for _ in range(self.num_buffers)]
        self.num_buffered_items = [0 for _ in range(self.num_buffers)]
        self.write_buffer = 0
        # This lock must be held when resizing the underlying arrays.
        # This is no-op when using a single-threaded algorithm, but it's
        # not worth removing and complicating the logic.
        self.resize_lock = threading.Lock()
        if self.num_threads > 0:
            # Buffer indexes are placed in the queues. The current write buffer
            # is obtained from the write_queue. Flush worker threads pull buffer
            # indexes from the flush queue, and push them back on to the write
            # queue when the buffer has been flushed.
            self.write_queue = queue.Queue()
            self.flush_queue = queue.Queue()
            # The initial write buffer is 0; place the others on the queue.
            for j in range(1, self.num_buffers):
                self.write_queue.put(j)
            # Make the flush threads.
            self.flush_threads = [
                threads.queue_consumer_thread(
                    self._flush_worker,
                    self.flush_queue,
                    name=f"flush-worker-{j}",
                )
                for j in range(self.num_threads)
            ]
            logger.info(f"Started {self.num_threads} flush worker threads")

    def _commit_write_buffer(self, write_buffer):
        start = self.start_offset[write_buffer]
        n = self.num_buffered_items[write_buffer]
        if n == 0:
            return
        end = start + n
        logger.debug(f"Flushing buffer {write_buffer}: start={start} n={n}")
        with self.resize_lock:
            if self.current_size < end:
                self.current_size = end
                for key, array in self.arrays.items():
                    shape = list(array.shape)
                    chunked_dimension = 1 if "full_haplotype" in key else 0
                    shape[chunked_dimension] = self.current_size
                    array.resize(*shape)
        for key, array in self.arrays.items():
            if "full_haplotype" in key:
                buffered = self.buffers[key][write_buffer][:, :n]
                array[:, start:end] = buffered
            else:
                buffered = self.buffers[key][write_buffer][:n]
                array[start:end] = buffered
            if key == "full_haplotype":
                self.buffers[key][write_buffer][...] = MISSING_DATA
            elif key == "full_haplotype_mask":
                self.buffers[key][write_buffer][...] = True

        logger.debug(f"Buffer {write_buffer} flush done")

    def _flush_worker(self, thread_index):
        """
        Thread worker responsible for flushing buffers. Read a buffer index
        from flush_queue and write it to disk. Push the index back on
        to the write queue to allow it be reused.
        """
        while True:
            buffer_index = self.flush_queue.get()
            if buffer_index is None:
                break
            self._commit_write_buffer(buffer_index)
            self.flush_queue.task_done()
            self.write_queue.put(buffer_index)
        self.flush_queue.task_done()

    def _queue_flush_buffer(self):
        """
        Flushes the buffered ancestors to the data file.
        """
        if self.num_threads > 0:
            logger.debug(f"Pushing buffer {self.write_buffer} to flush queue")
            self.flush_queue.put(self.write_buffer)
            self.write_buffer = self.write_queue.get()
        else:
            logger.debug("Syncronously flushing buffer")
            self._commit_write_buffer(self.write_buffer)
        self.num_buffered_items[self.write_buffer] = 0
        self.start_offset[self.write_buffer] = self.total_items

    def add(self, **kwargs):
        """
        Add an item to each of the arrays. The keyword arguments for this
        function correspond to the keys in the dictionary of arrays provided
        to the constructor.
        """
        if self.num_buffered_items[self.write_buffer] == self.chunk_size:
            self._queue_flush_buffer()
        offset = self.num_buffered_items[self.write_buffer]
        for key, value in kwargs.items():
            # Here we have to special case the haplotype for performance
            # reasons, as writing the full haplotype is expensive.
            if key == "haplotype":
                start = kwargs["start"]
                end = kwargs["end"]
                self.buffers["full_haplotype"][self.write_buffer][
                    start:end, offset, 0
                ] = value
                self.buffers["full_haplotype_mask"][self.write_buffer][
                    start:end, offset, 0
                ] = False
            else:
                self.buffers[key][self.write_buffer][offset] = value
        self.num_buffered_items[self.write_buffer] += 1
        self.total_items += 1
        return self.total_items - 1

    def flush(self):
        """
        Flush the remaining items to the destination arrays and return all
        items are safely commited.

        It is an error to call ``add`` after ``flush`` has been called.
        """
        self._queue_flush_buffer()
        # Stop the worker threads.
        for _ in range(self.num_threads):
            self.flush_queue.put(None)
        for j in range(self.num_threads):
            self.flush_threads[j].join()
        self.buffers = None


def zarr_summary(array):
    """
    Returns a string with a brief summary of the specified zarr array.
    """
    dtype = str(array.dtype)
    ret = f"shape={array.shape}; dtype={dtype};"
    if dtype != "object":
        # nbytes doesn't work correctly for object arrays.
        ret += f"uncompressed size={humanize.naturalsize(array.nbytes)}"
    return ret


def chunk_iterator(
    array, indexes=None, select=None, orthogonal_select=None, dimension=0
):
    """
    Utility to iterate over closely spaced rows in the specified array efficiently
    by accessing one chunk at a time (normally used as an iterator over each row)
    """
    # Only the first two dimensions are supported.
    assert dimension < 2
    if select is None:
        select = np.ones(array.shape[dimension], dtype=bool)
    if orthogonal_select is None:
        orthogonal_select = np.ones(array.shape[int(not dimension)], dtype=bool)
    if len(select) != array.shape[dimension]:
        raise ValueError("Mask must be the same length as the array")

    if indexes is None:
        indexes = range(np.sum(select))
    else:
        if len(indexes) > 0 and (
            np.any(np.diff(indexes) <= 0)
            or indexes[0] < 0
            or indexes[-1] >= array.shape[dimension]
        ):
            raise ValueError("Ids must be positive and in ascending order")

    # If there is a variant mask we need to translate the indexes from the masked
    # space to the unmasked space.
    if not np.all(select):
        indexes = np.nonzero(select)[0][indexes]
    chunk_size = array.chunks[dimension]
    prev_chunk_id = -1
    if dimension == 0:
        for j in indexes:
            chunk_id = j // chunk_size
            if chunk_id != prev_chunk_id:
                chunk = array[chunk_id * chunk_size : (chunk_id + 1) * chunk_size][:]
                prev_chunk_id = chunk_id
            yield chunk[j % chunk_size, orthogonal_select]
    elif dimension == 1:
        for j in indexes:
            chunk_id = j // chunk_size
            if chunk_id != prev_chunk_id:
                chunk = array[:, chunk_id * chunk_size : (chunk_id + 1) * chunk_size][:]
                prev_chunk_id = chunk_id
            yield chunk[orthogonal_select, j % chunk_size]


def merge_variants(sd1, sd2):
    """
    Returns an iterator over the merged variants in the specified
    SampleData files. Sites are merged by site position, and
    genotypes are set to missing data for sites are as not present
    in one of the data files.
    """
    var1_iter = iter(sd1.variants())
    var2_iter = iter(sd2.variants())
    var1 = next(var1_iter, None)
    var2 = next(var2_iter, None)
    n1 = sd1.num_samples
    n2 = sd2.num_samples
    genotypes = np.empty(n1 + n2, dtype=np.int8)
    while var1 is not None and var2 is not None:
        if var1.site.position == var2.site.position:
            # Checking metadata as well is probably overly strict, but
            # we can fix this later if needs be.
            if (
                var1.site.ancestral_state != var2.site.ancestral_state
                or not np.array_equal(var1.site.time, var2.site.time, equal_nan=True)
                or var1.site.metadata != var2.site.metadata
            ):
                raise ValueError(
                    "Merged sites must have the same ancestral_state, "
                    "time and metadata"
                )
            # If there is missing data the last allele is always None
            missing_data = False
            alleles = list(var1.site.alleles)
            if alleles[-1] is None:
                alleles = alleles[:-1]
                missing_data = True
            var2_genotypes = var2.genotypes.copy()
            for old_index, allele in enumerate(var2.site.alleles):
                if allele is None:
                    missing_data = True
                    break
                if allele not in alleles:
                    alleles.append(allele)
                new_index = alleles.index(allele)
                if old_index != new_index:
                    var2_genotypes[var2.genotypes == old_index] = new_index
            if missing_data:
                alleles.append(None)
            genotypes[:n1] = var1.genotypes
            genotypes[n1:] = var2_genotypes
            site = var1.site
            site.alleles = alleles
            # TODO not sure why we have alleles on both the Site and Variant
            var = Variant(site=site, genotypes=genotypes, alleles=alleles)
            yield var
            var1 = next(var1_iter, None)
            var2 = next(var2_iter, None)
        elif var1.site.position < var2.site.position:
            genotypes[:n1] = var1.genotypes
            genotypes[n1:] = MISSING_DATA
            var1.genotypes = genotypes
            yield var1
            var1 = next(var1_iter, None)
        else:
            genotypes[:n1] = MISSING_DATA
            genotypes[n1:] = var2.genotypes
            var2.genotypes = genotypes
            yield var2
            var2 = next(var2_iter, None)

    genotypes[n1:] = MISSING_DATA
    while var1 is not None:
        genotypes[:n1] = var1.genotypes
        var1.genotypes = genotypes
        yield var1
        var1 = next(var1_iter, None)

    genotypes[:n1] = MISSING_DATA
    while var2 is not None:
        genotypes[n1:] = var2.genotypes
        var2.genotypes = genotypes
        yield var2
        var2 = next(var2_iter, None)


class DataContainer:
    """
    Superclass of objects used to represent a collection of related
    data. Each datacontainer in a wrapper around a zarr group.
    """

    READ_MODE = 0
    BUILD_MODE = 1
    EDIT_MODE = 2

    # Must be defined by subclasses.
    FORMAT_NAME = None
    FORMAT_VERSION = None

    def __init__(
        self,
        *,
        path=None,
        num_flush_threads=0,
        compressor=DEFAULT_COMPRESSOR,
        chunk_size=1024,
        max_file_size=None,
    ):
        self._mode = self.BUILD_MODE
        self._num_flush_threads = num_flush_threads
        self._chunk_size = max(1, chunk_size)
        self._metadata_codec = numcodecs.JSON()
        self._compressor = compressor
        self.data = zarr.group()
        self.path = path
        if path is not None:
            store = self._new_lmdb_store(max_file_size)
            self.data = zarr.open_group(store=store, mode="w")
        self.data.attrs[FORMAT_NAME_KEY] = self.FORMAT_NAME
        self.data.attrs[FORMAT_VERSION_KEY] = self.FORMAT_VERSION

        chunks = self._chunk_size
        provenances_group = self.data.create_group("provenances")
        provenances_group.create_dataset(
            "timestamp",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=object,
            object_codec=self._metadata_codec,
        )
        provenances_group.create_dataset(
            "record",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=object,
            object_codec=self._metadata_codec,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._mode != self.READ_MODE:
            self.finalise()
        elif self.path is not None:
            self.close()

    def _open_readonly(self):
        if self.path is not None:
            store = open_lmbd_readonly(self.path)
        else:
            # This happens when we finalise an in-memory container.
            store = self.data.store
        self.data = zarr.open(store=store, mode="r")
        self._check_format()
        self._mode = self.READ_MODE

    def _new_lmdb_store(self, map_size=None):
        if os.path.exists(self.path):
            os.unlink(self.path)
        # The existence of a lock-file can confuse things, so delete it.
        remove_lmdb_lockfile(self.path)
        if map_size is None:
            map_size = DEFAULT_MAX_FILE_SIZE
        else:
            map_size = int(map_size)
            if map_size <= 0:
                raise ValueError("max_file_size must be > 0")
        return zarr.LMDBStore(self.path, subdir=False, map_size=map_size)

    @classmethod
    def load(cls, path):
        # Try to read the file. This should raise the correct error if we have a
        # directory, missing file, permissions, etc.
        with open(path):
            pass
        self = cls.__new__(cls)
        self.mode = self.READ_MODE
        self.path = path
        self._open_readonly()
        logger.info(f"Loaded {self.summary()}")
        return self

    def close(self):
        """
        Close this DataContainer. Any read or write operations attempted
        after calling this will fail.
        """
        if self._mode != self.READ_MODE:
            self.finalise()
        if self.data.store is not None:
            self.data.store.close()
        self.data = None
        self.mode = -1

    def copy(self, path=None, max_file_size=None):
        """
        Returns a copy of this DataContainer opened in 'edit' mode. If path
        is specified, this must not be equal to the path of the current
        data container.
        """
        if self._mode != self.READ_MODE:
            raise ValueError("Cannot copy unless in read mode.")
        if path is not None and self.path is not None:
            if os.path.abspath(path) == os.path.abspath(self.path):
                raise ValueError("Cannot copy to the same file")
        cls = type(self)
        other = cls.__new__(cls)
        other.path = path
        if path is None:
            # Have to work around a fairly weird bug in zarr where if we
            # try to use copy_store on an in-memory array we end up
            # overwriting the original values.
            other.data = zarr.group()
            with warnings.catch_warnings():
                # Another workaround: if we don't absorb warnings here
                # we get "FutureWarning: missing object_codec for object array;
                # this will raise a ValueError in v3." Since this is an internal
                # Zarr call it seems easiest to just ignore for now and deal with
                # the ValueError if/when it happens
                warnings.simplefilter("ignore")
                zarr.copy_all(source=self.data, dest=other.data)
            for key, value in self.data.attrs.items():
                other.data.attrs[key] = value
        else:
            store = other._new_lmdb_store(max_file_size)
            zarr.copy_store(self.data.store, store)
            other.data = zarr.group(store)
        other.data.attrs[FINALISED_KEY] = False
        other._mode = self.EDIT_MODE
        return other

    def finalise(self):
        """
        Ensures that the state of the data is flushed and writes the
        provenance for the current operation. The specified 'command' is used
        to fill the corresponding entry in the provenance dictionary.
        """
        self._check_write_modes()
        zarr.consolidate_metadata(self.data.store)
        self.data.attrs[FINALISED_KEY] = True
        if self.path is not None:
            store = self.data.store
            store.close()
            logger.debug("Fixing up LMDB file size")
            with lmdb.open(self.path, subdir=False, lock=False, writemap=True) as db:
                # LMDB maps a very large amount of space by default. While this
                # doesn't do any harm, it's annoying because we can't use ls to
                # see the file sizes and the amount of RAM we're mapping can
                # look like it's very large. So, we fix this up so that the
                # map size is equal to the number of pages in use.
                num_pages = db.info()["last_pgno"]
                page_size = db.stat()["psize"]
                db.set_mapsize(num_pages * page_size)
            # Remove the lock file as we don't need it after this point.
            remove_lmdb_lockfile(self.path)
        self._open_readonly()

    def _check_format(self):
        try:
            format_name = self.format_name
            format_version = self.format_version
        except KeyError:
            raise exceptions.FileFormatError("Incorrect file format")
        if format_name != self.FORMAT_NAME:
            raise exceptions.FileFormatError(
                "Incorrect file format: expected '{}' got '{}'".format(
                    self.FORMAT_NAME, format_name
                )
            )
        if format_version[0] < self.FORMAT_VERSION[0]:
            raise exceptions.FileFormatTooOld(
                "Format version {} too old. Current version = {}".format(
                    format_version, self.FORMAT_VERSION
                )
            )
        if format_version[0] > self.FORMAT_VERSION[0]:
            raise exceptions.FileFormatTooNew(
                "Format version {} too new. Current version = {}".format(
                    format_version, self.FORMAT_VERSION
                )
            )

    def _check_build_mode(self):
        if self._mode != self.BUILD_MODE:
            raise ValueError("Invalid operation: must be in build mode")

    def _check_edit_mode(self):
        if self._mode != self.EDIT_MODE:
            raise ValueError("Invalid operation: must be in edit mode")

    def _check_write_modes(self):
        if self._mode not in (self.EDIT_MODE, self.BUILD_MODE):
            raise ValueError("Invalid operation: must be in edit or build mode")

    def _check_finalised(self):
        if not self.finalised:
            error_msg = f"The {self.format_name} file"
            if self.path is not None:
                error_msg = f" at `{self.path}`"
            raise ValueError(error_msg + " is not finalised")

    @property
    def file_size(self):
        """
        Returns the size of the underlying file, or -1 if we do not have a
        file associated.
        """
        ret = -1
        if self.path is not None:
            ret = os.path.getsize(self.path)
        return ret

    def _check_metadata(self, metadata):
        ret = metadata
        if metadata is None:
            ret = {}
        elif not isinstance(metadata, abc.Mapping):
            raise TypeError("Metadata must be a JSON-like dictionary")
        return ret

    def add_provenance(self, timestamp, record):
        """
        Adds a new provenance record with the specified timestamp and record.
        Timestamps should ISO8601 formatted, and record is some JSON encodable
        object.
        """
        if self._mode not in (self.BUILD_MODE, self.EDIT_MODE):
            raise ValueError(
                "Invalid operation: cannot add provenances unless in BUILD "
                "or EDIT mode"
            )
        n = self.num_provenances
        self.provenances_timestamp.resize(n + 1)
        self.provenances_record.resize(n + 1)
        self.provenances_timestamp[n] = timestamp
        self.provenances_record[n] = record

    def record_provenance(self, command=None, **kwargs):
        """
        Records the provenance information for this file using the
        tskit provenances schema.
        """
        timestamp = datetime.datetime.now().isoformat()
        record = provenance.get_provenance_dict(command=command, **kwargs)
        self.add_provenance(timestamp, record)

    def clear_provenances(self):
        """
        Clear all provenances in this instance
        """
        if self._mode not in (self.BUILD_MODE, self.EDIT_MODE):
            raise ValueError(
                "Invalid operation: cannot clear provenances unless in BUILD "
                "or EDIT mode"
            )
        self.provenances_timestamp.resize(0)
        self.provenances_record.resize(0)

    @property
    def format_name(self):
        return self.data.attrs[FORMAT_NAME_KEY]

    @property
    def format_version(self):
        return tuple(self.data.attrs[FORMAT_VERSION_KEY])

    @property
    def finalised(self):
        ret = False
        if FINALISED_KEY in self.data.attrs:
            ret = self.data.attrs[FINALISED_KEY]
        return ret

    @property
    def num_provenances(self):
        return self.provenances_timestamp.shape[0]

    @property
    def provenances_timestamp(self):
        return self.data["provenances/timestamp"]

    @property
    def provenances_record(self):
        return self.data["provenances/record"]

    def _format_str(self, values):
        """
        Helper function for formatting __str__ output.
        """
        s = ""
        # Quick hack to make sure everything lines up.
        max_key = len("provenances/timestamp")
        for k, v in values:
            s += "{:<{}} = {}\n".format(k, max_key, v)
        return s

    def __eq__(self, other):
        ret = NotImplemented
        if isinstance(other, type(self)):
            ret = self.data_equal(other)
        return ret

    def __str__(self):
        values = [
            ("path", self.path),
            ("file_size", humanize.naturalsize(self.file_size, binary=True)),
            ("format_name", self.format_name),
            ("format_version", self.format_version),
            ("finalised", self.finalised),
            ("num_provenances", self.num_provenances),
            ("provenances/timestamp", zarr_summary(self.provenances_timestamp)),
            ("provenances/record", zarr_summary(self.provenances_record)),
        ]
        return self._format_str(values)

    def arrays(self):
        """
        Returns a list of all the zarr arrays in this DataContainer.
        """
        ret = []

        def visitor(name, obj):
            if isinstance(obj, zarr.Array):
                ret.append((name, obj))

        self.data.visititems(visitor)
        return ret

    @property
    def info(self):
        """
        Returns a string containing the zarr info for each array.
        """
        s = str(self.data.info)
        for _, array in self.arrays():
            s += ("-" * 80) + "\n"
            s += str(array.info)
        return s

    def provenances(self):
        """
        Returns an iterator over the (timestamp, record) pairs representing
        the provenances for this data container.
        """
        timestamp = self.provenances_timestamp[:]
        record = self.provenances_record[:]
        for j in range(self.num_provenances):
            yield timestamp[j], record[j]


@attr.s
class Site:
    """
    A single site. Mirrors the definition in tskit with some additional fields.
    """

    # TODO document properly.
    id = attr.ib()
    position = attr.ib()
    ancestral_allele = attr.ib()  # here -1 (tskit.MISSING_DATA) means none defined
    metadata = attr.ib()
    time = attr.ib()
    alleles = attr.ib()

    @property
    def ancestral_state(self):
        if self.ancestral_allele == MISSING_DATA:
            return None
        return self.alleles[self.ancestral_allele]

    def reorder_alleles(self):
        """
        The alleles list reordered so that the ancestral allele is first
        """
        if self.ancestral_allele > 0:
            return (
                (self.alleles[self.ancestral_allele],)
                + self.alleles[: self.ancestral_allele]
                + self.alleles[self.ancestral_allele + 1 :]
            )
        return self.alleles


@attr.s
class Variant:
    """
    A single variant. Mirrors the definition in tskit.
    """

    # TODO document properly.
    site = attr.ib()
    genotypes = attr.ib()
    alleles = attr.ib()


@attr.s
class Individual:
    """
    An Individual object, representing a single individual which may contain multiple
    *samples* (i.e. phased genomes). For instance, a diploid individual will have
    two sample genomes. This is deliberately similar to a :class:`tskit.Individual`.

    Individuals are created with :meth:`SampleData.add_individual`. If a tree sequence
    is inferred from a sample data file containing individuals, these individuals (and
    the data associated with them) will carry through to the inferred tree sequence.
    """

    # TODO document properly.
    id = attr.ib()
    flags = attr.ib()
    location = attr.ib()
    metadata = attr.ib()
    # the samples attribute is filled in programmatically, not stored per individual
    samples = attr.ib()
    # Not in equivalent tskit object
    population = attr.ib()  # NB: differs from tskit, which stores this per node
    time = attr.ib()  # NB: differs from tskit, which stores this per node


@attr.s
class Sample:
    """
    A Sample object, representing a single haploid genome or chromosome. Several
    Samples can be associated with the same :class:`Individual`: for example a
    diploid individual will have one maternal and one paternal sample.

    If a tree sequence is inferred from a set of samples, each sample will be
    associated with a tskit "node", which will be flagged up with
    :data:`tskit.NODE_IS_SAMPLE`.
    """

    # TODO document properly.
    id = attr.ib()
    individual = attr.ib()


@attr.s
class Population:
    """
    A Population object. Mirrors :class:`tskit.Population`.
    """

    # TODO document properly.
    id = attr.ib()
    metadata = attr.ib()


class SampleData(DataContainer):
    """
    SampleData(sequence_length=0, *, path=None, num_flush_threads=0, \
    compressor=DEFAULT_COMPRESSOR, chunk_size=1024, max_file_size=None)

    Class representing input sample data used for inference.
    See sample data file format :ref:`specifications <sec_file_formats_samples>`
    for details on the structure of this file.

    The most common usage for this class will be to import data from some
    external source and save it to file for later use. This will usually
    follow a pattern like:

    .. code-block:: python

        sample_data = tsinfer.SampleData(path="mydata.samples")
        sample_data.add_site(position=1234, genotypes=[0, 0, 1, 0], alleles=["G", "C"])
        sample_data.add_site(position=5678, genotypes=[1, 1, 1, 0], alleles=["A", "T"])
        sample_data.finalise()

    This creates a sample data file for four haploid samples and two sites, and
    saves it in the file "mydata.samples". Note that the call to
    :meth:`.finalise` is essential here to ensure that all data will be
    correctly flushed to disk. For convenience, a context manager may
    also be used to ensure this is done:

    .. code-block:: python

        with tsinfer.SampleData(path="mydata.samples") as sample_data:
            sample_data.add_site(1234, [0, 0, 1, 0], ["G", "C"])
            sample_data.add_site(5678, [1, 1, 1, 0], ["A", "T"])

    More complex :ref:`data models <sec_inference_data_model>` consisting
    of populations and polyploid individuals can also be specified. For
    example, we might have:

    .. code-block:: python

        with tsinfer.SampleData(path="mydata.samples") as sample_data:
            # Define populations
            sample_data.add_population(metadata={"name": "CEU"})
            sample_data.add_population(metadata={"name": "YRI"})
            # Define individuals
            sample_data.add_individual(ploidy=2, population=0, metadata={"name": "NA12"})
            sample_data.add_individual(ploidy=2, population=0, metadata={"name": "NA13"})
            sample_data.add_individual(ploidy=2, population=0, metadata={"name": "NA14"})
            sample_data.add_individual(ploidy=2, population=1, metadata={"name": "NA15"})
            # Define sites and genotypes
            sample_data.add_site(1234, [0, 1, 1, 1, 0, 0, 0, 0], ["G", "C"])
            sample_data.add_site(5678, [0, 0, 0, 0, 0, 0, 1, 1], ["A", "T"])

    In this example we defined two populations and four diploid individuals,
    and so our genotypes arrays are of length eight. Thus, at first site the
    first individual is heterozygous, the second is homozygous with the derived
    allele and the other two individuals are homozygous with the ancestral
    allele. To illustrate how we can use site and population metadata to link
    up with external data sources we use the 1000 genomes identifiers (although
    of course the genotype data is fake). Here we suppose that we have the
    famous NA12878 trio from the CEU population, and one other individual from
    the YRI population. This metadata is then embedded in the final tree
    sequence that we infer, allowing us to use it conveniently in downstream
    analyses.

    .. note:: If a ``path`` is specified, the ``max_file_size`` option puts an
        upper limit on the possible size of the created file. On non-Windows
        systems, space for this file is not immediately allocated but just
        "reserved" using sparse file systems. However, on Windows systems
        the file is allocated immediately, so ``max_file_size`` takes a smaller
        default value, to avoid allocating very large files for no reason.
        Users who wish to run large inferences on Windows may therefore need to
        explictly set an appropriate ``max_file_size``. Note that the
        ``max_file_size`` is only used while the file is being built: one the
        file has been finalised, it is shrunk to its minimum size.

    :param float sequence_length: If specified, this is the sequence length
        that will be associated with the tree sequence output by
        :func:`tsinfer.infer` and :func:`tsinfer.match_samples`. If provided
        site coordinates must be less than this value.
    :param str path: The path of the file to store the sample data. If None,
        the information is stored in memory and not persistent.
    :param int num_flush_threads: The number of background threads to use
        for compressing data and flushing to disc. If <= 0, do not spawn
        any threads but use a synchronous algorithm instead. Default=0.
    :param numcodecs.abc.Codec compressor: A :class:`numcodecs.abc.Codec`
        instance to use for compressing data. Any codec may be used, but
        problems may occur with very large datasets on certain codecs as
        they cannot compress buffers >2GB. If None, do not use any compression.
        Default=:class:`numcodecs.zstd.Zstd`.
    :param int chunk_size: The chunk size used for
        `zarr arrays <http://zarr.readthedocs.io/>`_. This affects
        compression level and algorithm performance. Default=1024.
    :param int max_file_size: If a file is being used to store this data, set
        a maximum size in bytes for the stored file. If None, the default
        value of 1GiB (2**30 bytes) is used on Windows and 1TiB (2**40 bytes)
        on other platforms (see above for details).
    """

    FORMAT_NAME = "tsinfer-sample-data"
    FORMAT_VERSION = (5, 1)

    # State machine for handling automatic addition of samples.
    ADDING_POPULATIONS = 0
    ADDING_SAMPLES = 1
    ADDING_SITES = 2

    def __init__(self, sequence_length=0, **kwargs):

        super().__init__(**kwargs)
        self.data.attrs["sequence_length"] = float(sequence_length)
        self.data.attrs["metadata"] = {}
        self.data.attrs[
            "metadata_schema"
        ] = tskit.MetadataSchema.permissive_json().schema
        chunks = (self._chunk_size,)
        populations_group = self.data.create_group("populations")
        metadata = populations_group.create_dataset(
            "metadata",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=object,
            object_codec=self._metadata_codec,
        )
        populations_group.attrs["metadata_schema"] = None
        self._populations_writer = BufferedItemWriter(
            {"metadata": metadata}, num_threads=self._num_flush_threads
        )

        individuals_group = self.data.create_group("individuals")
        individuals_group.attrs["metadata_schema"] = None
        metadata = individuals_group.create_dataset(
            "metadata",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=object,
            object_codec=self._metadata_codec,
        )
        location = individuals_group.create_dataset(
            "location",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype="array:f8",
        )
        time = individuals_group.create_dataset(
            "time",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.float64,
        )
        population = individuals_group.create_dataset(
            "population",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.int32,
        )
        flags = individuals_group.create_dataset(
            "flags",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.uint32,
        )
        self._individuals_writer = BufferedItemWriter(
            {
                "metadata": metadata,
                "location": location,
                "time": time,
                "population": population,
                "flags": flags,
            },
            num_threads=self._num_flush_threads,
        )

        samples_group = self.data.create_group("samples")
        individual = samples_group.create_dataset(
            "individual",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.int32,
        )
        self._samples_writer = BufferedItemWriter(
            {"individual": individual},
            num_threads=self._num_flush_threads,
        )

        sites_group = self.data.create_group("sites")
        sites_group.attrs["metadata_schema"] = None
        sites_group.create_dataset(
            "position",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.float64,
        )
        sites_group.create_dataset(
            "time",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.float64,
        )
        sites_group.create_dataset(
            "genotypes",
            shape=(0, 0),
            chunks=(self._chunk_size, self._chunk_size),
            compressor=self._compressor,
            dtype=np.int8,
        )
        sites_group.create_dataset(
            "alleles",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=object,
            object_codec=self._metadata_codec,
        )
        sites_group.create_dataset(
            "ancestral_allele",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=np.int8,
        )
        sites_group.create_dataset(
            "metadata",
            shape=(0,),
            chunks=chunks,
            compressor=self._compressor,
            dtype=object,
            object_codec=self._metadata_codec,
        )

        self._last_position = 0
        self._sites_writer = None
        # We are initially in the ADDING_POPULATIONS state.
        self._build_state = self.ADDING_POPULATIONS

    def summary(self):
        return "SampleData(num_samples={}, num_sites={})".format(
            self.num_samples, self.num_sites
        )

    # Note: abstracting the process of getting and setting the metadata schemas
    # out here so we can do better validation of the inputs/optionally accept
    # a tskit MetadataSchema object rather than a dict.
    def __metadata_schema_getter(self, zarr_group):
        return zarr_group.attrs["metadata_schema"]

    def __metadata_schema_setter(self, zarr_group, schema):
        # Make sure we can parse it.
        if schema is not None:
            parsed_schema = tskit.MetadataSchema(schema)
            # We only support the JSON codec for now for simplicity.
            if parsed_schema.schema["codec"] != "json":
                raise ValueError("Only the JSON codec is currently supported")
        zarr_group.attrs["metadata_schema"] = schema

    @property
    def sequence_length(self):
        return self.data.attrs["sequence_length"]

    @property
    def metadata_schema(self):
        return self.data.attrs["metadata_schema"]

    @metadata_schema.setter
    def metadata_schema(self, schema):
        if schema is None:
            raise ValueError("Must have a schema for top-level metadata")
        self.__metadata_schema_setter(self.data, schema)

    @property
    def metadata(self):
        return self.data.attrs["metadata"]

    @metadata.setter
    def metadata(self, metadata):
        self.data.attrs["metadata"] = metadata

    @property
    def populations_metadata_schema(self):
        return self.__metadata_schema_getter(self.data["populations"])

    @populations_metadata_schema.setter
    def populations_metadata_schema(self, schema):
        self.__metadata_schema_setter(self.data["populations"], schema)

    @property
    def individuals_metadata_schema(self):
        return self.__metadata_schema_getter(self.data["individuals"])

    @individuals_metadata_schema.setter
    def individuals_metadata_schema(self, schema):
        self.__metadata_schema_setter(self.data["individuals"], schema)

    @property
    def sites_metadata_schema(self):
        return self.__metadata_schema_getter(self.data["sites"])

    @sites_metadata_schema.setter
    def sites_metadata_schema(self, schema):
        self.__metadata_schema_setter(self.data["sites"], schema)

    @property
    def num_populations(self):
        return self.populations_metadata.shape[0]

    @property
    def num_samples(self):
        return self.samples_individual.shape[0]

    @property
    def num_individuals(self):
        return self.individuals_metadata.shape[0]

    @property
    def num_sites(self):
        return self.sites_position.shape[0]

    @property
    def populations_metadata(self):
        return self.data["populations/metadata"]

    @property
    def individuals_metadata(self):
        return self.data["individuals/metadata"]

    @property
    def individuals_location(self):
        return self.data["individuals/location"]

    @property
    def individuals_time(self):
        return self.data["individuals/time"]

    @property
    def individuals_population(self):
        return self.data["individuals/population"]

    @property
    def individuals_flags(self):
        return self.data["individuals/flags"]

    @property
    def samples_individual(self):
        return self.data["samples/individual"]

    @property
    def sites_genotypes(self):
        """
        The "raw" genotypes array for each site, as passed in when adding sites. The
        values in this array correspond to indexes into the :attr:`sites_alleles` array.
        """
        return self.data["sites/genotypes"]

    @property
    def sites_position(self):
        return self.data["sites/position"]

    @property
    def sites_time(self):
        return self.data["sites/time"]

    @sites_time.setter
    def sites_time(self, value):
        self._check_edit_mode()
        self.data["sites/time"][:] = np.asarray(value, dtype=np.float64)

    @property
    def sites_alleles(self):
        """
        The alleles list for each site, in the order given when adding sites. If
        missing data is present, the last allelic state will be ``None``.
        """
        return self.data["sites/alleles"]

    @property
    def sites_ancestral_allele(self):
        """
        The index into each :attr:`sites_alleles` list which corresponds to the
        ancestral state. If the ancestral state is unknown, this is indicated by
        a value of tskit.MISSING_DATA (-1).
        """
        try:
            return self.data["sites/ancestral_allele"]
        except KeyError:
            # Maintains backwards compatibility: in previous tsinfer versions the
            # ancestral allele was always the zeroth element in the alleles list
            return np.zeros(self.num_sites, dtype=np.int8)

    @property
    def sites_metadata(self):
        return self.data["sites/metadata"]

    def __str__(self):
        values = [
            ("sequence_length", self.sequence_length),
            ("metadata_schema", self.metadata_schema),
            ("metadata", self.metadata),
            ("num_populations", self.num_populations),
            ("num_individuals", self.num_individuals),
            ("num_samples", self.num_samples),
            ("num_sites", self.num_sites),
            ("populations/metadata_schema", self.populations_metadata_schema),
            ("populations/metadata", zarr_summary(self.populations_metadata)),
            ("individuals/metadata_schema", self.individuals_metadata_schema),
            ("individuals/metadata", zarr_summary(self.individuals_metadata)),
            ("individuals/location", zarr_summary(self.individuals_location)),
            ("individuals/time", zarr_summary(self.individuals_time)),
            ("individuals/population", zarr_summary(self.individuals_population)),
            ("individuals/flags", zarr_summary(self.individuals_flags)),
            ("samples/individual", zarr_summary(self.samples_individual)),
            ("sites/position", zarr_summary(self.sites_position)),
            ("sites/time", zarr_summary(self.sites_time)),
            ("sites/alleles", zarr_summary(self.sites_alleles)),
            ("sites/genotypes", zarr_summary(self.sites_genotypes)),
            ("sites/metadata_schema", self.sites_metadata_schema),
            ("sites/metadata", zarr_summary(self.sites_metadata)),
        ]
        return super().__str__() + self._format_str(values)

    def formats_equal(self, other):
        return (
            self.format_name == other.format_name
            and self.format_version == other.format_version
        )

    def populations_equal(self, other):
        return (
            self.num_populations == other.num_populations
            # Need to take a different approach with np object arrays.
            and np_obj_equal(
                self.populations_metadata[:], other.populations_metadata[:]
            )
        )

    def individuals_equal(self, other):
        return (
            self.num_individuals == other.num_individuals
            and np.allclose(
                self.individuals_time[:], other.individuals_time[:], equal_nan=True
            )
            and np.array_equal(self.individuals_flags[:], other.individuals_flags[:])
            and np.array_equal(
                self.individuals_population[:], other.individuals_population[:]
            )
            and np_obj_equal(
                self.individuals_metadata[:], other.individuals_metadata[:]
            )
            and np_obj_equal(
                self.individuals_location[:], other.individuals_location[:]
            )
        )

    def samples_equal(self, other):
        return self.num_samples == other.num_samples and np.all(
            self.samples_individual[:] == other.samples_individual[:]
        )

    def sites_equal(self, other):
        return (
            self.num_sites == other.num_sites
            and np.all(self.sites_position[:] == other.sites_position[:])
            and np.all(self.sites_genotypes[:] == other.sites_genotypes[:])
            and np.allclose(self.sites_time[:], other.sites_time[:], equal_nan=True)
            and np_obj_equal(self.sites_metadata[:], other.sites_metadata[:])
            and np_obj_equal(self.sites_alleles[:], other.sites_alleles[:])
        )

    def data_equal(self, other):
        """
        Returns True if all the data attributes of this input file and the
        specified input file are equal. This compares every attribute except
        the provenance.

        To compare two :class:`SampleData` instances for exact equality of
        all data including provenance data, use ``s1 == s2``.

        :param SampleData other: The other :class:`SampleData` instance to
            compare with.
        :return: ``True`` if the data held in this :class:`SampleData`
            instance is identical to the date held in the other instance.
        :rtype: bool
        """
        return (
            self.sequence_length == other.sequence_length
            and self.formats_equal(other)
            and self.populations_equal(other)
            and self.individuals_equal(other)
            and self.samples_equal(other)
            and self.sites_equal(other)
        )

    def assert_data_equal(self, other):
        """
        The same as :meth:`.data_equal`, but raises an assertion rather than returning
        False. This is useful for testing.
        """
        assert self.sequence_length == other.sequence_length
        assert self.formats_equal(other)
        assert self.populations_equal(other)
        assert self.individuals_equal(other)
        assert self.samples_equal(other)
        assert self.sites_equal(other)

    def subset(self, individuals=None, sites=None, *, sequence_length=None, **kwargs):
        """
        Returns a subset of this sample data file consisting of the specified
        individuals and sites. It is important to note that these are
        *individual* IDs and not *sample* IDs (corresponding to distinct
        haplotypes within an individual). When working with haploid data, the
        individual and sample IDs are guaranteed to be the same, and so can be
        used interchangably.

        :param arraylike individuals: The individual IDs to retain in the
            returned subset. IDs must be unique, and refer to valid individuals
            in the current dataset. IDs can be supplied in any order,
            and the order will be preserved in the returned data file (i.e.,
            ``individuals[0]`` will be the first individual in the new
            dataset, etc).
        :param arraylike sites: The site IDs to retain in the
            returned subset. IDs must be unique, and refer to valid sites
            in the current dataset. Site IDs can be supplied in any order,
            but the order will *not* be preserved in the returned data file,
            so that sites are always in position sorted order in the output.
        :param float sequence_length: The sequence length to use for the
            returned object. If None, use the same sequence length as in the
            original sample data file.
        :param \\**kwargs: Further arguments passed to the :class:`SampleData`
            constructor.
        :return: A :class:`.SampleData` object.
        :rtype: SampleData
        """
        if individuals is None:
            individuals = np.arange(self.num_individuals)
        individuals = np.array(individuals, dtype=np.int32)
        if np.any(individuals < 0) or np.any(individuals >= self.num_individuals):
            raise ValueError("Individual ID out of bounds")
        if len(set(individuals)) != len(individuals):
            raise ValueError("Duplicate individual IDs")
        if sites is None:
            sites = np.arange(self.num_sites)
        sites = np.array(sites, dtype=np.int32)
        if np.any(sites < 0) or np.any(sites >= self.num_sites):
            raise ValueError("Site ID out of bounds")
        num_sites = len(sites)
        # Store the sites as a set for quick lookup.
        sites = set(sites)
        if len(sites) != num_sites:
            raise ValueError("Duplicate site IDS")
        if sequence_length is None:
            sequence_length = self.sequence_length
        with SampleData(sequence_length=sequence_length, **kwargs) as subset:
            # NOTE We don't bother filtering the populations, but we could.
            for population in self.populations():
                subset.add_population(population.metadata)
            sample_selection = []
            for individual_id in individuals:
                individual = self.individual(individual_id)
                sample_selection.extend(individual.samples)
                subset.add_individual(
                    location=individual.location,
                    metadata=individual.metadata,
                    time=individual.time,
                    population=individual.population,
                    ploidy=len(individual.samples),
                )
            sample_selection = np.array(sample_selection, dtype=int)
            if len(sample_selection) < 1:
                raise ValueError("Must have at least one sample")
            for variant in self.variants():
                if variant.site.id in sites:
                    subset.add_site(
                        position=variant.site.position,
                        genotypes=variant.genotypes[sample_selection],
                        alleles=variant.alleles,
                        metadata=variant.site.metadata,
                        time=variant.site.time,
                    )
            for timestamp, record in self.provenances():
                subset.add_provenance(timestamp, record)
            subset.record_provenance(command="subset", **kwargs)
        return subset

    def min_site_times(self, individuals_only=False):
        """
        Returns a numpy array of the lower bound of the time of sites in the SampleData
        file. Each individual with a nonzero time (from the individuals_time array)
        gives a lower bound on the age of sites where the individual carries a
        derived allele.

        :return: A numpy array of the lower bound for each sites time.
        :rtype: numpy.ndarray(dtype=float64)
        """
        samples_individual = self.samples_individual[:]
        assert np.all(samples_individual >= 0)
        samples_time = self.individuals_time[:][samples_individual]
        if np.any(samples_time < 0):
            raise ValueError("Individuals cannot have negative times")
        historical_samples = samples_time != 0
        historical_samples_time = samples_time[historical_samples]
        sites_bound = np.zeros(self.num_sites)
        for var in self.variants():
            historical_genos = var.genotypes[historical_samples]
            derived = historical_genos > 0
            if np.any(derived):
                historical_bound = np.max(historical_samples_time[derived])
                if historical_bound > sites_bound[var.site.id]:
                    sites_bound[var.site.id] = historical_bound
        if not individuals_only:
            sites_bound = np.maximum(self.sites_time[:], sites_bound)
        return sites_bound

    ####################################
    # Write mode
    ####################################

    @classmethod
    def from_tree_sequence(
        cls,
        ts,
        use_sites_time=None,
        use_individuals_time=None,
        **kwargs,
    ):
        """
        Create a SampleData instance from the sample nodes in an existing tree sequence.
        Each sample node in the tree sequence results in a sample created in the returned
        object. Populations in the tree sequence will be copied into the returned object.
        Individuals in the tree sequence that are associated with any sample nodes will
        also be incorporated: the ploidy of each individual is assumed to be the number
        of sample nodes which reference that individual; individuals with no sample nodes
        are omitted. A new haploid individual is created for any sample node which lacks
        an associated individual in the existing tree sequence. Thus a tree sequence
        with ``u`` sample nodes but no individuals will be translated into a SampleData
        file with ``u`` haploid individuals and ``u`` samples.

        Metadata associated with individuals, populations, sites, and at the top level
        of the tree sequence, is also stored in the appropriate places in the returned
        SampleData instance. Any such metadata must either have a schema defined
        or be JSON encodable text. See the `tskit documentation
        <https://tskit.readthedocs.io/en/stable/metadata.html>`_ for more details
        on metadata schemas.

        :param tskit.TreeSequence ts: The :class:`tskit.TreeSequence` from which to
            generate samples.
        :param bool use_sites_time: If ``True``, the times of nodes in the tree
            sequence are used to set a time for each site (which affects the relative
            temporal order of ancestors during inference). Times for a site are only
            used if there is a single mutation at that site, in which case the node
            immediately below the mutation is taken as the origination time for the
            variant. If ``False``, the frequency of the variant is used as a proxy for
            the relative variant time (see :meth:`.add_site`). Defaults to ``False``.
        :param bool use_individuals_time: If ``True``, use the time of the sample nodes
            in the tree sequence as the time of the individuals associated with
            those nodes in the sample data file. This is likely only to be meaningful if
            ``use_sites_time`` is also ``True``. If ``False``, all individuals are set
            to time 0. Defaults to ``False``.
        :param \\**kwargs: Further arguments passed to the :class:`SampleData`
            constructor.
        :return: A :class:`.SampleData` object.
        :rtype: SampleData
        """

        def encode_metadata(metadata, schema):
            if schema is None:
                if len(metadata) > 0:
                    metadata = json.loads(metadata)
                else:
                    metadata = None
            return metadata

        if use_sites_time is None:
            use_sites_time = False
        if use_individuals_time is None:
            use_individuals_time = False

        tables = ts.tables
        self = cls(sequence_length=ts.sequence_length, **kwargs)

        schema = tables.metadata_schema.schema
        if schema is not None:
            self.metadata_schema = schema
            self.metadata = tables.metadata
        else:
            assert len(tables.metadata) == 0

        schema = tables.populations.metadata_schema.schema
        self.populations_metadata_schema = schema
        for population in ts.populations():
            self.add_population(metadata=encode_metadata(population.metadata, schema))

        schema = tables.individuals.metadata_schema.schema
        self.individuals_metadata_schema = schema
        for individual in ts.individuals():
            nodes = individual.nodes
            if len(nodes) > 0:
                time = 0
                first_node = ts.node(nodes[0])
                for u in nodes[1:]:
                    if ts.node(u).time != first_node.time:
                        raise ValueError(
                            "All nodes for individual {} must have the same time".format(
                                individual.id
                            )
                        )
                    if ts.node(u).population != first_node.population:
                        raise ValueError(
                            "All nodes for individual {} must be in the same "
                            "population".format(individual.id)
                        )
                metadata = encode_metadata(individual.metadata, schema)
                if use_individuals_time:
                    time = first_node.time
                    if time != 0 and not use_sites_time:
                        raise ValueError(
                            "Incompatible timescales: site frequencies used for times "
                            f"(use_sites_time=False), but node {first_node.id} in "
                            f"individual {individual.id} has a nonzero time and "
                            "use_individuals_time=True. Please set site times manually."
                        )
                self.add_individual(
                    location=individual.location,
                    metadata=metadata,
                    population=first_node.population,
                    flags=individual.flags,
                    time=time,
                    ploidy=len(nodes),
                )
        for u in ts.samples():
            node = ts.node(u)
            if node.individual == tskit.NULL:
                # The sample node has no individual: create a haploid individual for it
                time = 0
                if use_individuals_time:
                    time = node.time
                    if time != 0 and not use_sites_time:
                        raise ValueError(
                            "Incompatible timescales: site frequencies used for times "
                            f"(use_sites_time=False), but node {node.id} "
                            "has a nonzero time and use_individuals_time=True. "
                            "Please set site times manually."
                        )
                self.add_individual(
                    population=node.population,
                    time=node.time if use_individuals_time else 0,
                    ploidy=1,
                )

        schema = tables.sites.metadata_schema.schema
        self.sites_metadata_schema = schema
        for v in ts.variants():
            variant_time = tskit.UNKNOWN_TIME
            if use_sites_time:
                variant_time = np.nan
                if len(v.site.mutations) == 1:
                    variant_time = ts.node(v.site.mutations[0].node).time
            self.add_site(
                v.site.position,
                v.genotypes,
                v.alleles,
                metadata=encode_metadata(v.site.metadata, schema),
                time=variant_time,
            )
        # Insert all the provenance from the original tree sequence.
        for prov in ts.provenances():
            self.add_provenance(prov.timestamp, json.loads(prov.record))
        self.record_provenance(command="from-tree-sequence", **kwargs)
        self.finalise()
        return self

    def _alloc_site_writer(self):
        if self.num_samples < 1:
            raise ValueError("Must have at least 1 sample")
        self.sites_genotypes.resize(0, self.num_samples)
        arrays = {
            "position": self.sites_position,
            "genotypes": self.sites_genotypes,
            "alleles": self.sites_alleles,
            "metadata": self.sites_metadata,
            "time": self.sites_time,
            "ancestral_allele": self.sites_ancestral_allele,
        }
        self._sites_writer = BufferedItemWriter(
            arrays, num_threads=self._num_flush_threads
        )

    def add_population(self, metadata=None):
        """
        Adds a new :ref:`sec_inference_data_model_population` to this
        :class:`.SampleData` and returns its ID.

        All calls to this method must be made **before** individuals or sites
        are defined.

        :param dict metadata: A JSON encodable dict-like object containing
            metadata that is to be associated with this population.
        :return: The ID of the newly added population.
        :rtype: int
        """
        self._check_build_mode()
        if self._build_state != self.ADDING_POPULATIONS:
            raise ValueError("Cannot add populations after adding samples or sites")
        return self._populations_writer.add(metadata=self._check_metadata(metadata))

    def add_individual(
        self,
        ploidy=1,
        metadata=None,
        population=None,
        location=None,
        time=0,
        flags=None,
    ):
        """
        Adds a new :ref:`sec_inference_data_model_individual` to this
        :class:`.SampleData` and returns its ID and those of the resulting additional
        samples. Adding an individual with ploidy ``k`` results in ``k`` new samples
        being added, and each of these samples will be associated with the
        new individual. Each new sample will also be associated with the specified
        population ID. It is an error to specify a population ID that does not
        correspond to a population defined using :meth:`.add_population`.

        All calls to this method must be made **after** populations are defined
        using :meth:`.add_population` and **before** sites are defined using
        :meth:`.add_site`.

        :param int ploidy: The ploidy of this individual. This corresponds to the
            number of samples added that refer to this individual. Defaults to 1
            (haploid).
        :param dict metadata: A JSON encodable dict-like object containing
            metadata that is to be associated with this individual.
        :param int population: The ID of the population to associate with this
            individual (or more precisely, with the samples for this individual).
            If not specified or None, defaults to the null population (-1).
        :param arraylike location: An array-like object defining n-dimensional
            spatial location of this individual. If not specified or None, the
            empty location is stored.
        :param float time: The historical time into the past when the samples
            associated with this individual were taken. By default we assume that
            all samples come from the present time (i.e. the default time is 0).
        :param int flags: The bitwise flags for this individual.
        :return: The ID of the newly added individual and a list of the sample
            IDs also added.
        :rtype: tuple(int, list(int))
        """
        self._check_build_mode()
        if self._build_state == self.ADDING_POPULATIONS:
            self._populations_writer.flush()
            self._populations_writer = None
            self._build_state = self.ADDING_SAMPLES
        if self._build_state != self.ADDING_SAMPLES:
            raise ValueError("Cannot add individuals after adding sites")

        time = np.float64(time).item()
        if not np.isfinite(time):
            raise ValueError("time must be a single finite number")
        if population is None:
            population = tskit.NULL
        if population >= self.num_populations:
            raise ValueError("population ID out of bounds")
        if ploidy <= 0:
            raise ValueError("Ploidy must be at least 1")
        if location is None:
            location = []
        location = np.array(location, dtype=np.float64)
        if flags is None:
            flags = 0
        individual_id = self._individuals_writer.add(
            metadata=self._check_metadata(metadata),
            location=location,
            time=time,
            population=population,
            flags=flags,
        )
        sample_ids = []
        for _ in range(ploidy):
            sid = self._samples_writer.add(
                individual=individual_id,
            )
            sample_ids.append(sid)
        return individual_id, sample_ids

    def add_site(
        self,
        position,
        genotypes,
        alleles=None,
        metadata=None,
        inference=None,
        time=None,
        ancestral_allele=None,
    ):
        """
        Adds a new site to this :class:`.SampleData` and returns its ID.

        At a minimum, the new site must specify the ``position`` and
        ``genotypes``. Sites must be added in increasing order of position;
        duplicate positions are **not** supported. For each site a list of
        ``alleles`` may be supplied. This list defines the ancestral and
        derived states at the site. For example, if we set ``alleles=["A",
        "T"]`` then the ancestral state is "A" and the derived state is "T".
        The observed state for each sample is then encoded using the
        ``genotypes`` parameter. Thus if we have ``n`` samples then
        this must be a one dimensional array-like object with length ``n``.
        The ``genotypes`` index into the list of ``alleles``, so that for
        a given array ``g`` and sample index ``j``, ``g[j]`` should contain
        ``0`` if sample ``j`` carries the ancestral state at this site and
        ``1`` if it carries the derived state. For multiple derived states,
        there may be more than 2 ``alleles`, and ``g[j]`` can be greater
        than ``1``, but such sites are not used for inference. All sites must
        have genotypes for the same number of samples.

        All populations and individuals must be defined **before** this method
        is called. If no individuals have been defined using
        :meth:`.add_individual`, the first call to this method adds ``n``
        haploid individuals, where ``n`` is the length of the ``genotypes``
        array.

        :param float position: The floating point position of this site. Must be
            less than the ``sequence_length`` if provided to the :class:`.SampleData`
            constructor. Must be greater than all previously added sites.
        :param arraylike genotypes: An array-like object defining the sample
            genotypes at this site. The array of genotypes corresponds to the
            observed alleles for each sample, represented by indexes into the
            alleles array. Missing sample data can be represented by tskit.MISSING_DATA
            in this array. The input is converted to a numpy array with
            dtype ``np.int8``; therefore, for maximum efficiency ensure
            that the input array is also of this type.
        :param list(str) alleles: A list of strings defining the alleles at this
            site. Only biallelic sites can currently be used for inference. Sites
            with 3 or more non-missing alleles cannot have ``inference`` (below)
            set to ``True``. If missing data is present in the ``genotypes`` array,
            the stored list of alleles will be modified as necessary so that
            ``alleles[tskit.MISSING_DATA] == None``. If ``alleles`` is not specified
            or None, a default of ["0", "1"] is used.
        :param dict metadata: A JSON encodable dict-like object containing
            metadata that is to be associated with this site.
        :param float time: The time of occurence (pastwards) of the mutation to the
            derived state at this site. If not specified or None, the frequency of the
            derived alleles (i.e., the proportion of non-zero values in the genotypes,
            out of all the non-missing values) will be used in inference. For
            biallelic sites this frequency should provide a reasonable estimate
            of the relative time, as used to order ancestral haplotypes during the
            inference process. For sites not used in inference, such as singletons or
            sites with more than two alleles or when the time is specified as
            ``np.nan``, then the value is unused. Defaults to None.
        :param int ancestral_allele: A positive index into the alleles array, specifying
            which allele is the ancestral state, or ``tskit.MISSING_DATA`` (-1) if the
            ancestral state is unknown (in which case the site will not be used for
            inference, and the ancestral state will be inferred using parsimony).
            Default: ``None``, treated as ``0``, so that the first allele in the list
            is taken as the ancestral state.

        :return: The ID of the newly added site.
        :rtype: int
        """
        genotypes = tskit.util.safe_np_int_cast(genotypes, dtype=np.int8)
        self._check_build_mode()
        if self._build_state == self.ADDING_POPULATIONS:
            if genotypes.shape[0] == 0:
                # We could just raise an error here but we set the state
                # here so that we can raise the same error as other
                # similar conditions.
                self._build_state = self.ADDING_SAMPLES
            else:
                # Add in the default haploid samples.
                for _ in range(genotypes.shape[0]):
                    self.add_individual()
        if self._build_state == self.ADDING_SAMPLES:
            self._individuals_writer.flush()
            self._samples_writer.flush()
            self._alloc_site_writer()
            self._build_state = self.ADDING_SITES
            self._last_position = -1
        assert self._build_state == self.ADDING_SITES

        if alleles is None:
            alleles = ["0", "1"]
        n_alleles = len(alleles)
        non_missing = genotypes != MISSING_DATA
        if len(set(alleles)) != n_alleles:
            raise ValueError("Alleles must be distinct")
        if n_alleles > 64:
            # This is mandated by tskit's map_mutations function.
            raise ValueError("Cannot have more than 64 alleles")
        if np.any(genotypes == MISSING_DATA) and alleles[-1] is not None:
            # Don't modify the input parameter
            alleles = list(alleles) + [None]
        if np.any(np.logical_and(genotypes < 0, genotypes != MISSING_DATA)):
            raise ValueError("Non-missing values for genotypes cannot be negative")
        if genotypes.shape != (self.num_samples,):
            raise ValueError(f"Must have {self.num_samples} (num_samples) genotypes.")
        if np.any(genotypes[non_missing] >= n_alleles):
            raise ValueError("Non-missing values for genotypes must be < num alleles")
        if ancestral_allele is None:
            ancestral_allele = 0
        else:
            if ancestral_allele >= n_alleles or ancestral_allele < MISSING_DATA:
                raise ValueError(
                    "ancestral_allele needs to be an index into the alleles array "
                    "or tskit.MISSING_DATA"
                )
        if position < 0:
            raise ValueError("Site position must be > 0")
        if self.sequence_length > 0 and position >= self.sequence_length:
            raise ValueError("Site position must be less than the sequence length")
        if position <= self._last_position:
            raise ValueError(
                "Site positions must be unique and added in increasing order"
            )
        if inference is not None:
            raise ValueError(
                "Inference sites no longer be stored in the sample data file. "
                "Please use the exclude_positions option to generate_ancestors."
            )
        if time is None:
            time = tskit.UNKNOWN_TIME
        site_id = self._sites_writer.add(
            position=position,
            genotypes=genotypes,
            metadata=self._check_metadata(metadata),
            alleles=alleles,
            time=time,
            ancestral_allele=ancestral_allele,
        )
        self._last_position = position
        return site_id

    def append_sites(self, *additional_samples):
        # Append sites from additional sample data objects to the current object. This
        # allows input files (e.g. vcf files) to be read in parallel into separate
        # sample data files and the combined together. The additional samples should have
        # exactly the same populations, individuals, and samples, but with additional
        # sites. The additional sample data objects must be provided in the correct order
        # such that site positions are monotonically ascending.
        # The method is deliberately undocumented, as a more capable way of representing
        # variant data is planned in the future, which should include this functionality.
        self._check_write_modes()
        last_pos = self.sites_position[-1]
        for other in additional_samples:
            other._check_finalised()
            if other.sites_position[0] <= last_pos:
                raise ValueError(
                    "sample data files must be in ascending order of genome position"
                )
            last_pos = other.sites_position[-1]
            if not self.sequence_length == other.sequence_length:
                raise ValueError("sample data files must have the same sequence length")
            if not self.formats_equal(other):
                raise ValueError("sample data files must be of the same format")
            if not self.samples_equal(other):
                raise ValueError("sample data files must have identical samples")
            if not self.individuals_equal(other):
                raise ValueError("sample data files must have identical individuals")
            if not self.populations_equal(other):
                raise ValueError("sample data files must have identical populations")
        for other in additional_samples:
            for name, arr in self.arrays():
                if name.startswith("sites/"):
                    arr.append(other.data[name])

    def finalise(self):
        if self._mode == self.BUILD_MODE:
            if self._build_state == self.ADDING_POPULATIONS:
                raise ValueError("Must add at least one sample individual")
            elif self._build_state == self.ADDING_SAMPLES:
                self._individuals_writer.flush()
                self._samples_writer.flush()
            elif self._build_state == self.ADDING_SITES:
                self._sites_writer.flush()
            if self.num_sites == 0:
                raise ValueError("Must add at least one site")
            self._build_state = -1
            if self.sequence_length == 0:
                # Need to be careful that sequence_length is JSON serialisable here.
                self.data.attrs["sequence_length"] = float(self._last_position) + 1

        super().finalise()

    def __insert_individuals(self, other, pop_id_map=None):
        """
        Helper function to insert all the individuals in this SampleData file
        into the other. If pop_id_map is specified, use it to map
        population IDs in this dataset to IDs in other.
        """
        if pop_id_map is None:
            pop_id_map = {j: j for j in range(other.num_populations)}
            pop_id_map[tskit.NULL] = tskit.NULL
        for individual in other.individuals():
            self.add_individual(
                location=individual.location,
                metadata=individual.metadata,
                time=individual.time,
                flags=individual.flags,
                # We're assuming this is the same for all samples
                population=pop_id_map[individual.population],
                ploidy=len(individual.samples),
            )

    ####################################
    # Read mode
    ####################################

    def merge(self, other, **kwargs):
        """
        Returns a copy of this SampleData file merged with the specified
        other SampleData file. Subsequent keyword arguments are passed
        to the SampleData constructor for the returned merged dataset.

        The datasets are merged by following process:

        1. We add the populations from this dataset to the result, followed
           by the populations from other. Population references from the two
           datasets are updated accordingly.
        2. We add individual data from this dataset to the result, followed
           by the individuals from the other dataset.
        3. We merge the variant data from the two datasets by comparing sites
           by their position. If two sites in the datasets have the same
           position we combine the genotype data. The alleles from this dataset
           are updated to include any new alleles in other, and we then combine
           and update the genotypes accordingly. It is an error if sites with
           the same position have different ancestral state, time, or metadata
           values. For sites that exist in one dataset and not the other,
           we insert the site with ``tskit.MISSING_DATA`` present in the genotypes for
           the dataset that does not contain the site.
        4. We add the provenances for this dataset, followed by the provenances
           for the other dataset.

        :param SampleData other: The other :class:`SampleData` instance to
            to merge.
        :return: A new SampleData instance which contains the merged data
            from the two datasets.
        :rtype: :class:`SampleData`
        """
        self._check_finalised()
        other._check_finalised()
        if self.sequence_length != other.sequence_length:
            raise ValueError("Sample data files must have the same sequence length")
        with SampleData(sequence_length=self.sequence_length, **kwargs) as result:
            # Keep the same population IDs from self.
            for population in self.populations():
                result.add_population(population.metadata)
            # TODO we could avoid duplicate populations here by keying on the
            # metadata. It's slightly complicated by the case where the
            # metadata is all empty, but we could fall through to just
            # adding in all the populations as is, then.
            other_pop_map = {-1: -1}
            for population in other.populations():
                pid = result.add_population(population.metadata)
                other_pop_map[population.id] = pid

            result.__insert_individuals(self)
            result.__insert_individuals(other, other_pop_map)

            for variant in merge_variants(self, other):
                result.add_site(
                    position=variant.site.position,
                    genotypes=variant.genotypes,
                    alleles=variant.alleles,
                    metadata=variant.site.metadata,
                    time=variant.site.time,
                )

            for timestamp, record in list(self.provenances()) + list(
                other.provenances()
            ):
                result.add_provenance(timestamp, record)
            result.record_provenance(command="merge", **kwargs)
        return result

    def sites(self, ids=None):
        """
        Returns an iterator over the Site objects. A subset of the
        sites can be returned using the ``ids`` parameter. This must
        be a list of integer site IDs.
        """
        position_array = self.sites_position[:]
        alleles_array = self.sites_alleles[:]
        metadata_array = self.sites_metadata[:]
        time_array = self.sites_time[:]
        ancestral_allele_array = self.sites_ancestral_allele[:]
        if ids is None:
            ids = np.arange(0, self.num_sites, dtype=int)
        for j in ids:
            anc_idx = ancestral_allele_array[j]
            site = Site(
                id=j,
                position=position_array[j],
                ancestral_allele=anc_idx,
                alleles=tuple(alleles_array[j]),
                metadata=metadata_array[j],
                time=time_array[j],
            )
            yield site

    def num_alleles(self, sites=None):
        """
        Returns a numpy array of the number of alleles at each site. Missing data is
        not counted as an allele.

        :param array sites: A numpy array of sites for which to return data. If None
            (default) return all sites.

        :return: A numpy array of the number of alleles at each site.
        :rtype: numpy.ndarray(dtype=uint32)
        """
        if sites is None:
            sites = np.arange(self.num_sites)
        num_alleles = np.zeros(self.num_sites, dtype=np.uint32)
        for j, alleles in enumerate(self.sites_alleles):
            # Filter out empty alleles (generated by, for example sgkit)
            num_alleles[j] = len(
                [allele for allele in alleles if allele != b"" and allele != ""]
            )
            if alleles[-1] is None:
                num_alleles[j] -= 1
        return num_alleles[sites]

    def variants(self, sites=None, recode_ancestral=None):
        """
        Returns an iterator over the :class:`Variant` objects. This is equivalent to
        the :meth:`tskit.TreeSequence.variants` iterator. If recode_ancestral is
        ``True``, the ``.alleles`` attribute of each variant is guaranteed to return
        the alleles in an order such that the ancestral state is the first item
        in the list. In this case, ``variant.alleles`` may list the alleles in a
        different order from the input order as listed in ``variant.site.alleles``,
        and the values in genotypes array will be recoded so that the ancestral
        state will have a genotype of 0. If the ancestral state is unknown, the
        original input order is kept.

        If a variant contains missing data, it is guaranteed that the alleles
        attribute for that variant satisfies ``alleles[tskit.MISSING_DATA] == None``.

        :param array sites: A numpy array of ascending site ids for which to return
            data. If None (default) return all sites.
        :param bool recode_ancestral: If True, recode genotypes at sites where the
            ancestral state is known such that the ancestral state is coded as 0,
            as described above. Otherwise return genotypes in the input allele encoding.
            Default: ``None`` treated as ``False``.
        :return: An iterator over the variants in the sample data file.
        :rtype: iter(:class:`Variant`)
        """
        if recode_ancestral is None:
            recode_ancestral = False
        all_genotypes = chunk_iterator(self.sites_genotypes, indexes=sites)
        assert MISSING_DATA < 0  # required for geno_map to remap MISSING_DATA
        for genos, site in zip(all_genotypes, self.sites(ids=sites)):
            aa = site.ancestral_allele
            alleles = site.alleles
            if aa != MISSING_DATA and aa > 0 and recode_ancestral:
                # Need to recode this site
                alleles = site.reorder_alleles()
                # re-map the genotypes
                geno_map = np.arange(len(alleles) - MISSING_DATA, dtype=genos.dtype)
                geno_map[MISSING_DATA] = MISSING_DATA
                geno_map[aa] = 0
                geno_map[0:aa] += 1
                genos = geno_map[genos]
            yield Variant(site=site, alleles=alleles, genotypes=genos)

    def _all_haplotypes(self, sites=None, recode_ancestral=None):
        # We iterate over chunks vertically here, and it's not worth complicating
        # the chunk iterator to handle this.
        if recode_ancestral is None:
            recode_ancestral = False
        aa_index = self.sites_ancestral_allele[:]
        # If ancestral allele is missing, keep the order unchanged (aa_index of zero)
        aa_index[aa_index == MISSING_DATA] = 0
        chunk_size = self.sites_genotypes.chunks[1]
        for j in range(self.num_samples):
            if j % chunk_size == 0:
                chunk = self.sites_genotypes[:, j : j + chunk_size].T
            a = chunk[j % chunk_size]
            if recode_ancestral:
                # Remap the genotypes at all sites, depending on the aa_index
                a = np.where(
                    a == aa_index,
                    0,
                    np.where(np.logical_and(a != MISSING_DATA, a < aa_index), a + 1, a),
                )
            yield j, a if sites is None else a[sites]

    def haplotypes(self, samples=None, sites=None, recode_ancestral=None):
        """
        Returns an iterator over the (sample_id, haplotype) pairs. Each haplotype is
        an array of indexes, where the ``i`` th value is an index into the
        alleles list for the ``i`` th specified site (but see warning below).

        .. warning::
            If ``recode_ancestral=True``, the haplotype values may not correspond
            to indexes into the ``sites.alleles`` list. Instead, they will correspond to
            the ``variant.alleles`` list, returned when iterating over :meth:`variants`
            using ``variants(recode_ancestral=True)``.

        :param list samples: The sample IDs for which haplotypes are returned. If
            ``None``, return haplotypes for all sample nodes, otherwise this may be a
            numpy array (or array-like) object (converted to dtype=np.int32).
        :param array sites: A numpy array of sites to use, or ``None`` for all sites.
        :param bool recode_ancestral: If ``True``, recode genotypes so that the
            ancestral state is coded as 0 as described under :meth:`variants`. Otherwise
            return genotypes in the input allele encoding. Default: ``None``,
            treated as ``False``.
        :return: An iterator over (sample_id, haplotype) pairs.
        :rtype: iter(int, numpy.ndarray(dtype=int8))
        """

        if samples is None:
            samples = np.arange(self.num_samples)
        else:
            samples = tskit.util.safe_np_int_cast(samples, dtype=np.int32)
            if np.any(samples[:-1] >= samples[1:]):
                raise ValueError("sample indexes must be in increasing order.")
            if samples.shape[0] > 0 and samples[-1] >= self.num_samples:
                raise ValueError("Sample index too large.")
        j = 0

        for index, a in self._all_haplotypes(sites, recode_ancestral):
            if j == len(samples):
                break
            if index == samples[j]:
                yield index, a
                j += 1

    def individual(self, id_):
        # TODO document
        samples = np.where(self.samples_individual[:] == id_)[0]
        # Make sure the numpy arrays are converted to lists so that
        # we can compare individuals using ==
        return Individual(
            id_,
            location=list(self.individuals_location[id_]),
            metadata=self.individuals_metadata[id_],
            time=self.individuals_time[id_],
            population=self.individuals_population[id_],
            samples=list(samples),
            flags=self.individuals_flags[id_],
        )

    def individuals(self):
        individual_samples = [[] for _ in range(self.num_individuals)]
        for sample_id, individual_id in enumerate(self.samples_individual[:]):
            individual_samples[individual_id].append(sample_id)
        # TODO document
        iterator = zip(
            self.individuals_location[:],
            self.individuals_metadata[:],
            self.individuals_time[:],
            self.individuals_population[:],
            self.individuals_flags[:],
        )
        for j, (location, metadata, time, population, flags) in enumerate(iterator):
            yield Individual(
                j,
                location=list(location),
                metadata=metadata,
                time=time,
                population=population,
                samples=individual_samples[j],
                flags=flags,
            )

    def sample(self, id_):
        # TODO document
        return Sample(
            id_,
            individual=self.samples_individual[id_],
        )

    def samples(self):
        # TODO document
        iterator = self.samples_individual[:]
        for j, individual in enumerate(iterator):
            yield Sample(j, individual=individual)

    def population(self, id_):
        # TODO document
        return Population(id_, metadata=self.populations_metadata[id_])

    def populations(self):
        # TODO document
        iterator = self.populations_metadata[:]
        for j, metadata in enumerate(iterator):
            yield Population(j, metadata=metadata)


class VariantData(SampleData):
    """
    Class representing input variant data used for inference. This is
    mostly a thin wrapper for a Zarr dataset storing information in
    the VCF Zarr (.vcz) format, plus information specifing the ancestral allele
    and (optional) data masks. It then provides various derived properties and
    methods for accessing the data in a form suitable for inference.

    .. note::
        In the VariantData object, "samples" refer to the individuals in the dataset,
        each of which can be of arbitrary ploidy. This is in contrast to ``tskit``,
        in which each *haploid genome* is treated as a separate "sample". For example
        in a diploid dataset, the inferred tree sequence returned at the end of
        the inference process will have ``inferred_ts.num_samples`` equal to double
        the number returned by ``VariantData.num_samples``.

    :param Union(str, zarr.hierarchy.Group) path_or_zarr: The input dataset in
        `VCF Zarr <https://github.com/sgkit-dev/vcf-zarr-spec>`_ format.
        This can either a path to the Zarr dataset saved on disk, or the
        Zarr object itself.
    :param Union(array, str) ancestral_state: A numpy array of strings specifying
        the ancestral states (alleles) used in inference. This must be the same length
        as the number of unmasked sites in the dataset. Alternatively, a single string
        can be provided, giving the name of an array in the input dataset which contains
        the ancestral states. Unknown ancestral states can be specified using "N".
        Any ancestral states which do not match any of the known alleles at that site,
        will be tallied, and a warning issued summarizing the unknown ancestral states.
    :param Union(array, str) sample_mask: A numpy array of booleans specifying which
        samples to mask out (exclude) from the dataset. Alternatively, a string
        can be provided, giving the name of an array in the input dataset which contains
        the sample mask. If ``None`` (default), all samples are included.
    :param Union(array, str) site_mask: A numpy array of booleans specifying which
        sites to mask out (exclude) from the dataset. Alternatively, a string
        can be provided, giving the name of an array in the input dataset which contains
        the site mask. If ``None`` (default), all sites are included (unless restricted
        to a ``contig_id``, see below).
    :param Union(array, str) sites_time: A numpy array of floats specifying the relative
        time of occurrence of the mutation to the derived state at each site. This must
        be of the same length as the number of unmasked sites. Alternatively, a
        string can be provided, giving the name of an array in the input dataset
        which contains the site times. If ``None`` (default), the frequency of the
        derived allele is used as a proxy for the time of occurrence: this is usually a
        reasonable approximation to the relative order of ancestors used for inference.
        Time values are ignored for sites not used in inference, such as singletons,
        sites with more than two alleles, or sites with an unknown ancestral state.
    :param str contig_id: The name of the contig to use (e.g. "chr1"), if the .vcz file
        contains multiple contigs; contig names can be found in the `.contig_id array
        of the input dataset. If provided, sites associated with any other contigs will
        be added to the sites that are masked out. If ``None`` (default), do not mark
        out sites on the basis of their contig ID.
    """

    FORMAT_NAME = "tsinfer-variant-data"
    FORMAT_VERSION = (0, 1)

    def __init__(
        self,
        path_or_zarr,
        ancestral_state,
        *,
        sample_mask=None,
        site_mask=None,
        sites_time=None,
        contig_id=None,
    ):
        try:
            if len(path_or_zarr.call_genotype.shape) == 3:
                # Assumed to be a VCF Zarr hierarchy
                self.path = None
                self.data = path_or_zarr
            else:
                raise ValueError(
                    "Expecting a VCF Zarr object with 3D call_genotype array: "
                    "see https://github.com/sgkit-dev/vcf-zarr-spec/"
                )
        except AttributeError:
            self.path = path_or_zarr
            self.data = zarr.open(path_or_zarr, mode="r")

        genotypes_arr = self.data["call_genotype"]
        _, self._num_individuals_before_mask, self.ploidy = genotypes_arr.shape

        if site_mask is None:
            site_mask = np.full(self.data["variant_position"].shape, False, dtype=bool)
        elif isinstance(site_mask, np.ndarray):
            pass
        else:
            try:
                site_mask = self.data[site_mask][:]
            except KeyError:
                raise ValueError(
                    f"The sites mask {site_mask} was not found" f" in the dataset."
                )
        if site_mask.shape[0] != self.data["variant_position"].shape[0]:
            raise ValueError(
                "Site mask array must be the same length as the number of unmasked sites"
            )
        if contig_id is not None:
            contig_index = np.where(self.data.contig_id[:] == contig_id)[0]
            if len(contig_index) == 0:
                raise ValueError(
                    f'"{contig_id}" not found among the available contig IDs: '
                    + ",".join(f"{n}" for n in self.data.contig_id[:])
                )
            elif len(contig_index) > 1:
                raise ValueError(f'Multiple contigs named "{contig_id}"')
            contig_index = contig_index[0]
            site_mask = np.logical_or(
                site_mask, self.data["variant_contig"][:] != contig_index
            )

        # We negate the mask as it is much easier in numpy to have True=keep
        self.sites_select = ~site_mask.astype(bool)
        if np.sum(self.sites_select) == 0:
            raise ValueError("All sites have been masked out. Please unmask some")

        if sample_mask is None:
            sample_mask = np.full(self._num_individuals_before_mask, False, dtype=bool)
        elif isinstance(sample_mask, np.ndarray):
            pass
        else:
            try:
                # We negate the mask as it is much easier in numpy to have True=keep
                sample_mask = self.data[sample_mask][:]
            except KeyError:
                raise ValueError(
                    f"The samples mask {sample_mask} was not" f" found in the dataset."
                )
        if sample_mask.shape[0] != self._num_individuals_before_mask:
            raise ValueError(
                "Samples mask array must be the same length as the number of"
                " individuals"
            )
        self.individuals_select = ~sample_mask.astype(bool)

        self._num_sites = np.sum(self.sites_select)
        assert self.ploidy == self.data["call_genotype"].chunks[2]
        if self.ploidy > 1:
            if "call_genotype_phased" not in self.data:
                raise ValueError(
                    "The call_genotype_phased array is missing from the"
                    " zarr dataset, indicating that all the genotypes are"
                    " unphased"
                )

        used_contigs = self.data.variant_contig[:][self.sites_select]
        self._contig_index = used_contigs[0]
        self._contig_id = self.data.contig_id[self._contig_index]

        if np.any(used_contigs != self._contig_index):
            contig_names = ", ".join(
                f'"{self.data.contig_id[c]}"' for c in np.unique(used_contigs)
            )
            raise ValueError(
                f"Sites belong to multiple contigs ({contig_names}). Please restrict "
                "sites to one contig e.g. via the `contig_id` argument."
            )

        if np.any(np.diff(self.sites_position) <= 0):
            raise ValueError(
                "Values taken from the variant_position array are not strictly "
                "increasing (i.e. have duplicate or out-of-order values). "
                "These must be masked out to run tsinfer."
            )

        if sites_time is None:
            self._sites_time = np.full(self.num_sites, tskit.UNKNOWN_TIME)
        elif isinstance(sites_time, np.ndarray):
            if sites_time.shape[0] != self.num_sites:
                raise ValueError(
                    "Sites time array must be the same length as the number of selected"
                    " sites"
                )
            self._sites_time = sites_time
        else:
            try:
                self._sites_time = self.data[sites_time][:][self.sites_select]
            except KeyError:
                raise ValueError(
                    f"The sites time array {sites_time} was not found in the dataset"
                )

        if isinstance(ancestral_state, np.ndarray):
            if ancestral_state.shape[0] != self.num_sites:
                raise ValueError(
                    "Ancestral state array must be the same length as the number of"
                    " selected sites"
                )
        else:
            try:
                ancestral_state = self.data[ancestral_state][:][self.sites_select]
            except KeyError:
                raise ValueError(
                    f"The ancestral state array {ancestral_state} was not"
                    f" found in the dataset."
                )
        ancestral_state = ancestral_state.astype(str)
        if np.any(ancestral_state == ""):
            raise ValueError("Ancestral state array cannot contain empty strings")

        self._sites_ancestral_allele = np.full(self.num_sites, -1, dtype=np.int8)
        unknown_alleles = collections.Counter()
        for i, (anc_state, site) in enumerate(zip(ancestral_state, self.sites())):
            if anc_state in {"N", "n"} or anc_state not in site.alleles:
                unknown_alleles[anc_state] += 1
            else:
                self._sites_ancestral_allele[i] = site.alleles.index(anc_state)
        deliberately_unknown = sum([unknown_alleles.get(c, 0) for c in ("N", "n")])
        tot = sum(unknown_alleles.values())
        if tot > 0:
            frac_bad = tot / self.num_sites
            frac_bad_per_type = [v / self.num_sites for v in unknown_alleles.values()]
            summarise_unknown = [
                f"'{k}': {v} ({frac * 100:.2f}% of sites)"  # Summarise per allele type
                for (k, v), frac in zip(unknown_alleles.items(), frac_bad_per_type)
            ]
            if tot == deliberately_unknown:
                logging.info(
                    f"{tot} sites ({frac_bad * 100 :.2f}%) were deliberately marked as "
                    "of unknown ancestral state. They will not be used for inference"
                )
            else:
                warnings.warn(
                    "An ancestral allele was not found in the variant_allele array for "
                    + f"the {tot} sites ({frac_bad * 100 :.2f}%) listed below. "
                    + "They will be treated as of unknown ancestral state:\n "
                    + "\n ".join(summarise_unknown)
                )

        # Create zarr arrays for convenience when iterating over chunks
        self.z_sites_select = zarr.array(
            self.sites_select, chunks=self.data["call_genotype"].chunks[0], dtype=bool
        )
        # Find the first chunk from the left and right that contains an unmasked site
        self.sites_used_chunks = []
        for sites_chunk in range(self.z_sites_select.cdata_shape[0]):
            if np.sum(self.z_sites_select.blocks[sites_chunk]) > 0:
                self.sites_used_chunks.append(sites_chunk)

        logger.info(f"Number of sites after applying mask: {self.num_sites}")
        logger.info(
            f"Sites chunks used: {len(self.sites_used_chunks)} - "
            f"of {self.z_sites_select.cdata_shape[0]}"
        )
        logger.info(
            f"Number of individuals after applying mask: {self.num_individuals}"
        )

    @functools.cached_property
    def format_name(self):
        return self.FORMAT_NAME

    @functools.cached_property
    def format_version(self):
        return self.FORMAT_VERSION

    @property
    def finalised(self):
        return True

    @functools.cached_property
    def sequence_length(self):
        """
        The sequence length of the contig associated with sites used in the dataset.
        If the dataset has a "sequence_length" attribute, this is always used, otherwise
        if the dataset has recorded contig lengths, the appropriate length is taken,
        otherwise the length is calculated from the maximum variant position plus one.
        """
        try:
            return self.data.attrs["sequence_length"]
        except KeyError:
            if self._contig_index is not None:
                try:
                    if self._contig_index < len(self.data.contig_length):
                        return self.data.contig_length[self._contig_index]
                except AttributeError:
                    pass  # contig_length is optional, fall back to calculating length
        return int(np.max(self.data["variant_position"])) + 1

    @property
    def contig_id(self):
        """
        The contig ID (name) for all used sites, or None if no
        contig IDs were provided
        """
        return self._contig_id

    @property
    def num_sites(self):
        return self._num_sites

    @functools.cached_property
    def samples_select(self):
        # Samples in sgkit are individuals in tskit, so we need to expand
        # the mask to cover all the samples for each individual.
        return np.repeat(self.individuals_select, self.ploidy)

    @functools.cached_property
    def sites_metadata_schema(self):
        try:
            return tskit.metadata.parse_metadata_schema(
                self.data.attrs["sites_metadata_schema"]
            ).schema
        except KeyError:
            return tskit.MetadataSchema.permissive_json().schema

    @functools.cached_property
    def sites_metadata(self):
        schema = tskit.MetadataSchema(self.sites_metadata_schema)
        try:
            return [
                schema.decode_row(r)
                for r in self.data["sites_metadata"][:][self.sites_select]
            ]
        except KeyError:
            return [{} for _ in range(self.num_sites)]

    @property
    def sites_time(self):
        return self._sites_time

    @functools.cached_property
    def sites_position(self):
        return self.data["variant_position"][:][self.sites_select]

    @functools.cached_property
    def sites_alleles(self):
        return self.data["variant_allele"][:][self.sites_select].astype(str)

    @property
    def sites_ancestral_allele(self):
        return self._sites_ancestral_allele

    @functools.cached_property
    def sites_genotypes(self):
        gt = self.data["call_genotype"]
        # This method is only used for test/debug so we retrieve and
        # reshape the entire array.
        ret = gt[...][self.sites_select, :, :]
        ret = ret[:, self.individuals_select, :]
        return ret.reshape(ret.shape[0], ret.shape[1] * ret.shape[2])

    @functools.cached_property
    def provenances_timestamp(self):
        try:
            return self.data["provenances_timestamp"]
        except KeyError:
            return np.array([], dtype=object)

    @functools.cached_property
    def provenances_record(self):
        try:
            return [json.loads(r) for r in self.data["provenances_record"]]
        except KeyError:
            return np.array([], dtype=object)

    @functools.cached_property
    def num_samples(self):
        return np.sum(self.samples_select)

    @functools.cached_property
    def samples_individual(self):
        ret = np.zeros((self.num_samples), dtype=np.int32)
        for p in range(self.ploidy):
            ret[p :: self.ploidy] = np.arange(self.num_individuals)
        return ret

    @functools.cached_property
    def metadata_schema(self):
        try:
            return tskit.metadata.parse_metadata_schema(
                self.data.attrs["metadata_schema"]
            ).schema
        except KeyError:
            return tskit.MetadataSchema.permissive_json().schema

    @functools.cached_property
    def metadata(self):
        try:
            return tskit.MetadataSchema(self.metadata_schema).decode_row(
                self.data.attrs["metadata"].encode()
            )
        except KeyError:
            return {}

    @functools.cached_property
    def num_populations(self):
        try:
            return len(self.data["populations_metadata"])
        except KeyError:
            return 0

    @functools.cached_property
    def populations_metadata(self):
        schema = tskit.MetadataSchema(self.populations_metadata_schema)
        try:
            return [schema.decode_row(r) for r in self.data["populations_metadata"][:]]
        except KeyError:
            return [{} for _ in range(self.num_populations)]

    @functools.cached_property
    def populations_metadata_schema(self):
        try:
            return tskit.metadata.parse_metadata_schema(
                self.data.attrs["populations_metadata_schema"]
            ).schema
        except KeyError:
            return tskit.MetadataSchema.permissive_json().schema

    @property
    def num_individuals(self):
        return np.sum(self.individuals_select)

    @functools.cached_property
    def individuals_time(self):
        try:
            return self.data["individuals_time"][:][self.individuals_select]
        except KeyError:
            return np.full(self.num_individuals, tskit.UNKNOWN_TIME)

    @functools.cached_property
    def individuals_metadata_schema(self):
        try:
            return tskit.metadata.parse_metadata_schema(
                self.data.attrs["individuals_metadata_schema"]
            ).schema
        except KeyError:
            return tskit.MetadataSchema.permissive_json().schema

    @functools.cached_property
    def individuals_metadata(self):
        schema = tskit.MetadataSchema(self.individuals_metadata_schema)
        # We set the sample_id in the individual metadata as this is often useful,
        # however we silently don't overwrite if the key exists
        if "individuals_metadata" in self.data:
            assert (
                len(self.data["individuals_metadata"])
                == self._num_individuals_before_mask
            )
            assert self._num_individuals_before_mask == len(self.data["sample_id"])
            md_list = []
            for sample_id, r in zip(
                self.data["sample_id"][:][self.individuals_select],
                self.data["individuals_metadata"][:][self.individuals_select],
            ):
                md = schema.decode_row(r)
                if "variant_data_sample_id" not in md:
                    md["variant_data_sample_id"] = sample_id
                md_list.append(md)
            return md_list
        else:
            return [
                {"variant_data_sample_id": sample_id}
                for sample_id in self.data["sample_id"][:][self.individuals_select]
            ]

    @functools.cached_property
    def individuals_location(self):
        try:
            return self.data["individuals_location"][:][self.individuals_select]
        except KeyError:
            return np.array([[]] * self.num_individuals, dtype=float)

    @functools.cached_property
    def individuals_population(self):
        try:
            return self.data["individuals_population"][:][self.individuals_select]
        except KeyError:
            return np.full((self.num_individuals), tskit.NULL, dtype=np.int32)

    @functools.cached_property
    def individuals_flags(self):
        try:
            return self.data["individuals_flags"][:][self.individuals_select]
        except KeyError:
            return np.full((self.num_individuals), 0, dtype=np.int32)

    @staticmethod
    def _trim_allele_array(allele_array, site_id):
        # Trim a list of allelic states to remove any trailing "" entries.
        assert allele_array.shape[0] > 0
        used = np.flatnonzero(allele_array != "")
        allele_array = allele_array[: (used[-1] + 1)]
        if np.any(allele_array == ""):
            raise ValueError(
                f'Bad alleles: fill value "" in middle of list: {allele_array}'
            )
        if len(set(allele_array)) != len(allele_array):
            raise ValueError(f"Duplicate allele values provided at site {site_id}")
        return allele_array

    def sites(self, ids=None):
        """
        Returns an iterator over the Site objects. A subset of the
        sites can be returned using the ``ids`` parameter. This must
        be a list of integer site IDs.
        """
        position_array = self.sites_position[:]
        alleles_array = self.sites_alleles[:]
        metadata_array = self.sites_metadata[:]
        time_array = self.sites_time[:]
        ancestral_allele_array = self.sites_ancestral_allele[:]
        if ids is None:
            ids = np.arange(0, self.num_sites, dtype=int)
        for j in ids:
            anc_idx = ancestral_allele_array[j]
            site = Site(
                id=j,
                position=position_array[j],
                ancestral_allele=anc_idx,
                alleles=tuple(self._trim_allele_array(alleles_array[j], j)),
                metadata=metadata_array[j],
                time=time_array[j],
            )
            yield site

    def variants(self, sites=None, recode_ancestral=None):
        """
        Returns an iterator over the :class:`Variant` objects. This is equivalent to
        the :meth:`tskit.TreeSequence.variants` iterator. If recode_ancestral is
        ``True``, the ``.alleles`` attribute of each variant is guaranteed to return
        the alleles in an order such that the ancestral state is the first item
        in the list. In this case, ``variant.alleles`` may list the alleles in a
        different order from the input order as listed in ``variant.site.alleles``,
        and the values in genotypes array will be recoded so that the ancestral
        state will have a genotype of 0. If the ancestral state is unknown, the
        original input order is kept.

        :param array sites: A numpy array of ascending site ids for which to return
            data. If None (default) return all sites.
        :param bool recode_ancestral: If True, recode genotypes at sites where the
            ancestral state is known such that the ancestral state is coded as 0,
            as described above. Otherwise return genotypes in the input allele encoding.
            Default: ``None`` treated as ``False``.
        :return: An iterator over the variants in the sample data file.
        :rtype: iter(:class:`Variant`)
        """
        if recode_ancestral is None:
            recode_ancestral = False
        all_genotypes = chunk_iterator(
            self.data["call_genotype"],
            indexes=sites,
            select=self.sites_select,
            orthogonal_select=self.individuals_select,
        )
        assert MISSING_DATA < 0  # required for geno_map to remap MISSING_DATA
        for genos, site in zip(all_genotypes, self.sites(ids=sites)):
            # We have an extra ploidy dimension when coming from sgkit
            genos = genos.reshape(self.num_samples)
            aa = site.ancestral_allele
            alleles = site.alleles
            if aa != MISSING_DATA and aa > 0 and recode_ancestral:
                # Need to recode this site
                alleles = site.reorder_alleles()
                # re-map the genotypes
                geno_map = np.arange(len(alleles) - MISSING_DATA, dtype=genos.dtype)
                geno_map[MISSING_DATA] = MISSING_DATA
                geno_map[aa] = 0
                geno_map[0:aa] += 1
                genos = geno_map[genos]
            # Empty alleles (padded by sgkit) should never be seen
            assert all(a != "" for a in alleles)
            yield Variant(site=site, alleles=alleles, genotypes=genos)

    def _all_haplotypes(self, sites=None, recode_ancestral=None, samples_slice=None):
        # We iterate over chunks vertically here, and it's not worth complicating
        # the chunk iterator to handle this.
        if samples_slice is None:
            samples_slice = (0, self.num_samples)
        if samples_slice[0] % self.ploidy != 0 or samples_slice[1] % self.ploidy != 0:
            raise ValueError("Samples slice must be a multiple of ploidy")
        # Make an individuals mask that respects the samples slice
        ind_indexes = np.where(self.individuals_select)[0]
        ind_select = zarr.zeros(
            self.individuals_select.shape,
            chunks=self.data["call_genotype"].chunks[1],
            dtype=bool,
        )
        ind_select[
            ind_indexes[
                samples_slice[0] // self.ploidy : samples_slice[1] // self.ploidy
            ]
        ] = True

        sample_index = samples_slice[0]
        for ind_chunk_i in range(ind_select.cdata_shape[0]):
            ind_select_chunk = ind_select.blocks[ind_chunk_i]
            num_samples_in_chunk = np.sum(ind_select_chunk) * self.ploidy
            if num_samples_in_chunk == 0:
                continue
            final_data = np.zeros((self.num_sites, num_samples_in_chunk), dtype=np.int8)
            site_insertion_position = 0
            for sites_chunk_i in self.sites_used_chunks:
                site_select = self.z_sites_select.blocks[sites_chunk_i]
                if np.sum(site_select) == 0:
                    continue
                gt_chunk = self.data["call_genotype"].blocks[
                    sites_chunk_i, ind_chunk_i, :
                ]
                gt_chunk = gt_chunk[site_select, :, :]
                gt_chunk = gt_chunk[:, ind_select_chunk, :]
                gt_chunk = gt_chunk.reshape(len(gt_chunk), num_samples_in_chunk)
                final_data[
                    site_insertion_position : site_insertion_position + len(gt_chunk), :
                ] = gt_chunk
                site_insertion_position += len(gt_chunk)
            assert site_insertion_position == self.num_sites
            for s in range(num_samples_in_chunk):
                a = final_data[:, s]
                if recode_ancestral:
                    # Remap the genotypes at all sites, depending on the aa_index
                    a = np.where(
                        a == self.sites_ancestral_allele,
                        0,
                        np.where(
                            np.logical_and(
                                a != MISSING_DATA, a < self.sites_ancestral_allele
                            ),
                            a + 1,
                            a,
                        ),
                    )
                yield sample_index, a if sites is None else a[sites]
                sample_index += 1
        assert sample_index == samples_slice[1]

    @classmethod
    def from_tree_sequence(*args, **kwargs):
        raise NotImplementedError("This method is not implemented for VariantData")


@attr.s(order=False, eq=False)
class Ancestor:
    """
    An ancestor object.
    """

    # TODO document properly.
    id = attr.ib()
    start = attr.ib()
    end = attr.ib()
    time = attr.ib()
    focal_sites = attr.ib()
    full_haplotype = attr.ib()

    @property
    def haplotype(self):
        return self.full_haplotype[self.start : self.end]

    def __eq__(self, other):
        return (
            self.id == other.id
            and self.start == other.start
            and self.end == other.end
            and self.time == other.time
            and np.array_equal(self.focal_sites, other.focal_sites)
            and np.array_equal(self.full_haplotype, other.full_haplotype)
        )


class AncestorData(DataContainer):
    """
    AncestorData(position, sequence_length, *, path=None, num_flush_threads=0, \
    compressor=None, chunk_size=1024, max_file_size=None)

    Class representing the stored ancestor data produced by
    :func:`generate_ancestors`. See the ancestor data file format
    :ref:`specifications <sec_file_formats_ancestors>` for details on the structure
    of this file.

    See the documentation for :class:`SampleData` for a discussion of the
    ``max_file_size`` parameter.

    :param arraylike position: Integer array of the site positions of the ancestors.
        All values should be >0 and the array should be monotonically increasing.
    :param float sequence_length: Total length of the sequence, site positions must
        be less than this value.
    :param str path: The path of the file to store the ancestor data. If None,
        the information is stored in memory and not persistent.
    :param int num_flush_threads: The number of background threads to use
        for compressing data and flushing to disc. If <= 0, do not spawn
        any threads but use a synchronous algorithm instead. Default=0.
    :param numcodecs.abc.Codec compressor: A :class:`numcodecs.abc.Codec`
        instance to use for compressing data. Any codec may be used, but
        problems may occur with very large datasets on certain codecs as
        they cannot compress buffers >2GB. If None, do not use any compression.
        Default=:class:`numcodecs.zstd.Zstd`.
    :param int chunk_size: The chunk size used for
         `zarr arrays <http://zarr.readthedocs.io/>`_ in the sample dimension. This
         affects compression level and algorithm performance. Default=1024.
    :param int chunk_size_sites: The chunk size used for the genotype
        `zarr arrays <http://zarr.readthedocs.io/>`_ in the sites dimension. This affects
        compression level and algorithm performance. Default=16384.
    :param int max_file_size: If a file is being used to store this data, set
        a maximum size in bytes for the stored file. If None, the default
        value of 1GiB is used on Windows and 1TiB on other
        platforms (see above for details).
    """

    FORMAT_NAME = "tsinfer-ancestor-data"
    FORMAT_VERSION = (3, 0)

    def __init__(self, position, sequence_length, chunk_size_sites=None, **kwargs):
        super().__init__(**kwargs)
        self._last_time = 0
        self.inference_sites_set = False
        if chunk_size_sites is None:
            self._chunk_size_sites = 16384
        else:
            self._chunk_size_sites = chunk_size_sites

        self.data.attrs["sequence_length"] = sequence_length
        if self.sequence_length <= 0:
            raise ValueError("Bad samples file: sequence_length cannot be zero or less")

        # We specify fill_value here due to https://github.com/pydata/xarray/issues/7292
        self.create_dataset("sample_start", dtype=np.int32)
        self.create_dataset("sample_end", dtype=np.int32)
        self.create_dataset("sample_time", dtype=np.float64)
        self.create_dataset("sample_focal_sites", dtype="array:i4")

        self.create_dataset(
            "variant_position",
            data=position,
            shape=position.shape,
            chunks=self._chunk_size_sites,
            dtype=np.float64,
            dimensions=["variants"],
        )

        # We have to include a ploidy dimension sgkit compatibility
        a = self.create_dataset(
            "call_genotype",
            dtype="i1",
            shape=(self.num_sites, 0, 1),
            chunks=(self._chunk_size_sites, self._chunk_size, 1),
            dimensions=["variants", "samples", "ploidy"],
        )
        a.attrs["mixed_ploidy"] = False

        a = self.create_dataset(
            "call_genotype_mask",
            dtype="i1",
            shape=(self.num_sites, 0, 1),
            chunks=(self._chunk_size_sites, self._chunk_size, 1),
            dimensions=["variants", "samples", "ploidy"],
        )
        # We add this to be identical to sgkit generated arrays
        a.attrs["dtype"] = "bool"

        self._alloc_ancestor_writer()

    def create_dataset(
        self,
        name,
        *,
        data=None,
        shape=None,
        chunks=None,
        dtype=None,
        compressor=None,
        dimensions=None,
    ):
        if shape is None:
            shape = (0,)
        if chunks is None:
            chunks = self._chunk_size
        if compressor is None:
            compressor = self._compressor
        if dimensions is None:
            dimensions = ["samples"]

        ds = self.data.create_dataset(
            name,
            data=data,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=None,
        )
        ds.attrs["_ARRAY_DIMENSIONS"] = dimensions
        return ds

    def _alloc_ancestor_writer(self):
        self.ancestor_writer = BufferedItemWriter(
            {
                "start": self.ancestors_start,
                "end": self.ancestors_end,
                "time": self.ancestors_time,
                "focal_sites": self.ancestors_focal_sites,
                "full_haplotype": self.ancestors_full_haplotype,
                "full_haplotype_mask": self.ancestors_full_haplotype_mask,
            },
            num_threads=self._num_flush_threads,
        )

    def summary(self):
        return "AncestorData(num_ancestors={}, num_sites={})".format(
            self.num_ancestors, self.num_sites
        )

    def __str__(self):
        values = [
            ("sequence_length", self.sequence_length),
            ("num_ancestors", self.num_ancestors),
            ("num_sites", self.num_sites),
            ("variant_position", zarr_summary(self.sites_position)),
            ("sample_start", zarr_summary(self.ancestors_start)),
            ("sample_end", zarr_summary(self.ancestors_end)),
            ("sample_time", zarr_summary(self.ancestors_time)),
            ("sample_focal_sites", zarr_summary(self.ancestors_focal_sites)),
            ("call_genotype", zarr_summary(self.ancestors_full_haplotype)),
        ]
        return super().__str__() + self._format_str(values)

    def data_equal(self, other):
        """
        Returns True if all the data attributes of this input file and the
        specified input file are equal. This compares every attribute.
        """
        return (
            self.sequence_length == other.sequence_length
            and self.format_name == other.format_name
            and self.format_version == other.format_version
            and self.num_ancestors == other.num_ancestors
            and self.num_sites == other.num_sites
            and np.array_equal(self.sites_position[:], other.sites_position[:])
            and np.array_equal(self.ancestors_start[:], other.ancestors_start[:])
            and np.array_equal(self.ancestors_end[:], other.ancestors_end[:])
            # Need to take a different approach with np object arrays.
            and np_obj_equal(
                self.ancestors_focal_sites[:], other.ancestors_focal_sites[:]
            )
            # TODO For large sets of ancestors, this needs to be done chunk-wise
            and np_obj_equal(
                self.ancestors_full_haplotype[:], other.ancestors_full_haplotype[:]
            )
        )

    def assert_data_equal(self, other):
        if self.data_equal(other):
            return

        assert self.format_name == other.format_name
        assert self.format_version == other.format_version
        assert self.num_ancestors == other.num_ancestors
        assert self.num_sites == other.num_sites
        np.testing.assert_array_equal(self.sites_position[:], other.sites_position[:])
        np.testing.assert_array_equal(self.ancestors_start[:], other.ancestors_start[:])
        np.testing.assert_array_equal(self.ancestors_end[:], other.ancestors_end[:])
        fc_self = self.ancestors_focal_sites[:]
        fc_other = other.ancestors_focal_sites[:]
        assert len(fc_self) == len(fc_other)
        for sites_self, sites_other in zip(fc_self, fc_other):
            np.testing.assert_array_equal(sites_self, sites_other)
        haps_self = self.ancestors_full_haplotype[:]
        haps_other = other.ancestors_full_haplotype[:]
        for hap_self, hap_other in zip(haps_self, haps_other):
            np.testing.assert_array_equal(hap_self.T, hap_other.T)
        # Put this assert last to have an easy to change attribute so we can
        # test this function.
        assert self.sequence_length == other.sequence_length
        raise AssertionError("Bug in this function")

    @property
    def sequence_length(self):
        """
        Returns the sequence length.
        """
        return self.data.attrs["sequence_length"]

    @property
    def num_ancestors(self):
        return self.ancestors_start.shape[0]

    @functools.cached_property
    def num_sites(self):
        """
        The number of inference sites used to generate the ancestors
        """
        return self.sites_position.shape[0]

    @property
    def sites_position(self):
        """
        The positions of the inference sites used to generate the ancestors
        """
        return self.data["variant_position"]

    @property
    def ancestors_start(self):
        return self.data["sample_start"]

    @property
    def ancestors_end(self):
        return self.data["sample_end"]

    @property
    def ancestors_time(self):
        return self.data["sample_time"]

    @property
    def ancestors_focal_sites(self):
        return self.data["sample_focal_sites"]

    @property
    def ancestors_full_haplotype(self):
        # Named and shaped to be compatible with sgkit
        return self.data["call_genotype"]

    @property
    def ancestors_full_haplotype_mask(self):
        # Only required for sgkit compatibility
        return self.data["call_genotype_mask"]

    @property
    def ancestors_length(self):
        """
        Returns the length of ancestors in physical coordinates.
        """
        # Ancestor start and end are half-closed. The last site is assumed
        # to cover the region up to sequence length.
        pos = np.hstack([self.sites_position[:], [self.sequence_length]])
        start = self.ancestors_start[:]
        end = self.ancestors_end[:]
        return pos[end] - pos[start]

    def insert_proxy_samples(
        self,
        sample_data,
        *,
        sample_ids=None,
        epsilon=None,
        allow_mutation=False,
        require_same_sample_data=True,
        **kwargs,
    ):
        """
        Take a set of samples from a ``sample_data`` instance and create additional
        "proxy sample ancestors" from them, returning a new :class:`.AncestorData`
        instance including both the current ancestors and the additional ancestors
        at the appropriate time points.

        A *proxy sample ancestor* is an ancestor based upon a known sample. At
        sites used in the full inference process, the haplotype of this ancestor
        is identical to that of the sample on which it is based. The time of the
        ancestor is taken to be a fraction ``epsilon`` older than the sample on
        which it is based.

        A common use of this function is to provide ancestral nodes for anchoring
        historical samples at the correct time when matching them into a tree
        sequence during the :func:`tsinfer.match_samples` stage of inference.
        For this reason, by default, the samples chosen from ``sample_data``
        are those associated with historical (i.e. non-contemporary)
        :ref:`individuals <sec_inference_data_model_individual>`. This can be
        altered by using the ``sample_ids`` parameter.

        .. note::

            The proxy sample ancestors inserted here will correspond to extra nodes
            in the inferred tree sequence. At sites which are not used in the full
            inference process (e.g. sites unique to a single historical sample),
            these proxy sample ancestor nodes may have a different genotype from
            their corresponding sample.

        :param SampleData sample_data: The :class:`.SampleData` instance
            from which to select the samples used to create extra ancestors.
        :param list(int) sample_ids: A list of sample ids in the ``sample_data``
            instance that will be selected to create the extra ancestors. If
            ``None`` (default) select all the historical samples, i.e. those
            associated with an :ref:`sec_inference_data_model_individual` whose
            time is greater than zero. The order of ids is ignored, as are
            duplicate ids.
        :param list(float) epsilon: An list of small time increments
            determining how much older each proxy sample ancestor is than the
            corresponding sample listed in ``sample_ids``. A single value is also
            allowed, in which case it is used as the time increment for all selected
            proxy sample ancestors. If None (default) find :math:`{\\delta}t`, the
            smallest time difference between between the sample times and the next
            oldest ancestor in the current :class:`.AncestorData` instance, setting
            ``epsilon`` = :math:`{\\delta}t / 100` (or, if all selected samples
            are at least as old as the oldest ancestor, take :math:`{\\delta}t`
            to be the smallest non-zero time difference between existing ancestors).
        :param bool allow_mutation: If ``False`` (the default), any site in a proxy
            sample ancestor that has a derived allele must have a pre-existing
            mutation in an older (non-proxy) ancestor, otherwise an error is raised.
            Alternatively, if ``allow_mutation`` is ``True``, proxy ancestors can
            contain a de-novo mutation at a site that also has a mutation elsewhere
            (i.e. breaking the infinite sites assumption), allowing them to possess
            derived alleles at sites where there are no pre-existing mutations in
            older ancestors.
        :param bool require_same_sample_data: **Deprecated** Has no effect.
        :param \\**kwargs: Further arguments passed to the constructor when creating
            the new :class:`AncestorData` instance which will be returned.

        :return: A new :class:`.AncestorData` object.
        :rtype: AncestorData
        """
        self._check_finalised()
        sample_data._check_finalised()
        if self.sequence_length != sample_data.sequence_length:
            raise ValueError("sample_data does not have the correct sequence length")
        used_sites = np.isin(sample_data.sites_position[:], self.sites_position[:])
        if np.sum(used_sites) != self.num_sites:
            raise ValueError("Genome positions in ancestors missing from sample_data")

        if sample_ids is None:
            sample_ids = []
            for i in sample_data.individuals():
                if i.time > 0:
                    sample_ids += i.samples
        # sort by ID and make unique for quick haplotype access
        sample_ids, unique_indices = np.unique(np.array(sample_ids), return_index=True)

        sample_times = np.zeros(len(sample_ids), dtype=self.ancestors_time.dtype)
        for i, s in enumerate(sample_ids):
            sample = sample_data.sample(s)
            if sample.individual != tskit.NULL:
                sample_times[i] = sample_data.individual(sample.individual).time

        if epsilon is not None:
            epsilons = np.atleast_1d(epsilon)
            if len(epsilons) == 1:
                # all get the same epsilon
                epsilons = np.repeat(epsilons, len(sample_ids))
            else:
                if len(epsilons) != len(unique_indices):
                    raise ValueError(
                        "The number of epsilon values must equal the number of "
                        f"sample_ids ({len(sample_ids)})"
                    )
                epsilons = epsilons[unique_indices]

        else:
            anc_times = self.ancestors_time[:][::-1]  # find ascending time order
            older_index = np.searchsorted(anc_times, sample_times, side="right")
            # Don't include times older than the oldest ancestor
            allowed = older_index < self.num_ancestors
            if np.sum(allowed) > 0:
                delta_t = anc_times[older_index[allowed]] - sample_times[allowed]
            else:
                # All samples have times equal to or older than the oldest curr ancestor
                time_diffs = np.diff(anc_times)
                delta_t = np.min(time_diffs[time_diffs > 0])
            epsilons = np.repeat(np.min(delta_t) / 100.0, len(sample_ids))

        proxy_times = sample_times + epsilons
        reverse_time_sorted_indexes = np.argsort(proxy_times)[::-1]
        # In cases where we have more than a handful of samples to use as proxies, it is
        # inefficient to access the haplotypes out of order, so we iterate and cache
        # (caution: the haplotypes list may be quite large in this case)
        haplotypes = [
            h[1] for h in sample_data.haplotypes(samples=sample_ids, sites=used_sites)
        ]

        with AncestorData(
            sample_data.sites_position[:][used_sites],
            sample_data.sequence_length,
            **kwargs,
        ) as other:
            mutated_sites = set()  # To check if mutations have ocurred yet
            ancestors_iter = self.ancestors()
            ancestor = next(ancestors_iter, None)
            for i in reverse_time_sorted_indexes:
                proxy_time = proxy_times[i]
                sample_id = sample_ids[i]
                haplotype = haplotypes[i]
                while ancestor is not None and ancestor.time > proxy_time:
                    ancestor_dict = attr.asdict(
                        ancestor, filter=exclude_id_and_full_haplotype
                    )
                    ancestor_dict["haplotype"] = ancestor.haplotype
                    other.add_ancestor(**ancestor_dict)
                    mutated_sites.update(ancestor.focal_sites)
                    ancestor = next(ancestors_iter, None)
                if not allow_mutation:
                    derived_sites = set(np.where(haplotype > 0)[0])
                    if not derived_sites.issubset(mutated_sites):
                        raise ValueError(
                            f"Sample {sample_id} contains a new derived allele, which "
                            "requires a novel mutation, but `allow_mutation` is False."
                        )
                logger.debug(
                    f"Inserting proxy ancestor: sample {sample_id} at time {proxy_time}"
                )
                other.add_ancestor(
                    start=0,
                    end=self.num_sites,
                    time=proxy_time,
                    focal_sites=[],
                    haplotype=haplotype,
                )
            # Add any ancestors remaining in the current instance
            while ancestor is not None:
                ancestor_dict = attr.asdict(
                    ancestor, filter=exclude_id_and_full_haplotype
                )
                ancestor_dict["haplotype"] = ancestor.haplotype
                other.add_ancestor(**ancestor_dict)
                ancestor = next(ancestors_iter, None)

            # TODO - set metadata on these ancestors, once ancestors have metadata
            other.clear_provenances()
            for timestamp, record in self.provenances():
                other.add_provenance(timestamp, record)
            other.record_provenance(command="insert_proxy_samples", **kwargs)

        assert other.num_ancestors == self.num_ancestors + len(sample_ids)
        return other

    def truncate_ancestors(
        self,
        lower_time_bound,
        upper_time_bound,
        length_multiplier=2,
        buffer_length=1000,
        **kwargs,
    ):
        """
        Truncates the length of ancestors above a given time and returns a new
        :class:`.AncestorData` instance.

        Given a set of haplotypes H such that ``lower_time_bound`` <= ``h.time`` <
        ``upper_time_bound``, we let ``max_len = length_multiplier * max(max(h.length)
        for h in H)``. Then, we truncate all haplotypes containing at least one focal
        site where ``h.time >= upper``, ensuring these haplotypes extend no further than
        half of ``max_len`` to the either side of the leftmost and
        rightmost focal sites of the ancestral haplotype. Note that ancestors above
        ``upper_time_bound`` may still be longer than ``max_len`` if the ancestor
        contains greater than 2 focal sites.

        This function should be used when :func:`tsinfer.generate_ancestors` generates
        old ancestors which are very long, as these can significantly slow down matching
        time. Older ancestors should generally be shorter than younger ancestors, so
        truncating the lengths of older ancestors has negligible effect on inference
        accuracy.

        .. note::
            Please ensure that the time values provided to ``lower_time_bound`` and
            ``upper_time_bound`` match the units used in the :class:`.AncestorData`
            file, i.e. if your ancestors do not have site times specified,
            ``upper_time_bound`` should be between 0 and 1.

        :param float lower_time_bound: Defines the lower bound (inclusive) of the half
            open interval where we search for a truncation value.
        :param float upper_time_bound: Defines the upper bound (exclusive) of the half
            open interval where we search for a truncation value. The truncation value
            is the length of the longest haplotype in this interval multiplied by
            ``length_multiplier``. The length of ancestors as old or older than
            ``upper_time_bound`` will be truncated using this value.
        :param float length_multiplier: A multiplier for the length of the longest
            ancestor in the half-open interval between ``lower_time_bound`` (inclusive)
            and ``uppper_time_bound`` (exclusive), i.e.
            if the longest ancestor in the interval is 1 megabase, a
            ``length_multiplier`` of 2 creates a maximum length of 2 megabases.
        :param int buffer_length: The number of changed ancestors to buffer before
            writing to disk.
        :param \\**kwargs: Further arguments passed to the :func:`AncestorData.copy`
            when creating the new :class:`AncestorData` instance which will be returned.

        :return: A new :class:`.AncestorData` object.
        :rtype: AncestorData
        """
        self._check_finalised()
        if self.num_ancestors == 0:
            logger.debug("Passed an AncestorData file with 0 ancestors. Nothing to do")
            return self
        if upper_time_bound < 0 or lower_time_bound < 0:
            raise ValueError("Time bounds cannot be negative")
        if length_multiplier <= 0:
            raise ValueError("Length multiplier cannot be zero or negative")
        if upper_time_bound < lower_time_bound:
            raise ValueError("Upper bound must be >= lower bound")

        position = self.sites_position[:]
        time = self.ancestors_time[:]
        if upper_time_bound > np.max(time) or lower_time_bound > np.max(time):
            raise ValueError("Time bounds cannot be greater than older ancestor")

        anc_in_bound = np.logical_and(
            time >= lower_time_bound,
            time < upper_time_bound,
        )
        if np.sum(anc_in_bound) == 0:
            raise ValueError("No ancestors in time bound")
        max_length = length_multiplier * np.max(self.ancestors_length[:][anc_in_bound])

        truncated = self.copy(**kwargs)

        # Create a buffer of buffer_length ancestors with their indexes
        index_buffer = np.zeros(buffer_length, dtype=np.int32)
        start_buffer = np.zeros(buffer_length, dtype=self.ancestors_start.dtype)
        end_buffer = np.zeros(buffer_length, dtype=self.ancestors_end.dtype)
        time_buffer = np.zeros(buffer_length, dtype=self.ancestors_time.dtype)
        focal_sites_buffer = np.zeros(
            buffer_length, dtype=self.ancestors_focal_sites.dtype
        )
        haplotype_buffer = np.full(
            (self.ancestors_full_haplotype.shape[0], buffer_length, 1),
            tskit.MISSING_DATA,
            dtype=self.ancestors_full_haplotype.dtype,
        )
        buffer_pos = 0

        def flush_buffers(buffer_pos):
            # As we find ancestors that need to be truncated, we write them to the
            # buffers, with index_buffer storing the index of the ancestor in the
            # original AncestorData file. We can use then specify this index array to
            # zarr to just write those changed lines to the new AncestorData file.
            truncated.ancestors_start.set_orthogonal_selection(
                index_buffer[:buffer_pos], start_buffer[:buffer_pos]
            )
            truncated.ancestors_end.set_orthogonal_selection(
                index_buffer[:buffer_pos], end_buffer[:buffer_pos]
            )
            truncated.ancestors_time.set_orthogonal_selection(
                index_buffer[:buffer_pos], time_buffer[:buffer_pos]
            )
            truncated.ancestors_focal_sites.set_orthogonal_selection(
                index_buffer[:buffer_pos], focal_sites_buffer[:buffer_pos]
            )
            truncated.ancestors_full_haplotype.set_orthogonal_selection(
                (slice(None), index_buffer[:buffer_pos]),
                haplotype_buffer[:, :buffer_pos],
            )
            truncated.ancestors_full_haplotype_mask.set_orthogonal_selection(
                (slice(None), index_buffer[:buffer_pos]),
                haplotype_buffer[:, :buffer_pos] == tskit.MISSING_DATA,
            )

        for anc_index, anc in enumerate(self.ancestors()):
            if anc.time >= upper_time_bound and len(anc.focal_sites) > 0:
                if position[anc.end - 1] - position[anc.start] > max_length:
                    left_focal_pos = position[np.min(anc.focal_sites)]
                    right_focal_pos = position[np.max(anc.focal_sites)]
                    insert_pos_start = np.maximum(
                        anc.start,
                        np.searchsorted(position, left_focal_pos - max_length / 2),
                    )
                    insert_pos_end = np.minimum(
                        anc.end,
                        np.searchsorted(position, right_focal_pos + max_length / 2),
                    )
                    original_length = position[anc.end - 1] - position[anc.start]
                    new_length = (
                        position[insert_pos_end - 1] - position[insert_pos_start]
                    )
                    assert new_length <= original_length
                    logger.debug(
                        f"Truncating ancestor {anc.id} at time {anc.time}"
                        "Original length {original_length}. New length {new_length}"
                    )
                    index_buffer[buffer_pos] = anc_index
                    start_buffer[buffer_pos] = insert_pos_start
                    end_buffer[buffer_pos] = insert_pos_end
                    time_buffer[buffer_pos] = anc.time
                    focal_sites_buffer[buffer_pos] = anc.focal_sites
                    haplotype_buffer[
                        insert_pos_start:insert_pos_end, buffer_pos, 0
                    ] = anc.full_haplotype[insert_pos_start:insert_pos_end]
                    buffer_pos += 1
                    if buffer_pos == buffer_length:
                        flush_buffers(buffer_length)
                        buffer_pos = 0
        if buffer_pos > 0:
            flush_buffers(buffer_pos)
        truncated.record_provenance(command="truncate_ancestors")
        truncated.finalise()

        assert self.num_ancestors == truncated.num_ancestors
        assert np.array_equal(time, truncated.ancestors_time)
        assert np.array_equal(position, truncated.sites_position[:])
        return truncated

    ####################################
    # Write mode (building and editing)
    ####################################

    def add_ancestor(self, start, end, time, focal_sites, haplotype):
        """
        Adds an ancestor with the specified haplotype, with ancestral material over the
        interval [start:end], that is associated with the specified timepoint and has new
        mutations at the specified list of focal sites. Ancestors should be added in time
        order, with the oldest first. The id of the added ancestor is returned.
        """
        self._check_build_mode()
        haplotype = tskit.util.safe_np_int_cast(haplotype, dtype=np.int8, copy=True)
        focal_sites = tskit.util.safe_np_int_cast(
            focal_sites, dtype=np.int32, copy=True
        )
        if start < 0:
            raise ValueError("Start must be >= 0")
        if end > self.num_sites:
            raise ValueError("end must be <= num_sites")
        if start >= end:
            raise ValueError("start must be < end")
        if haplotype.shape != (end - start,):
            raise ValueError("haplotypes incorrect shape.")
        if np.any(focal_sites < start) or np.any(focal_sites >= end):
            raise ValueError("focal sites must be between start and end")
        if time <= 0:
            raise ValueError("time must be > 0")
        if self._last_time != 0 and time > self._last_time:
            raise ValueError("older ancestors must be added before younger ones")
        self._last_time = time
        return self.ancestor_writer.add(
            start=start,
            end=end,
            time=time,
            focal_sites=focal_sites,
            haplotype=haplotype,
        )

    def finalise(self):
        if self._mode == self.BUILD_MODE:
            self.ancestor_writer.flush()

        try:
            del self.data["variant_allele"]
        except KeyError:
            pass
        self.create_dataset(
            "variant_allele",
            data=np.tile(["0", "1"], (self.num_sites, 1)),
            shape=(self.num_sites, 2),
            chunks=(self.sites_position.chunks[0], 2),
            dtype="U1",
            compressor=self.sites_position.compressor,
            dimensions=["variants", "alleles"],
        )

        try:
            del self.data["sample_id"]
        except KeyError:
            pass
        self.create_dataset(
            "sample_id",
            data=[f"tsinf_anc_{i}" for i in range(len(self.ancestors_start))],
            shape=(len(self.ancestors_start),),
            chunks=self.ancestors_start.chunks,
            compressor=self.ancestors_start.compressor,
        )
        super().finalise()

    ####################################
    # Read mode
    ####################################

    def ancestor(self, id_):
        """
        Returns the ancestor with the specified ID.

        :rtype: `Ancestor`
        """
        return Ancestor(
            id=id_,
            start=self.ancestors_start[id_],
            end=self.ancestors_end[id_],
            time=self.ancestors_time[id_],
            focal_sites=self.ancestors_focal_sites[id_],
            full_haplotype=self.ancestors_full_haplotype[:, id_, 0],
        )

    def ancestors(self, indexes=None):
        """
        Returns an iterator over all the ancestors. If indexes is provided, it should
        be a sorted list of indexes giving a subset of ancestors to return.
        For efficiency, the indexes should be a numpy integer array.
        """
        start = self.ancestors_start[:]
        end = self.ancestors_end[:]
        time = self.ancestors_time[:]
        focal_sites = self.ancestors_focal_sites[:]
        haplotypes = chunk_iterator(self.ancestors_full_haplotype, indexes, dimension=1)
        if indexes is None:
            indexes = range(len(time))
        for j, h in zip(indexes, haplotypes):
            yield Ancestor(
                id=j,
                start=start[j],
                end=end[j],
                time=time[j],
                focal_sites=focal_sites[j],
                # [0] to remove ploidy dimension
                full_haplotype=h[:, 0],
            )


def load(path):
    """
    Loads a tsinfer :class:`.SampleData` or :class:`.AncestorData` file from
    the specified path. The correct class will be determined by the content
    of the file. If the file is format not recognised a
    :class:`.FileFormatError` will be thrown.

    :param str path: The path of the file we wish to load.
    :return: The corresponding :class:`.SampleData` or :class:`.AncestorData`
        instance opened in read only mode.
    :rtype: :class:`.AncestorData` or :class:`.SampleData`.
    :raises: :class:`.FileFormatError` if the file cannot be read.
    """
    # TODO This is pretty inelegant, but it works. Really we should call the
    # load on the superclass which can dispatch to the registered subclasses
    # for a given format_name.
    tsinfer_file = None
    try:
        logger.debug("Trying SampleData file")
        tsinfer_file = SampleData.load(path)
        logger.debug("Loaded SampleData file")
    except exceptions.FileFormatError as e:
        logger.debug(f"SampleData load failed: {e}")
    try:
        logger.debug("Trying AncestorData file")
        tsinfer_file = AncestorData.load(path)
        logger.debug("Loaded AncestorData file")
    except exceptions.FileFormatError as e:
        logger.debug(f"AncestorData load failed: {e}")
    if tsinfer_file is None:
        raise exceptions.FileFormatError(
            "Unrecognised file format. Try running with -vv and check the log "
            "for more details on what went wrong"
        )
    return tsinfer_file
