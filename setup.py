import os
import platform

import numpy
from setuptools import Extension
from setuptools import setup

IS_WINDOWS = platform.system() == "Windows"

libdir = "lib"
tskroot = os.path.join(libdir, "subprojects", "tskit")
tskdir = os.path.join(tskroot, "tskit")
kasdir = os.path.join(tskroot, "subprojects", "kastore")
includes = [libdir, tskroot, tskdir, kasdir]

tsi_source_files = [
    "ancestor_matcher.c",
    "ancestor_builder.c",
    "object_heap.c",
    "tree_sequence_builder.c",
    "err.c",
    "avl.c",
]
tsk_source_files = ["core.c"]
kas_source_files = ["kastore.c"]

sources = (
    ["_tsinfermodule.c"]
    + [os.path.join(libdir, f) for f in tsi_source_files]
    + [os.path.join(tskdir, f) for f in tsk_source_files]
    + [os.path.join(kasdir, f) for f in kas_source_files]
)

if IS_WINDOWS:
    libraries = ["Advapi32"]
    extra_compile_args = ["/std:c11"]
else:
    libraries = []
    extra_compile_args = ["-std=c11"]

_tsinfer_module = Extension(
    "_tsinfer",
    sources=sources,
    extra_compile_args=extra_compile_args,
    libraries=libraries,
    undef_macros=["NDEBUG"],
    include_dirs=includes + [numpy.get_include()],
)

setup(
    ext_modules=[_tsinfer_module],
)
