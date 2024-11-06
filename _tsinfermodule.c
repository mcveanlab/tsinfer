
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>

#include "lib/tsinfer.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define MODULE_DOC \
"Low-level tsinfer interface."

static PyObject *TsinfLibraryError;
static PyObject *TsinfMatchImpossible;

typedef struct {
    PyObject_HEAD
    ancestor_builder_t *builder;
} AncestorBuilder;

typedef struct {
    PyObject_HEAD
    tree_sequence_builder_t *tree_sequence_builder;
} TreeSequenceBuilder;

typedef struct {
    PyObject_HEAD
    ancestor_matcher_t *ancestor_matcher;
    TreeSequenceBuilder *tree_sequence_builder;
} AncestorMatcher;

static void
handle_library_error(int err)
{
    if (err == TSI_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else if (err == TSI_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA
            || err == TSI_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION) {
        PyErr_Format(TsinfMatchImpossible, "%s", tsi_strerror(err));
    } else {
        PyErr_Format(TsinfLibraryError, "%s", tsi_strerror(err));
    }
}

static int
uint64_PyArray_converter(PyObject *in, PyObject **out)
{
    PyObject *ret = PyArray_FROMANY(in, NPY_UINT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (ret == NULL) {
        return NPY_FAIL;
    }
    *out = ret;
    return NPY_SUCCEED;
}

static int
int8_PyArray_converter(PyObject *in, PyObject **out)
{
    PyObject *ret = PyArray_FROMANY(in, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (ret == NULL) {
        return NPY_FAIL;
    }
    *out = ret;
    return NPY_SUCCEED;
}

/*===================================================================
 * AncestorBuilder
 *===================================================================
 */

static int
AncestorBuilder_check_state(AncestorBuilder *self)
{
    int ret = 0;
    if (self->builder == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorBuilder not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorBuilder_dealloc(AncestorBuilder* self)
{
    if (self->builder != NULL) {
        ancestor_builder_free(self->builder);
        PyMem_Free(self->builder);
        self->builder = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorBuilder_init(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"num_samples", "max_sites", NULL};
    int num_samples, max_sites;
    int flags = 0;

    self->builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                &num_samples, &max_sites)) {
        goto out;
    }
    self->builder = PyMem_Malloc(sizeof(ancestor_builder_t));
    if (self->builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_alloc(self->builder, num_samples, max_sites, flags);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
AncestorBuilder_add_site(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"time", "genotypes", NULL};
    PyObject *ret = NULL;
    double time;
    PyObject *genotypes = NULL;
    PyArrayObject *genotypes_array = NULL;
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dO", kwlist,
            &time, &genotypes)) {
        goto out;
    }
    genotypes_array = (PyArrayObject *) PyArray_FROM_OTF(genotypes, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (genotypes_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(genotypes_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(genotypes_array);
    if (shape[0] != (npy_intp) self->builder->num_samples) {
        PyErr_SetString(PyExc_ValueError, "genotypes array wrong size.");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_add_site(self->builder, time,
            (allele_t *) PyArray_DATA(genotypes_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(genotypes_array);
    return ret;
}

static PyObject *
AncestorBuilder_make_ancestor(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"focal_sites", "ancestor", NULL};
    PyObject *ancestor = NULL;
    PyArrayObject *ancestor_array = NULL;
    PyObject *focal_sites = NULL;
    PyArrayObject *focal_sites_array = NULL;
    size_t num_focal_sites;
    size_t num_sites;
    tsk_id_t start, end;
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO!", kwlist,
            &focal_sites, &PyArray_Type, &ancestor)) {
        goto out;
    }
    num_sites = self->builder->num_sites;
    focal_sites_array = (PyArrayObject *) PyArray_FROM_OTF(focal_sites, NPY_INT32,
            NPY_ARRAY_IN_ARRAY);
    if (focal_sites_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(focal_sites_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(focal_sites_array);
    num_focal_sites = shape[0];
    if (num_focal_sites == 0 || num_focal_sites > num_sites) {
        PyErr_SetString(PyExc_ValueError, "num_focal_sites must > 0 and <= num_sites");
        goto out;
    }
    ancestor_array = (PyArrayObject *) PyArray_FROM_OTF(ancestor, NPY_INT8,
            NPY_ARRAY_INOUT_ARRAY);
    if (ancestor_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(ancestor_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(ancestor_array);
    if (shape[0] != (npy_intp) num_sites) {
        PyErr_SetString(PyExc_ValueError, "input ancestor wrong size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_make_ancestor(self->builder, num_focal_sites,
        (int32_t *) PyArray_DATA(focal_sites_array),
        &start, &end, (int8_t *) PyArray_DATA(ancestor_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("ii", start, end);
out:
    Py_XDECREF(focal_sites_array);
    Py_XDECREF(ancestor_array);
    return ret;
}

static PyObject *
AncestorBuilder_ancestor_descriptors(AncestorBuilder *self)
{
    PyObject *ret = NULL;
    PyObject *descriptors = NULL;
    PyObject *py_descriptor = NULL;
    PyArrayObject *site_array = NULL;
    ancestor_descriptor_t *descriptor;
    size_t j;
    npy_intp dims;
    int err;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    err = ancestor_builder_finalise(self->builder);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    /* ancestor_builder_print_state(self->builder, stdout); */
    descriptors = PyTuple_New(self->builder->num_ancestors);
    if (descriptors == NULL) {
        goto out;
    }
    for (j = 0; j < self->builder->num_ancestors; j++) {
        descriptor = &self->builder->descriptors[j];
        dims = descriptor->num_focal_sites;
        site_array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32);

        if (site_array == NULL) {
            goto out;
        }
        memcpy(PyArray_DATA(site_array), descriptor->focal_sites,
                descriptor->num_focal_sites * sizeof(tsk_id_t));
        py_descriptor = Py_BuildValue("dO", descriptor->time, site_array);
        if (py_descriptor == NULL) {
            Py_DECREF(site_array);
            goto out;
        }
        PyTuple_SET_ITEM(descriptors, j, py_descriptor);
    }
    /* j = 0; */
    /* /1* It's not great that we're breaking encapsulation here and looking */
    /*  * directly in to the builder's data structures. However, it's quite an */
    /*  * awkward set of data to communicate, so it seems OK. *1/ */
    /* for (f = self->builder->num_samples; f > 0; f--) { */
    /*     for (a = self->builder->frequency_map[f].head; a != NULL; a = a->next) { */
    /*         map_elem = (pattern_map_t *) a->item; */
    /*         dims = map_elem->num_sites; */
    /*         site_array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32); */
    /*         if (site_array == NULL) { */
    /*             goto out; */
    /*         } */
    /*         site_array_data = (int32_t *) PyArray_DATA(site_array); */
    /*         /1* The elements are listed backwards, so reverse them *1/ */
    /*         k = map_elem->num_sites - 1; */
    /*         for (s = map_elem->sites; s != NULL; s = s->next) { */
    /*             site_array_data[k] = (int32_t) s->site; */
    /*             k--; */
    /*         } */
    /*         descriptor = Py_BuildValue("kO", (unsigned long) f, site_array); */
    /*         if (descriptor == NULL) { */
    /*             Py_DECREF(site_array); */
    /*             goto out; */
    /*         } */
    /*         PyTuple_SET_ITEM(descriptors, j, descriptor); */
    /*         j++; */
    /*     } */
    /* } */
    ret = descriptors;
    descriptors = NULL;
out:
    Py_XDECREF(descriptors);
    return ret;
}

static PyObject *
AncestorBuilder_get_num_sites(AncestorBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->builder->num_sites);
out:
    return ret;
}

static PyObject *
AncestorBuilder_get_num_ancestors(AncestorBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->builder->num_ancestors);
out:
    return ret;
}

static PyMemberDef AncestorBuilder_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef AncestorBuilder_getsetters[] = {
    {"num_sites", (getter) AncestorBuilder_get_num_sites, NULL, "The number of sites."},
    {"num_ancestors", (getter) AncestorBuilder_get_num_ancestors, NULL, "The number of ancestors."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorBuilder_methods[] = {
    {"add_site", (PyCFunction) AncestorBuilder_add_site,
        METH_VARARGS|METH_KEYWORDS,
        "Adds the specified site to this ancestor builder."},
    {"make_ancestor", (PyCFunction) AncestorBuilder_make_ancestor,
        METH_VARARGS|METH_KEYWORDS,
        "Makes the specified ancestor."},
    {"ancestor_descriptors", (PyCFunction) AncestorBuilder_ancestor_descriptors,
        METH_NOARGS,
        "Returns a list of ancestor (frequency, focal_sites) tuples."},
    {NULL}  /* Sentinel */
};

static PyTypeObject AncestorBuilderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.AncestorBuilder",             /* tp_name */
    sizeof(AncestorBuilder),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)AncestorBuilder_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "AncestorBuilder objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    AncestorBuilder_methods,             /* tp_methods */
    AncestorBuilder_members,             /* tp_members */
    AncestorBuilder_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AncestorBuilder_init,      /* tp_init */
};

/*===================================================================
 * TreeSequenceBuilder
 *===================================================================
 */

static int
TreeSequenceBuilder_check_state(TreeSequenceBuilder *self)
{
    int ret = 0;
    if (self->tree_sequence_builder == NULL) {
        PyErr_SetString(PyExc_SystemError, "TreeSequenceBuilder not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeSequenceBuilder_dealloc(TreeSequenceBuilder* self)
{
    if (self->tree_sequence_builder != NULL) {
        tree_sequence_builder_free(self->tree_sequence_builder);
        PyMem_Free(self->tree_sequence_builder);
        self->tree_sequence_builder = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
TreeSequenceBuilder_init(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"num_alleles", "max_nodes", "max_edges", "ancestral_state",
        NULL};
    PyArrayObject *num_alleles = NULL;
    PyArrayObject *ancestral_state = NULL;
    int8_t *ancestral_state_data = NULL;
    unsigned long max_nodes = 1024;
    unsigned long max_edges = 1024;
    unsigned long num_sites;
    npy_intp *shape;
    int flags = 0;

    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|kkO&", kwlist,
            uint64_PyArray_converter, &num_alleles,
            &max_nodes, &max_edges,
            int8_PyArray_converter, &ancestral_state)) {
        goto out;
    }
    shape = PyArray_DIMS(num_alleles);
    num_sites = shape[0];
    if (ancestral_state != NULL) {
        shape = PyArray_DIMS(ancestral_state);
        if (shape[0] != (npy_intp) num_sites) {
            PyErr_SetString(PyExc_ValueError, "ancestral state array wrong size");
            goto out;
        }
        ancestral_state_data = PyArray_DATA(ancestral_state);
    }
    self->tree_sequence_builder = PyMem_Malloc(sizeof(tree_sequence_builder_t));
    if (self->tree_sequence_builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tree_sequence_builder_alloc(self->tree_sequence_builder,
            num_sites,
            PyArray_DATA(num_alleles),
            ancestral_state_data,
            max_nodes, max_edges, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(num_alleles);
    Py_XDECREF(ancestral_state);
    return ret;
}

static PyObject *
TreeSequenceBuilder_add_node(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;

    static char *kwlist[] = {"time", "flags", NULL};
    double time;
    int flags = 1;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|i", kwlist, &time,
                &flags)) {
        goto out;
    }

    err = tree_sequence_builder_add_node(self->tree_sequence_builder,
            time, (uint32_t) flags);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_add_path(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    int flags = 0;
    PyObject *left = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    int child;
    size_t num_edges;
    npy_intp *shape;
    int compress = 1;
    int extended_checks = 0;

    static char *kwlist[] = {"child", "left", "right", "parent",
        "compress", "extended_checks", NULL};

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kOOO|ii", kwlist,
            &child, &left, &right, &parent, &compress, &extended_checks)) {
        goto out;
    }

    if (compress) {
        flags = TSI_COMPRESS_PATH;
    }
    if (extended_checks) {
        flags |= TSI_EXTENDED_CHECKS;
    }

    /* left */
    left_array = (PyArrayObject *) PyArray_FROM_OTF(left, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (left_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(left_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(left_array);
    num_edges = shape[0];

    /* right */
    right_array = (PyArrayObject *) PyArray_FROM_OTF(right, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (right_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(right_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(right_array);
    if (shape[0] != (npy_intp) num_edges) {
        PyErr_SetString(PyExc_ValueError, "right wrong size");
        goto out;
    }

    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (parent_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != (npy_intp) num_edges) {
        PyErr_SetString(PyExc_ValueError, "parent wrong size");
        goto out;
    }

    /* WARNING!! This isn't fully safe as we're using pointers to data that can
     * be modified in Python. Must make sure that these arrays are not modified
     * by other threads. */
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_add_path(self->tree_sequence_builder,
            child, num_edges,
            (tsk_id_t *) PyArray_DATA(left_array),
            (tsk_id_t *) PyArray_DATA(right_array),
            (tsk_id_t *) PyArray_DATA(parent_array),
            flags);
    Py_END_ALLOW_THREADS

    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(parent_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_add_mutations(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *derived_state = NULL;
    PyArrayObject *derived_state_array = NULL;
    int node;
    size_t num_mutations;
    npy_intp *shape;

    static char *kwlist[] = {"node", "site", "derived_state", NULL};

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kOO", kwlist,
            &node, &site, &derived_state)) {
        goto out;
    }

    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (site_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(site_array);
    num_mutations = shape[0];

    /* derived_state */
    derived_state_array = (PyArrayObject *) PyArray_FROM_OTF(derived_state, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (derived_state_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(derived_state_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(derived_state_array);
    if (shape[0] != (npy_intp) num_mutations) {
        PyErr_SetString(PyExc_ValueError, "derived_state wrong size");
        goto out;
    }

    /* WARNING!! This isn't fully safe as we're using pointers to data that can
     * be modified in Python. Must make sure that these arrays are not modified
     * by other threads. */
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_add_mutations(self->tree_sequence_builder,
            node, num_mutations,
            (tsk_id_t *) PyArray_DATA(site_array),
            (allele_t *) PyArray_DATA(derived_state_array));
    Py_END_ALLOW_THREADS

    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(site_array);
    Py_XDECREF(derived_state_array);
    return ret;
}


static PyObject *
TreeSequenceBuilder_restore_nodes(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"time", "flags", NULL};
    PyObject *time = NULL;
    PyArrayObject *time_array = NULL;
    PyObject *flags = NULL;
    PyArrayObject *flags_array = NULL;
    npy_intp *shape;
    size_t num_nodes;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &time, &flags)) {
        goto out;
    }

    /* time */
    time_array = (PyArrayObject *) PyArray_FROM_OTF(time, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (time_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(time_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(time_array);
    num_nodes = shape[0];

    /* flags */
    flags_array = (PyArrayObject *) PyArray_FROM_OTF(flags, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (flags_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(flags_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(flags_array);
    if (shape[0] != (npy_intp) num_nodes) {
        PyErr_SetString(PyExc_ValueError, "flags array incorrect size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_restore_nodes(self->tree_sequence_builder,
            num_nodes,
            (uint32_t *) PyArray_DATA(flags_array),
            (double *) PyArray_DATA(time_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(time_array);
    Py_XDECREF(flags_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_restore_edges(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"left", "right", "parent", "child", NULL};
    size_t num_edges;
    PyObject *left = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    PyObject *child = NULL;
    PyArrayObject *child_array = NULL;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist,
            &left, &right, &parent, &child)) {
        goto out;
    }

    /* left */
    left_array = (PyArrayObject *) PyArray_FROM_OTF(left, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (left_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(left_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(left_array);
    num_edges = shape[0];

    /* right */
    right_array = (PyArrayObject *) PyArray_FROM_OTF(right, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (right_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(right_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(right_array);
    if (shape[0] != (npy_intp) num_edges) {
        PyErr_SetString(PyExc_ValueError, "right wrong size");
        goto out;
    }

    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (parent_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != (npy_intp) num_edges) {
        PyErr_SetString(PyExc_ValueError, "parent wrong size");
        goto out;
    }

    /* child */
    child_array = (PyArrayObject *) PyArray_FROM_OTF(child, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (child_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(child_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(child_array);
    if (shape[0] != (npy_intp) num_edges) {
        PyErr_SetString(PyExc_ValueError, "child wrong size");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_restore_edges(self->tree_sequence_builder,
            num_edges,
            (tsk_id_t *) PyArray_DATA(left_array),
            (tsk_id_t *) PyArray_DATA(right_array),
            (tsk_id_t *) PyArray_DATA(parent_array),
            (tsk_id_t *) PyArray_DATA(child_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(parent_array);
    Py_XDECREF(child_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_restore_mutations(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"site", "node", "derived_state", "parent", NULL};
    size_t num_mutations;
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *node = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *derived_state = NULL;
    PyArrayObject *derived_state_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist,
            &site, &node, &derived_state, &parent)) {
        goto out;
    }

    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (site_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(site_array);
    num_mutations = shape[0];

    /* node */
    node_array = (PyArrayObject *) PyArray_FROM_OTF(node, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (node_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(node_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(node_array);
    if (shape[0] != (npy_intp) num_mutations) {
        PyErr_SetString(PyExc_ValueError, "node wrong size");
        goto out;
    }

    /* derived_state */
    derived_state_array = (PyArrayObject *) PyArray_FROM_OTF(derived_state, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (derived_state_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(derived_state_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(derived_state_array);
    if (shape[0] != (npy_intp) num_mutations) {
        PyErr_SetString(PyExc_ValueError, "derived_state wrong size");
        goto out;
    }

    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (parent_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != (npy_intp) num_mutations) {
        PyErr_SetString(PyExc_ValueError, "parent wrong size");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_restore_mutations(self->tree_sequence_builder,
            num_mutations,
            (tsk_id_t *) PyArray_DATA(site_array),
            (tsk_id_t *) PyArray_DATA(node_array),
            (allele_t *) PyArray_DATA(derived_state_array));
    /* NOTE: we are ignoring parent here! */
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(site_array);
    Py_XDECREF(node_array);
    Py_XDECREF(derived_state_array);
    Py_XDECREF(parent_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_dump_nodes(TreeSequenceBuilder *self)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *time = NULL;
    PyArrayObject *flags= NULL;
    npy_intp shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    shape = self->tree_sequence_builder->num_nodes;
    time = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_FLOAT64);
    flags = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_UINT32);
    if (time == NULL || flags == NULL) {
       goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_dump_nodes(self->tree_sequence_builder,
        (uint32_t *) PyArray_DATA(flags),
        (double *) PyArray_DATA(time));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("OO", flags, time);
    if (ret == NULL) {
        goto out;
    }
    time = NULL;
    flags = NULL;
out:
    Py_XDECREF(time);
    Py_XDECREF(flags);
    return ret;
}

static PyObject *
TreeSequenceBuilder_dump_edges(TreeSequenceBuilder *self)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    PyArrayObject *child = NULL;
    npy_intp shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    shape = tree_sequence_builder_get_num_edges(self->tree_sequence_builder);
    left = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    right = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    child = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    if (left == NULL || right == NULL || parent == NULL || child == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_dump_edges(self->tree_sequence_builder,
        (tsk_id_t *) PyArray_DATA(left),
        (tsk_id_t *) PyArray_DATA(right),
        (tsk_id_t *) PyArray_DATA(parent),
        (tsk_id_t *) PyArray_DATA(child));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("OOOO", left, right, parent ,child);
    if (ret == NULL) {
        goto out;
    }
    left = NULL;
    right = NULL;
    parent = NULL;
    child = NULL;
out:
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    Py_XDECREF(child);
    return ret;
}

static PyObject *
TreeSequenceBuilder_dump_mutations(TreeSequenceBuilder *self)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *site = NULL;
    PyArrayObject *node = NULL;
    PyArrayObject *derived_state = NULL;
    PyArrayObject *parent = NULL;
    npy_intp shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    shape = self->tree_sequence_builder->num_mutations;
    site = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    node = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    derived_state = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT8);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, &shape, NPY_INT32);
    if (site == NULL || node == NULL || derived_state == NULL || parent == NULL) {
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_dump_mutations(self->tree_sequence_builder,
        (tsk_id_t *) PyArray_DATA(site),
        (tsk_id_t *) PyArray_DATA(node),
        (allele_t *) PyArray_DATA(derived_state),
        (tsk_id_t *) PyArray_DATA(parent));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("OOOO", site, node, derived_state, parent);
    if (ret == NULL) {
        goto out;
    }
    site = NULL;
    node = NULL;
    derived_state = NULL;
    parent = NULL;
out:
    Py_XDECREF(site);
    Py_XDECREF(node);
    Py_XDECREF(derived_state);
    Py_XDECREF(parent);
    return ret;
}

static PyObject *
TreeSequenceBuilder_freeze_indexes(TreeSequenceBuilder *self)
{
    int err;
    PyObject *ret = NULL;

    err = tree_sequence_builder_freeze_indexes(self->tree_sequence_builder);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_edges(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            tree_sequence_builder_get_num_edges(self->tree_sequence_builder));
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_nodes(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            tree_sequence_builder_get_num_nodes(self->tree_sequence_builder));
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_match_nodes(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k",
        (unsigned long) self->tree_sequence_builder->num_match_nodes);
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_sites(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->tree_sequence_builder->num_sites);
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_mutations(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            tree_sequence_builder_get_num_mutations(self->tree_sequence_builder));
out:
    return ret;
}
static PyMemberDef TreeSequenceBuilder_members[] = {
    {NULL}  /* Sentinel */

};

static PyGetSetDef TreeSequenceBuilder_getsetters[] = {
    {"num_nodes", (getter) TreeSequenceBuilder_get_num_nodes, NULL,
        "The number of nodes."},
    {"num_match_nodes", (getter) TreeSequenceBuilder_get_num_match_nodes, NULL,
        "The number of nodes available to match against."},
    {"num_edges", (getter) TreeSequenceBuilder_get_num_edges, NULL,
        "The number of edges."},
    {"num_sites", (getter) TreeSequenceBuilder_get_num_sites, NULL,
        "The total number of sites."},
    {"num_mutations", (getter) TreeSequenceBuilder_get_num_mutations, NULL,
        "The total number of mutations."},
    {NULL}  /* Sentinel */
};

static PyMethodDef TreeSequenceBuilder_methods[] = {
    {"add_node", (PyCFunction) TreeSequenceBuilder_add_node,
        METH_VARARGS|METH_KEYWORDS,
        "Adds a new node to the tree sequence builder and returns its ID."},
    {"add_path", (PyCFunction) TreeSequenceBuilder_add_path,
        METH_VARARGS|METH_KEYWORDS,
        "Updates the builder with the specified copy results for a given child."},
    {"add_mutations", (PyCFunction) TreeSequenceBuilder_add_mutations,
        METH_VARARGS|METH_KEYWORDS,
        "Updates the builder with mutations for a given node."},
    {"restore_nodes", (PyCFunction) TreeSequenceBuilder_restore_nodes,
        METH_VARARGS|METH_KEYWORDS,
        "Restores the nodes in this tree sequence builder."},
    {"restore_edges", (PyCFunction) TreeSequenceBuilder_restore_edges,
        METH_VARARGS|METH_KEYWORDS,
        "Restores the edges in this tree sequence builder."},
    {"restore_mutations", (PyCFunction) TreeSequenceBuilder_restore_mutations,
        METH_VARARGS|METH_KEYWORDS,
        "Restores the mutations in this tree sequence builder."},
    {"dump_nodes", (PyCFunction) TreeSequenceBuilder_dump_nodes, METH_NOARGS,
        "Dumps node data into numpy arrays."},
    {"dump_edges", (PyCFunction) TreeSequenceBuilder_dump_edges, METH_NOARGS,
        "Dumps edgeset data into numpy arrays."},
    {"dump_mutations", (PyCFunction) TreeSequenceBuilder_dump_mutations, METH_NOARGS,
        "Dumps mutation data into numpy arrays."},
    {"freeze_indexes", (PyCFunction) TreeSequenceBuilder_freeze_indexes, METH_NOARGS,
        "Freezes the indexes used for ancestor matching."},
    {NULL}  /* Sentinel */
};

static PyTypeObject TreeSequenceBuilderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.TreeSequenceBuilder",             /* tp_name */
    sizeof(TreeSequenceBuilder),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)TreeSequenceBuilder_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "TreeSequenceBuilder objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    TreeSequenceBuilder_methods,             /* tp_methods */
    TreeSequenceBuilder_members,             /* tp_members */
    TreeSequenceBuilder_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)TreeSequenceBuilder_init,      /* tp_init */
};


/*===================================================================
 * AncestorMatcher
 *===================================================================
 */

static int
AncestorMatcher_check_state(AncestorMatcher *self)
{
    int ret = 0;
    if (self->ancestor_matcher == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorMatcher not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorMatcher_dealloc(AncestorMatcher* self)
{
    if (self->ancestor_matcher != NULL) {
        ancestor_matcher_free(self->ancestor_matcher);
        PyMem_Free(self->ancestor_matcher);
        self->ancestor_matcher = NULL;
    }
    Py_XDECREF(self->tree_sequence_builder);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorMatcher_init(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    int extended_checks = 0;
    static char *kwlist[] = {"tree_sequence_builder", "recombination",
        "mismatch", "likelihood_threshold", "extended_checks", NULL};
    TreeSequenceBuilder *tree_sequence_builder = NULL;
    PyObject *recombination = NULL;
    PyObject *mismatch = NULL;
    PyArrayObject *recombination_array = NULL;
    PyArrayObject *mismatch_array = NULL;
    npy_intp *shape;
    double likelihood_threshold = DBL_MIN;
    int flags = 0;

    self->ancestor_matcher = NULL;
    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|di", kwlist,
                &TreeSequenceBuilderType, &tree_sequence_builder,
                &recombination, &mismatch, &likelihood_threshold,
                &extended_checks)) {
        goto out;
    }
    self->tree_sequence_builder = tree_sequence_builder;
    Py_INCREF(self->tree_sequence_builder);
    if (TreeSequenceBuilder_check_state(self->tree_sequence_builder) != 0) {
        goto out;
    }

    recombination_array = (PyArrayObject *) PyArray_FromAny(recombination,
            PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
            NPY_ARRAY_IN_ARRAY, NULL);
    if (recombination_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(recombination_array);
    if (shape[0] != (npy_intp) tree_sequence_builder->tree_sequence_builder->num_sites) {
        PyErr_SetString(PyExc_ValueError,
                "Size of recombination array must be num_sites");
        goto out;
    }
    mismatch_array = (PyArrayObject *) PyArray_FromAny(mismatch,
            PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
            NPY_ARRAY_IN_ARRAY, NULL);
    if (mismatch_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(mismatch_array);
    if (shape[0] != (npy_intp) tree_sequence_builder->tree_sequence_builder->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Size of mismatch array must be num_sites");
        goto out;
    }

    self->ancestor_matcher = PyMem_Malloc(sizeof(ancestor_matcher_t));
    if (self->ancestor_matcher == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    if (extended_checks) {
        flags = TSI_EXTENDED_CHECKS;
    }
    err = ancestor_matcher_alloc(self->ancestor_matcher,
            self->tree_sequence_builder->tree_sequence_builder,
            PyArray_DATA(recombination_array),
            PyArray_DATA(mismatch_array),
            likelihood_threshold, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(recombination_array);
    Py_XDECREF(mismatch_array);
    return ret;
}

static PyObject *
AncestorMatcher_find_path(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"haplotype", "start", "end", "match", NULL};
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyObject *match = NULL;
    PyArrayObject *match_array = NULL;
    npy_intp *shape;
    size_t num_edges;
    int start, end;
    tsk_id_t *ret_left, *ret_right;
    tsk_id_t *ret_parent;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    npy_intp dims[1];

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OiiO!", kwlist,
                &haplotype, &start, &end, &PyArray_Type, &match)) {
        goto out;
    }
    haplotype_array = (PyArrayObject *) PyArray_FROM_OTF(haplotype, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(haplotype_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != (npy_intp) self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }

    match_array = (PyArrayObject *) PyArray_FROM_OTF(match, NPY_INT8,
            NPY_ARRAY_INOUT_ARRAY);
    if (match_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(match_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(match_array);
    if (shape[0] != (npy_intp) self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "input match wrong size");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = ancestor_matcher_find_path(self->ancestor_matcher,
            (tsk_id_t) start, (tsk_id_t) end, (allele_t *) PyArray_DATA(haplotype_array),
            (allele_t *) PyArray_DATA(match_array),
            &num_edges, &ret_left, &ret_right, &ret_parent);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = num_edges;
    left = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT32);
    right = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT32);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (left == NULL || right == NULL || parent == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(left), ret_left, num_edges * sizeof(*ret_left));
    memcpy(PyArray_DATA(right), ret_right, num_edges * sizeof(*ret_right));
    memcpy(PyArray_DATA(parent), ret_parent, num_edges * sizeof(*ret_parent));
    ret = Py_BuildValue("(OOO)", left, right, parent);
    if (ret == NULL) {
        goto out;
    }
    left = NULL;
    right = NULL;
    parent = NULL;
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(match_array);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    return ret;
}

static PyObject *
AncestorMatcher_get_traceback(AncestorMatcher *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned long site;
    node_state_list_t *list;
    PyObject *dict = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;
    int j;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "k", &site)) {
        goto out;
    }
    if (site >= self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "site out of range");
        goto out;
    }
    dict = PyDict_New();
    if (dict == NULL) {
        goto out;
    }
    list = &self->ancestor_matcher->traceback[site];
    for (j = 0; j < list->size; j++) {
        key = Py_BuildValue("k", (unsigned long) list->node[j]);
        value = Py_BuildValue("i", (int) list->recombination_required[j]);
        if (key == NULL || value == NULL) {
            goto out;
        }
        if (PyDict_SetItem(dict, key, value) != 0) {
            goto out;
        }
        Py_DECREF(key);
        key = NULL;
        Py_DECREF(value);
        value = NULL;
    }
    ret = dict;
    dict = NULL;
out:
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(dict);
    return ret;
}

static PyObject *
AncestorMatcher_get_mean_traceback_size(AncestorMatcher *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("d", ancestor_matcher_get_mean_traceback_size(
                self->ancestor_matcher));
out:
    return ret;
}

static PyObject *
AncestorMatcher_get_total_memory(AncestorMatcher *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            ancestor_matcher_get_total_memory(self->ancestor_matcher));
out:
    return ret;
}


static PyMemberDef AncestorMatcher_members[] = {
    {NULL}  /* Sentinel */

};

static PyGetSetDef AncestorMatcher_getsetters[] = {
    {"mean_traceback_size", (getter) AncestorMatcher_get_mean_traceback_size,
        NULL, "The mean size of the traceback per site."},
    {"total_memory", (getter) AncestorMatcher_get_total_memory,
        NULL, "The total amount of memory used by this matcher."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorMatcher_methods[] = {
    {"find_path", (PyCFunction) AncestorMatcher_find_path,
        METH_VARARGS|METH_KEYWORDS,
        "Returns a best match path for the specified haplotype through the ancestors."},
    {"get_traceback", (PyCFunction) AncestorMatcher_get_traceback,
        METH_VARARGS, "Returns the traceback likelihood dictionary at the specified site."},
    {NULL}  /* Sentinel */
};

static PyTypeObject AncestorMatcherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.AncestorMatcher",             /* tp_name */
    sizeof(AncestorMatcher),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)AncestorMatcher_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "AncestorMatcher objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    AncestorMatcher_methods,             /* tp_methods */
    AncestorMatcher_members,             /* tp_members */
    AncestorMatcher_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AncestorMatcher_init,      /* tp_init */
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyMethodDef tsinfer_methods[] = {
    {NULL}        /* Sentinel */
};

/* Initialisation code supports Python 2.x and 3.x. The framework uses the
 * recommended structure from http://docs.python.org/howto/cporting.html.
 * I've ignored the point about storing state in globals, as the examples
 * from the Python documentation still use this idiom.
 */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef tsinfermodule = {
    PyModuleDef_HEAD_INIT,
    "_tsinfer",   /* name of module */
    MODULE_DOC, /* module documentation, may be NULL */
    -1,
    tsinfer_methods,
    NULL, NULL, NULL, NULL
};

#define INITERROR return NULL

PyObject *
PyInit__tsinfer(void)

#else
#define INITERROR return

void
init_tsinfer(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&tsinfermodule);
#else
    PyObject *module = Py_InitModule3("_tsinfer", tsinfer_methods, MODULE_DOC);
#endif
    if (module == NULL) {
        INITERROR;
    }
    /* Initialise numpy */
    import_array();

    /* AncestorBuilder type */
    AncestorBuilderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorBuilderType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorBuilderType);
    PyModule_AddObject(module, "AncestorBuilder", (PyObject *) &AncestorBuilderType);
    /* AncestorMatcher type */
    AncestorMatcherType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorMatcherType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorMatcherType);
    PyModule_AddObject(module, "AncestorMatcher", (PyObject *) &AncestorMatcherType);
    /* TreeSequenceBuilder type */
    TreeSequenceBuilderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&TreeSequenceBuilderType) < 0) {
        INITERROR;
    }
    Py_INCREF(&TreeSequenceBuilderType);
    PyModule_AddObject(module, "TreeSequenceBuilder", (PyObject *) &TreeSequenceBuilderType);

    TsinfLibraryError = PyErr_NewException("_tsinfer.LibraryError", NULL, NULL);
    Py_INCREF(TsinfLibraryError);
    PyModule_AddObject(module, "LibraryError", TsinfLibraryError);

    TsinfMatchImpossible= PyErr_NewException("_tsinfer.MatchImpossible", NULL, NULL);
    Py_INCREF(TsinfMatchImpossible);
    PyModule_AddObject(module, "MatchImpossible", TsinfMatchImpossible);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
