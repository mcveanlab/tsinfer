/*
** Copyright (C) 2018-2020 University of Oxford
**
** This file is part of tsinfer.
**
** tsinfer is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** tsinfer is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "tskit.h"
#include "err.h"
#include "object_heap.h"
#include "avl.h"

/* TODO remove this when we update tskit version. */
#define TSK_MISSING_DATA (-1)

/* NULL_LIKELIHOOD represents a compressed path and NONZERO_ROOT_LIKELIHOOD
 * marks a node that is not in the current tree. */
#define NULL_LIKELIHOOD (-1)
#define NONZERO_ROOT_LIKELIHOOD (-2)

#define NULL_NODE (-1)
#define CACHE_UNSET (-1)

#define TSI_COMPRESS_PATH 1
#define TSI_EXTENDED_CHECKS 2

#define TSI_NODE_IS_PC_ANCESTOR ((tsk_flags_t)(1u << 16))

typedef int8_t allele_t;

typedef struct {
    tsk_id_t left;
    tsk_id_t right;
    tsk_id_t parent;
    tsk_id_t child;
} edge_t;

typedef struct _indexed_edge_t {
    edge_t edge;
    double time;
    struct _indexed_edge_t *next;
} indexed_edge_t;

typedef struct _node_segment_list_node_t {
    tsk_id_t start;
    tsk_id_t end;
    struct _node_segment_list_node_t *next;
} node_segment_list_node_t;

typedef struct {
    double time;
    allele_t *genotypes;
} site_t;

typedef struct {
    tsk_id_t *start;
    tsk_id_t *end;
    size_t num_segments;
    double position;
} site_state_t;

typedef struct _site_list_t {
    tsk_id_t site;
    struct _site_list_t *next;
} site_list_t;

typedef struct {
    allele_t *genotypes;
    size_t num_samples;
    size_t num_sites;
    site_list_t *sites;
} pattern_map_t;

typedef struct {
    double time;
    size_t num_focal_sites;
    tsk_id_t *focal_sites;
} ancestor_descriptor_t;

/* Maps all ancestors with a specific time to their genotype patterns  */
typedef struct {
    double time;
    avl_tree_t pattern_map;
} time_map_t;

typedef struct {
    size_t num_sites;
    size_t max_sites;
    size_t num_samples;
    size_t num_ancestors;
    int flags;
    site_t *sites;
    avl_tree_t time_map;
    tsi_blkalloc_t allocator;
    ancestor_descriptor_t *descriptors;
} ancestor_builder_t;

typedef struct _mutation_list_node_t {
    tsk_id_t node;
    allele_t derived_state;
    struct _mutation_list_node_t *next;
} mutation_list_node_t;

typedef struct {
    int32_t size;
    tsk_id_t *node;
    int8_t *recombination_required;
} node_state_list_t;

typedef struct {
    int flags;
    size_t num_sites;
    struct {
        mutation_list_node_t **mutations;
        tsk_size_t *num_alleles;
        allele_t *ancestral_state;
    } sites;
    /* TODO add nodes struct */
    double *time;
    uint32_t *node_flags;
    indexed_edge_t **path;
    size_t nodes_chunk_size;
    size_t edges_chunk_size;
    size_t max_nodes;
    size_t num_nodes;
    size_t num_match_nodes;
    size_t num_mutations;
    tsi_blkalloc_t tsi_blkalloc;
    object_heap_t avl_node_heap;
    object_heap_t edge_heap;
    /* Dynamic edge indexes used for tree generation and path compression. The
     * AVL trees are used to keep the indexes updated without needing to perform
     * repeated sorting */
    avl_tree_t left_index;
    avl_tree_t right_index;
    avl_tree_t path_index;
    /* The static tree generation indexes. We populate these at the end of each
     * epoch using the order defined by the AVL trees. */
    edge_t *left_index_edges;
    edge_t *right_index_edges;
    size_t num_edges; /* the number of edges in the frozen indexes */
} tree_sequence_builder_t;

typedef struct {
    int flags;
    tree_sequence_builder_t *tree_sequence_builder;
    size_t num_nodes;
    size_t num_sites;
    size_t max_nodes;
    /* Input LS model rates */
    double likelihood_threshold;
    double *recombination_rate;
    double *mismatch_rate;
    /* The quintuply linked tree */
    tsk_id_t *parent;
    tsk_id_t *left_child;
    tsk_id_t *right_child;
    tsk_id_t *left_sib;
    tsk_id_t *right_sib;
    double *likelihood;
    double *likelihood_cache;
    allele_t *allelic_state;
    int num_likelihood_nodes;
    /* At each site, record a node with the maximum likelihood. */
    tsk_id_t *max_likelihood_node;
    /* Used during traceback to map nodes where recombination is required. */
    int8_t *recombination_required;
    tsk_id_t *likelihood_nodes_tmp;
    tsk_id_t *likelihood_nodes;
    node_state_list_t *traceback;
    tsi_blkalloc_t traceback_allocator;
    size_t total_traceback_size;
    struct {
        tsk_id_t *left;
        tsk_id_t *right;
        tsk_id_t *parent;
        size_t size;
        size_t max_size;
    } output;
} ancestor_matcher_t;

int ancestor_builder_alloc(
    ancestor_builder_t *self, size_t num_samples, size_t num_sites, int flags);
int ancestor_builder_free(ancestor_builder_t *self);
int ancestor_builder_print_state(ancestor_builder_t *self, FILE *out);
int ancestor_builder_add_site(
    ancestor_builder_t *self, double time, allele_t *genotypes);
int ancestor_builder_make_ancestor(ancestor_builder_t *self, size_t num_focal_sites,
    tsk_id_t *focal_sites, tsk_id_t *start, tsk_id_t *end, allele_t *haplotype);
int ancestor_builder_finalise(ancestor_builder_t *self);

int ancestor_matcher_alloc(ancestor_matcher_t *self,
    tree_sequence_builder_t *tree_sequence_builder, double *recombination_rate,
    double *mismatch_rate, double likelihood_threshold, int flags);
int ancestor_matcher_free(ancestor_matcher_t *self);
int ancestor_matcher_find_path(ancestor_matcher_t *self, tsk_id_t start, tsk_id_t end,
    allele_t *haplotype, allele_t *matched_haplotype, size_t *num_output_edges,
    tsk_id_t **left_output, tsk_id_t **right_output, tsk_id_t **parent_output);
int ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out);
double ancestor_matcher_get_mean_traceback_size(ancestor_matcher_t *self);
size_t ancestor_matcher_get_total_memory(ancestor_matcher_t *self);

int tree_sequence_builder_alloc(tree_sequence_builder_t *self, size_t num_sites,
    tsk_size_t *num_alleles, allele_t *ancestral_state, size_t nodes_chunk_size,
    size_t edges_chunk_size, int flags);
int tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out);
int tree_sequence_builder_free(tree_sequence_builder_t *self);
int tree_sequence_builder_add_node(
    tree_sequence_builder_t *self, double time, uint32_t flags);
int tree_sequence_builder_add_path(tree_sequence_builder_t *self, tsk_id_t child,
    size_t num_edges, tsk_id_t *left, tsk_id_t *right, tsk_id_t *parent, int flags);
int tree_sequence_builder_add_mutation(
    tree_sequence_builder_t *self, tsk_id_t node, tsk_id_t site, allele_t derived_state);
int tree_sequence_builder_add_mutations(tree_sequence_builder_t *self, tsk_id_t node,
    size_t num_mutations, tsk_id_t *site, allele_t *derived_state);
int tree_sequence_builder_freeze_indexes(tree_sequence_builder_t *self);

size_t tree_sequence_builder_get_num_nodes(tree_sequence_builder_t *self);
size_t tree_sequence_builder_get_num_edges(tree_sequence_builder_t *self);
size_t tree_sequence_builder_get_num_mutations(tree_sequence_builder_t *self);

/* Restore the state of a previous tree sequence builder. */
int tree_sequence_builder_restore_nodes(
    tree_sequence_builder_t *self, size_t num_nodes, uint32_t *flags, double *time);
int tree_sequence_builder_restore_edges(tree_sequence_builder_t *self, size_t num_edges,
    tsk_id_t *left, tsk_id_t *right, tsk_id_t *parent, tsk_id_t *child);
int tree_sequence_builder_restore_mutations(tree_sequence_builder_t *self,
    size_t num_mutations, tsk_id_t *site, tsk_id_t *node, allele_t *derived_state);

/* Dump the state */
int tree_sequence_builder_dump_nodes(
    tree_sequence_builder_t *self, uint32_t *flags, double *time);
int tree_sequence_builder_dump_edges(tree_sequence_builder_t *self, tsk_id_t *left,
    tsk_id_t *right, tsk_id_t *parent, tsk_id_t *children);
int tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self, tsk_id_t *site,
    tsk_id_t *node, allele_t *derived_state, tsk_id_t *parent);

#define tsi_safe_free(pointer)                                                          \
    do {                                                                                \
        if (pointer != NULL) {                                                          \
            free(pointer);                                                              \
        }                                                                               \
    } while (0)
