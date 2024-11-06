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
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "tsinfer.h"
#include "err.h"

#include "avl.h"

/* Time increment between path compression ancestors and their parents.
 * Power-of-two value chosen here so that we can manipulate time values
 * reasonably losslessly. This is about 2.3e-10. This should be enough
 * that even with very large samples (and therefore small frequency
 * values) we have enough space to add many synthetic ancestors. It's a
 * hack though, and we should have a better approach. */
#define PC_ANCESTOR_INCREMENT (1.0 / (1LL << 32))

static int
cmp_edge_left_increasing_time(const void *a, const void *b)
{
    const indexed_edge_t *ca = (const indexed_edge_t *) a;
    const indexed_edge_t *cb = (const indexed_edge_t *) b;
    int ret = (ca->edge.left > cb->edge.left) - (ca->edge.left < cb->edge.left);
    if (ret == 0) {
        ret = (ca->time > cb->time) - (ca->time < cb->time);
        if (ret == 0) {
            ret = (ca->edge.child > cb->edge.child) - (ca->edge.child < cb->edge.child);
        }
    }
    return ret;
}

static int
cmp_edge_right_decreasing_time(const void *a, const void *b)
{
    const indexed_edge_t *ca = (const indexed_edge_t *) a;
    const indexed_edge_t *cb = (const indexed_edge_t *) b;
    int ret = (ca->edge.right > cb->edge.right) - (ca->edge.right < cb->edge.right);
    if (ret == 0) {
        ret = (ca->time < cb->time) - (ca->time > cb->time);
        if (ret == 0) {
            ret = (ca->edge.child > cb->edge.child) - (ca->edge.child < cb->edge.child);
        }
    }
    return ret;
}

static int
cmp_edge_path(const void *a, const void *b)
{
    const indexed_edge_t *ca = (const indexed_edge_t *) a;
    const indexed_edge_t *cb = (const indexed_edge_t *) b;
    int ret = (ca->edge.left > cb->edge.left) - (ca->edge.left < cb->edge.left);
    if (ret == 0) {
        ret = (ca->edge.right > cb->edge.right) - (ca->edge.right < cb->edge.right);
        if (ret == 0) {
            ret = (ca->edge.parent > cb->edge.parent)
                  - (ca->edge.parent < cb->edge.parent);
            if (ret == 0) {
                ret = (ca->edge.child > cb->edge.child)
                      - (ca->edge.child < cb->edge.child);
            }
        }
    }
    return ret;
}

static void
print_edge_path(indexed_edge_t *head, FILE *out)
{
    indexed_edge_t *e;

    for (e = head; e != NULL; e = e->next) {
        fprintf(out, "(%d, %d, %d, %d)", e->edge.left, e->edge.right, e->edge.parent,
            e->edge.child);
        if (e->next != NULL) {
            fprintf(out, "->");
        }
    }
    fprintf(out, "\n");
}

static void
tree_sequence_builder_check_index_integrity(tree_sequence_builder_t *self)
{
    avl_node_t *avl_node;
    indexed_edge_t *edge;
    size_t j;

    for (j = 0; j < self->num_nodes; j++) {
        for (edge = self->path[j]; edge != NULL; edge = edge->next) {
            avl_node = avl_search(&self->left_index, edge);
            assert(avl_node != NULL);
            assert(avl_node->item == (void *) edge);

            avl_node = avl_search(&self->right_index, edge);
            assert(avl_node != NULL);
            assert(avl_node->item == (void *) edge);

            avl_node = avl_search(&self->path_index, edge);
            assert(avl_node != NULL);
            assert(avl_node->item == (void *) edge);
        }
    }
}

static void
tree_sequence_builder_check_state(tree_sequence_builder_t *self)
{
    tsk_id_t child;
    indexed_edge_t *e;
    size_t total_edges = 0;
    size_t num_match_nodes = 1;

    for (child = 0; child < (tsk_id_t) self->num_nodes; child++) {
        num_match_nodes += self->path[child] != NULL;
        for (e = self->path[child]; e != NULL; e = e->next) {
            total_edges++;
            assert(e->edge.child == child);
            if (e->next != NULL) {
                assert(e->next->edge.left == e->edge.right);
            }
        }
    }
    assert(num_match_nodes == self->num_match_nodes);
    assert(avl_count(&self->left_index) == total_edges);
    assert(avl_count(&self->right_index) == total_edges);
    assert(avl_count(&self->path_index) == total_edges);
    assert(total_edges == object_heap_get_num_allocated(&self->edge_heap));
    assert(3 * total_edges == object_heap_get_num_allocated(&self->avl_node_heap));
    tree_sequence_builder_check_index_integrity(self);
}

int
tree_sequence_builder_print_state(tree_sequence_builder_t *self, FILE *out)
{
    size_t j;
    mutation_list_node_t *u;
    avl_node_t *a;
    edge_t *edge;

    fprintf(out, "Tree sequence builder state\n");
    fprintf(out, "flags = %d\n", (int) self->flags);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_nodes = %d\n", (int) self->num_nodes);
    fprintf(out, "num_edges = %d\n", (int) tree_sequence_builder_get_num_edges(self));
    fprintf(out, "num_match_nodes  = %d\n", (int) self->num_match_nodes);
    fprintf(out, "num_frozen_edges = %d\n", (int) self->num_edges);
    fprintf(out, "max_nodes = %d\n", (int) self->max_nodes);
    fprintf(out, "nodes_chunk_size = %d\n", (int) self->nodes_chunk_size);
    fprintf(out, "edges_chunk_size = %d\n", (int) self->edges_chunk_size);

    fprintf(out, "nodes = \n");
    fprintf(out, "id\tflags\ttime\tpath\n");
    for (j = 0; j < self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%f ", (int) j, self->node_flags[j], self->time[j]);
        print_edge_path(self->path[j], out);
    }

    fprintf(out, "mutations = \n");
    fprintf(out, "site\t(node, derived_state),...\n");
    for (j = 0; j < self->num_sites; j++) {
        if (self->sites.mutations[j] != NULL) {
            fprintf(out, "%d\t", (int) j);
            for (u = self->sites.mutations[j]; u != NULL; u = u->next) {
                fprintf(out, "(%d, %d) ", u->node, u->derived_state);
            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "path index \n");
    for (a = self->path_index.head; a != NULL; a = a->next) {
        edge = (edge_t *) a->item;
        fprintf(
            out, "%d\t%d\t%d\t%d\n", edge->left, edge->right, edge->parent, edge->child);
    }

    fprintf(out, "tsi_blkalloc = \n");
    tsi_blkalloc_print_state(&self->tsi_blkalloc, out);
    fprintf(out, "avl_node_heap = \n");
    object_heap_print_state(&self->avl_node_heap, out);
    fprintf(out, "edge_heap = \n");
    object_heap_print_state(&self->edge_heap, out);

    tree_sequence_builder_check_state(self);
    return 0;
}

int
tree_sequence_builder_alloc(tree_sequence_builder_t *self, size_t num_sites,
    tsk_size_t *num_alleles, allele_t *ancestral_state, size_t nodes_chunk_size,
    size_t edges_chunk_size, int flags)
{
    int ret = 0;
    size_t j;

    memset(self, 0, sizeof(tree_sequence_builder_t));

    assert(num_sites < INT32_MAX);

    self->num_sites = num_sites;
    self->nodes_chunk_size = nodes_chunk_size;
    self->edges_chunk_size = edges_chunk_size;
    self->flags = flags;
    self->num_nodes = 0;
    self->num_match_nodes = 1;
    self->max_nodes = nodes_chunk_size;

    self->time = malloc(self->max_nodes * sizeof(*self->time));
    self->node_flags = malloc(self->max_nodes * sizeof(*self->node_flags));
    self->path = calloc(self->max_nodes, sizeof(*self->path));
    self->sites.mutations = calloc(self->num_sites, sizeof(*self->sites.mutations));
    self->sites.num_alleles = calloc(self->num_sites, sizeof(*self->sites.num_alleles));
    self->sites.ancestral_state
        = calloc(self->num_sites, sizeof(*self->sites.ancestral_state));
    if (self->time == NULL || self->node_flags == NULL || self->path == NULL
        || self->sites.mutations == NULL || self->sites.num_alleles == NULL
        || self->sites.ancestral_state == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    ret = object_heap_init(
        &self->avl_node_heap, sizeof(avl_node_t), self->edges_chunk_size, NULL);
    if (ret != 0) {
        goto out;
    }
    ret = object_heap_init(
        &self->edge_heap, sizeof(indexed_edge_t), self->edges_chunk_size, NULL);
    if (ret != 0) {
        goto out;
    }
    ret = tsi_blkalloc_init(&self->tsi_blkalloc,
        TSK_MAX(8192, num_sites * sizeof(mutation_list_node_t) / 4));
    if (ret != 0) {
        goto out;
    }
    avl_init_tree(&self->left_index, cmp_edge_left_increasing_time, NULL);
    avl_init_tree(&self->right_index, cmp_edge_right_decreasing_time, NULL);
    avl_init_tree(&self->path_index, cmp_edge_path, NULL);

    for (j = 0; j < num_sites; j++) {
        if (num_alleles == NULL) {
            self->sites.num_alleles[j] = 2;
        } else {
            if (num_alleles[j] < 2 || num_alleles[j] > INT8_MAX) {
                ret = TSI_ERR_BAD_NUM_ALLELES;
                goto out;
            }
            self->sites.num_alleles[j] = num_alleles[j];
        }
        if (ancestral_state != NULL) {
            if (ancestral_state[j] < 0) {
                ret = TSI_ERR_BAD_ANCESTRAL_STATE;
                goto out;
            }
            self->sites.ancestral_state[j] = ancestral_state[j];
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_free(tree_sequence_builder_t *self)
{
    tsi_safe_free(self->time);
    tsi_safe_free(self->path);
    tsi_safe_free(self->node_flags);
    tsi_safe_free(self->sites.mutations);
    tsi_safe_free(self->sites.num_alleles);
    tsi_safe_free(self->sites.ancestral_state);
    tsi_safe_free(self->left_index_edges);
    tsi_safe_free(self->right_index_edges);
    tsi_blkalloc_free(&self->tsi_blkalloc);
    object_heap_free(&self->avl_node_heap);
    object_heap_free(&self->edge_heap);
    return 0;
}

static inline avl_node_t *WARN_UNUSED
tree_sequence_builder_alloc_avl_node(tree_sequence_builder_t *self, indexed_edge_t *e)
{
    avl_node_t *ret = NULL;

    if (object_heap_empty(&self->avl_node_heap)) {
        if (object_heap_expand(&self->avl_node_heap) != 0) {
            goto out;
        }
    }
    ret = (avl_node_t *) object_heap_alloc_object(&self->avl_node_heap);
    avl_init_node(ret, e);
out:
    return ret;
}

static inline void
tree_sequence_builder_free_avl_node(tree_sequence_builder_t *self, avl_node_t *node)
{
    object_heap_free_object(&self->avl_node_heap, node);
}

static inline indexed_edge_t *WARN_UNUSED
tree_sequence_builder_alloc_edge(tree_sequence_builder_t *self, tsk_id_t left,
    tsk_id_t right, tsk_id_t parent, tsk_id_t child, indexed_edge_t *next)
{
    indexed_edge_t *ret = NULL;

    if (object_heap_empty(&self->edge_heap)) {
        if (object_heap_expand(&self->edge_heap) != 0) {
            goto out;
        }
    }
    assert(parent < (tsk_id_t) self->num_nodes);
    assert(child < (tsk_id_t) self->num_nodes);
    assert(self->time[parent] > self->time[child]);
    ret = (indexed_edge_t *) object_heap_alloc_object(&self->edge_heap);
    ret->edge.left = left;
    ret->edge.right = right;
    ret->edge.parent = parent;
    ret->edge.child = child;
    ret->time = self->time[child];
    ret->next = next;
out:
    return ret;
}

static inline void
tree_sequence_builder_free_edge(tree_sequence_builder_t *self, indexed_edge_t *edge)
{
    object_heap_free_object(&self->edge_heap, edge);
}

static int WARN_UNUSED
tree_sequence_builder_expand_nodes(tree_sequence_builder_t *self)
{
    int ret = 0;
    void *tmp;

    self->max_nodes += self->nodes_chunk_size;
    tmp = realloc(self->time, self->max_nodes * sizeof(double));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->time = tmp;
    tmp = realloc(self->node_flags, self->max_nodes * sizeof(uint32_t));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->node_flags = tmp;
    tmp = realloc(self->path, self->max_nodes * sizeof(edge_t *));
    if (tmp == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->path = tmp;
    /* Zero out the extra nodes. */
    memset(self->path + self->num_nodes, 0,
        (self->max_nodes - self->num_nodes) * sizeof(edge_t *));
out:
    return ret;
}

tsk_id_t WARN_UNUSED
tree_sequence_builder_add_node(
    tree_sequence_builder_t *self, double time, uint32_t flags)
{
    int ret = 0;

    if (self->num_nodes == self->max_nodes) {
        ret = tree_sequence_builder_expand_nodes(self);
        if (ret != 0) {
            goto out;
        }
    }
    assert(self->num_nodes < self->max_nodes);
    ret = (int) self->num_nodes;
    self->time[ret] = time;
    self->node_flags[ret] = flags;
    self->num_nodes++;
out:
    return ret;
}

int WARN_UNUSED
tree_sequence_builder_add_mutation(
    tree_sequence_builder_t *self, tsk_id_t site, tsk_id_t node, allele_t derived_state)
{
    int ret = 0;
    mutation_list_node_t *list_node, *tail;

    if (node < 0 || node >= (tsk_id_t) self->num_nodes) {
        ret = TSI_ERR_BAD_MUTATION_NODE;
        goto out;
    }
    if (site < 0 || site >= (tsk_id_t) self->num_sites) {
        ret = TSI_ERR_BAD_MUTATION_SITE;
        goto out;
    }
    if (derived_state < 0) {
        ret = TSI_ERR_BAD_MUTATION_DERIVED_STATE;
        goto out;
    }
    /* Make sure there are no other mutations on this node */
    for (list_node = self->sites.mutations[site]; list_node != NULL;
         list_node = list_node->next) {
        if (list_node->node == node) {
            ret = TSI_ERR_BAD_MUTATION_DUPLICATE_NODE;
            goto out;
        }
    }

    list_node = tsi_blkalloc_get(&self->tsi_blkalloc, sizeof(mutation_list_node_t));
    if (list_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    list_node->node = node;
    list_node->derived_state = derived_state;
    list_node->next = NULL;
    if (self->sites.mutations[site] == NULL) {
        self->sites.mutations[site] = list_node;
    } else {
        tail = self->sites.mutations[site];
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = list_node;
    }
    self->num_mutations++;
out:
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_unindex_edge(tree_sequence_builder_t *self, indexed_edge_t *edge)
{
    int ret = 0;
    avl_node_t *avl_node;

    avl_node = avl_search(&self->left_index, edge);
    assert(avl_node != NULL);
    avl_unlink_node(&self->left_index, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);

    avl_node = avl_search(&self->right_index, edge);
    assert(avl_node != NULL);
    avl_unlink_node(&self->right_index, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);

    avl_node = avl_search(&self->path_index, edge);
    assert(avl_node != NULL);
    avl_unlink_node(&self->path_index, avl_node);
    tree_sequence_builder_free_avl_node(self, avl_node);
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_index_edge(tree_sequence_builder_t *self, indexed_edge_t *edge)
{
    int ret = 0;
    avl_node_t *avl_node;

    avl_node = tree_sequence_builder_alloc_avl_node(self, edge);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->left_index, avl_node);
    assert(avl_node != NULL);

    avl_node = tree_sequence_builder_alloc_avl_node(self, edge);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->right_index, avl_node);
    assert(avl_node != NULL);

    avl_node = tree_sequence_builder_alloc_avl_node(self, edge);
    if (avl_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    avl_node = avl_insert_node(&self->path_index, avl_node);
    assert(avl_node != NULL);
out:
    return ret;
}

static int WARN_UNUSED
tree_sequence_builder_index_edges(tree_sequence_builder_t *self, tsk_id_t node)
{
    int ret = 0;
    indexed_edge_t *e;

    for (e = self->path[node]; e != NULL; e = e->next) {
        ret = tree_sequence_builder_index_edge(self, e);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* Looks up the path index to find a matching edge, and returns it.
 */
static indexed_edge_t *
tree_sequence_builder_find_match(tree_sequence_builder_t *self, indexed_edge_t *query)
{
    indexed_edge_t *ret = NULL;
    indexed_edge_t search, *found;
    avl_node_t *avl_node;

    search.edge.left = query->edge.left;
    search.edge.right = query->edge.right;
    search.edge.parent = query->edge.parent;
    search.edge.child = 0;

    avl_search_closest(&self->path_index, &search, &avl_node);
    if (avl_node != NULL) {
        found = (indexed_edge_t *) avl_node->item;
        if (found->edge.left == query->edge.left
            && found->edge.right == query->edge.right
            && found->edge.parent == query->edge.parent) {
            ret = found;
        } else {
            /* Check the adjacent nodes. */
            if (avl_node->prev != NULL) {
                found = (indexed_edge_t *) avl_node->prev->item;
                if (found->edge.left == query->edge.left
                    && found->edge.right == query->edge.right
                    && found->edge.parent == query->edge.parent) {
                    ret = found;
                }
            }
            if (ret == NULL && avl_node->next != NULL) {
                found = (indexed_edge_t *) avl_node->next->item;
                if (found->edge.left == query->edge.left
                    && found->edge.right == query->edge.right
                    && found->edge.parent == query->edge.parent) {
                    ret = found;
                }
            }
        }
    }
    return ret;
}

typedef struct {
    indexed_edge_t *source;
    indexed_edge_t *dest;
} edge_map_t;

static void
tree_sequence_builder_squash_edges(tree_sequence_builder_t *self, tsk_id_t node)
{
    indexed_edge_t *x, *prev, *next;

    prev = self->path[node];
    assert(prev != NULL);
    x = prev->next;
    while (x != NULL) {
        next = x->next;
        assert(x->edge.child == node);
        if (prev->edge.right == x->edge.left && prev->edge.parent == x->edge.parent) {
            prev->edge.right = x->edge.right;
            prev->next = next;
            tree_sequence_builder_free_edge(self, x);
        } else {
            prev = x;
        }
        x = next;
    }
}

/* Squash edges that can be squashed, but take into account that any modified
 * edges must be re-indexed. Some edges in the input chain may already be unindexed,
 * which are marked with a child value of NULL_NODE. */
static int WARN_UNUSED
tree_sequence_builder_squash_indexed_edges(tree_sequence_builder_t *self, tsk_id_t node)
{
    int ret = 0;
    indexed_edge_t *x, *prev, *next;

    prev = self->path[node];
    assert(prev != NULL);
    x = prev->next;
    while (x != NULL) {
        next = x->next;
        if (prev->edge.right == x->edge.left && prev->edge.parent == x->edge.parent) {
            /* We are pulling x out of the chain and extending prev to cover
             * the corresponding interval. Therefore, we must unindex prev and x. */
            if (prev->edge.child != NULL_NODE) {
                ret = tree_sequence_builder_unindex_edge(self, prev);
                if (ret != 0) {
                    goto out;
                }
                prev->edge.child = NULL_NODE;
            }
            if (x->edge.child != NULL_NODE) {
                ret = tree_sequence_builder_unindex_edge(self, x);
                if (ret != 0) {
                    goto out;
                }
            }
            prev->edge.right = x->edge.right;
            prev->next = next;
            tree_sequence_builder_free_edge(self, x);
        } else {
            prev = x;
        }
        x = next;
    }

    /* Now index all the edges that have been unindexed */
    for (x = self->path[node]; x != NULL; x = x->next) {
        if (x->edge.child == NULL_NODE) {
            x->edge.child = node;
            ret = tree_sequence_builder_index_edge(self, x);
            if (ret != 0) {
                goto out;
            }
        }
    }
out:
    return ret;
}

/* Create a new pc ancestor which consists of the shared path
 * segments of existing ancestors. */
static int
tree_sequence_builder_make_pc_node(
    tree_sequence_builder_t *self, edge_map_t *mapped, size_t num_mapped)
{
    int ret = 0;
    tsk_id_t pc_node;
    indexed_edge_t *edge;
    indexed_edge_t *head = NULL;
    indexed_edge_t *prev = NULL;
    double min_parent_time;
    tsk_id_t mapped_child = mapped[0].dest->edge.child;
    double mapped_child_time = self->time[mapped_child];
    size_t j;

    min_parent_time = self->time[0] + 1;
    for (j = 0; j < num_mapped; j++) {
        assert(mapped[j].dest->edge.child == mapped_child);
        min_parent_time
            = TSK_MIN(min_parent_time, self->time[mapped[j].source->edge.parent]);
    }
    min_parent_time -= PC_ANCESTOR_INCREMENT;
    if (min_parent_time <= mapped_child_time) {
        ret = TSI_ERR_PC_ANCESTOR_TIME;
        goto out;
    }

    ret = tree_sequence_builder_add_node(self, min_parent_time, TSI_NODE_IS_PC_ANCESTOR);
    if (ret < 0) {
        goto out;
    }
    pc_node = ret;
    self->num_match_nodes++;

    for (j = 0; j < num_mapped; j++) {
        edge = tree_sequence_builder_alloc_edge(self, mapped[j].source->edge.left,
            mapped[j].source->edge.right, mapped[j].source->edge.parent, pc_node, NULL);
        if (edge == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        if (head == NULL) {
            head = edge;
        } else {
            prev->next = edge;
        }
        prev = edge;
        mapped[j].source->edge.parent = pc_node;
        /* We are modifying the existing edge, so we must remove it
         * from the indexes. Mark that it is unindexed by setting the
         * child value to NULL_NODE. */
        ret = tree_sequence_builder_unindex_edge(self, mapped[j].dest);
        if (ret != 0) {
            goto out;
        }
        mapped[j].dest->edge.parent = pc_node;
        mapped[j].dest->edge.child = NULL_NODE;
    }
    self->path[pc_node] = head;
    tree_sequence_builder_squash_edges(self, pc_node);
    ret = tree_sequence_builder_squash_indexed_edges(self, mapped_child);
    if (ret != 0) {
        goto out;
    }
    ret = tree_sequence_builder_index_edges(self, pc_node);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

static int
tree_sequence_builder_compress_path(tree_sequence_builder_t *self, tsk_id_t child)
{
    int ret = 0;
    indexed_edge_t *c_edge, *match_edge;
    edge_t last_match;
    edge_map_t *mapped = NULL;
    size_t *contig_offsets = NULL;
    size_t path_length = 0;
    size_t num_contigs = 0;
    size_t num_mapped = 0;
    size_t j, k, contig_size;
    tsk_id_t mapped_child;

    for (c_edge = self->path[child]; c_edge != NULL; c_edge = c_edge->next) {
        path_length++;
    }
    mapped = malloc(path_length * sizeof(*mapped));
    contig_offsets = malloc((path_length + 1) * sizeof(*contig_offsets));
    if (mapped == NULL || contig_offsets == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    last_match.right = -1;
    last_match.child = NULL_NODE;

    for (c_edge = self->path[child]; c_edge != NULL; c_edge = c_edge->next) {
        /* Can we find a match for this edge? */
        match_edge = tree_sequence_builder_find_match(self, c_edge);
        if (match_edge != NULL) {
            mapped[num_mapped].source = c_edge;
            mapped[num_mapped].dest = match_edge;
            if (!(c_edge->edge.left == last_match.right
                    && match_edge->edge.child == last_match.child)) {
                contig_offsets[num_contigs] = num_mapped;
                num_contigs++;
            }
            last_match = match_edge->edge;
            num_mapped++;
        }
    }
    contig_offsets[num_contigs] = num_mapped;

    for (j = 0; j < num_contigs; j++) {
        contig_size = contig_offsets[j + 1] - contig_offsets[j];
        if (contig_size > 1) {
            mapped_child = mapped[contig_offsets[j]].dest->edge.child;
            if ((self->node_flags[mapped_child] & TSI_NODE_IS_PC_ANCESTOR) != 0) {
                /* Remap the edges in the set of matches to point to the already
                 * existing synthethic node. */
                for (k = contig_offsets[j]; k < contig_offsets[j + 1]; k++) {
                    mapped[k].source->edge.parent = mapped_child;
                }
            } else {
                ret = tree_sequence_builder_make_pc_node(
                    self, mapped + contig_offsets[j], contig_size);
                if (ret != 0) {
                    goto out;
                }
            }
        }
    }
    tree_sequence_builder_squash_edges(self, child);
out:
    tsi_safe_free(mapped);
    tsi_safe_free(contig_offsets);
    return ret;
}

static int
tree_sequence_builder_check_edge(tree_sequence_builder_t *self, tsk_id_t left,
    tsk_id_t right, tsk_id_t parent, tsk_id_t child)
{
    int ret = 0;
    double child_time;

    if (child < 0 || child >= (tsk_id_t) self->num_nodes) {
        ret = TSI_ERR_BAD_PATH_CHILD;
        goto out;
    }
    child_time = self->time[child];

    if (parent < 0 || parent >= (tsk_id_t) self->num_nodes) {
        ret = TSI_ERR_BAD_PATH_PARENT;
        goto out;
    }
    if (self->time[parent] <= child_time) {
        ret = TSI_ERR_BAD_PATH_TIME;
        goto out;
    }
    if (left < 0) {
        ret = TSI_ERR_BAD_PATH_LEFT_LESS_ZERO;
        goto out;
    }
    if (right > (tsk_id_t) self->num_sites) {
        ret = TSI_ERR_BAD_PATH_RIGHT_GREATER_NUM_SITES;
        goto out;
    }
    if (left >= right) {
        ret = TSI_ERR_BAD_PATH_INTERVAL;
        goto out;
    }
out:
    return ret;
}

int
tree_sequence_builder_add_path(tree_sequence_builder_t *self, tsk_id_t child,
    size_t num_edges, tsk_id_t *left, tsk_id_t *right, tsk_id_t *parent, int flags)
{
    int ret = 0;
    indexed_edge_t *head = NULL;
    indexed_edge_t *prev = NULL;
    indexed_edge_t *e;
    int j;

    /* Edges must be provided in reverse order */
    for (j = (int) num_edges - 1; j >= 0; j--) {
        ret = tree_sequence_builder_check_edge(
            self, left[j], right[j], parent[j], child);
        if (ret != 0) {
            goto out;
        }
        e = tree_sequence_builder_alloc_edge(
            self, left[j], right[j], parent[j], child, NULL);
        if (e == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        if (head == NULL) {
            head = e;
        } else {
            prev->next = e;
            if (prev->edge.right != e->edge.left) {
                ret = TSI_ERR_NONCONTIGUOUS_EDGES;
                goto out;
            }
        }
        prev = e;
    }
    self->num_match_nodes++;
    self->path[child] = head;
    if (flags & TSI_COMPRESS_PATH) {
        ret = tree_sequence_builder_compress_path(self, child);
        if (ret != 0) {
            goto out;
        }
    }
    ret = tree_sequence_builder_index_edges(self, child);
    if (flags & TSI_EXTENDED_CHECKS) {
        tree_sequence_builder_check_state(self);
    }
out:
    return ret;
}

int
tree_sequence_builder_add_mutations(tree_sequence_builder_t *self, tsk_id_t node,
    size_t num_mutations, tsk_id_t *site, allele_t *derived_state)
{
    int ret = 0;
    size_t j;

    for (j = 0; j < num_mutations; j++) {
        ret = tree_sequence_builder_add_mutation(self, site[j], node, derived_state[j]);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* Freeze the tree traversal indexes from the state of the dynamic AVL
 * tree based indexes. This is done because it is *much* more efficient
 * to get the edges sequentially than to find the randomly around memory
 *
 * This also means that edges and mutations added will have no effect
 * on matching *until* freeze_indexes is called.
 */
int
tree_sequence_builder_freeze_indexes(tree_sequence_builder_t *self)
{
    int ret = 0;
    avl_node_t *restrict a;
    size_t j = 0;

    tsi_safe_free(self->left_index_edges);
    tsi_safe_free(self->right_index_edges);
    self->num_edges = avl_count(&self->left_index);
    assert(self->num_edges == avl_count(&self->right_index));

    self->left_index_edges = malloc(self->num_edges * sizeof(*self->left_index_edges));
    self->right_index_edges = malloc(self->num_edges * sizeof(*self->right_index_edges));
    if (self->left_index_edges == NULL || self->right_index_edges == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }

    j = 0;
    for (a = self->left_index.head; a != NULL; a = a->next) {
        self->left_index_edges[j] = ((indexed_edge_t *) a->item)->edge;
        j++;
    }
    j = 0;
    for (a = self->right_index.head; a != NULL; a = a->next) {
        self->right_index_edges[j] = ((indexed_edge_t *) a->item)->edge;
        j++;
    }
out:
    return ret;
}

int
tree_sequence_builder_restore_nodes(
    tree_sequence_builder_t *self, size_t num_nodes, uint32_t *flags, double *time)
{
    int ret = -1;
    size_t j;

    for (j = 0; j < num_nodes; j++) {
        ret = tree_sequence_builder_add_node(self, time[j], flags[j]);
        if (ret < 0) {
            goto out;
        }
    }
    ret = 0;
out:
    return ret;
}

int
tree_sequence_builder_restore_edges(tree_sequence_builder_t *self, size_t num_edges,
    tsk_id_t *left, tsk_id_t *right, tsk_id_t *parent, tsk_id_t *child)
{
    int ret = -1;
    size_t j;
    indexed_edge_t *e, *prev;

    prev = NULL;
    for (j = 0; j < num_edges; j++) {
        ret = tree_sequence_builder_check_edge(
            self, left[j], right[j], parent[j], child[j]);
        if (ret != 0) {
            goto out;
        }
        if (j > 0 && child[j - 1] > child[j]) {
            ret = TSI_ERR_UNSORTED_EDGES;
            goto out;
        }
        e = tree_sequence_builder_alloc_edge(
            self, left[j], right[j], parent[j], child[j], NULL);
        if (e == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        if (self->path[child[j]] == NULL) {
            self->path[child[j]] = e;
            self->num_match_nodes++;
        } else {
            if (prev->edge.right > e->edge.left) {
                ret = TSI_ERR_UNSORTED_EDGES;
                goto out;
            }
            prev->next = e;
        }
        ret = tree_sequence_builder_index_edge(self, e);
        if (ret != 0) {
            goto out;
        }
        prev = e;
    }
    ret = tree_sequence_builder_freeze_indexes(self);
out:
    return ret;
}

int
tree_sequence_builder_restore_mutations(tree_sequence_builder_t *self,
    size_t num_mutations, tsk_id_t *site, tsk_id_t *node, allele_t *derived_state)
{
    int ret = 0;
    size_t j = 0;

    for (j = 0; j < num_mutations; j++) {
        ret = tree_sequence_builder_add_mutation(
            self, site[j], node[j], derived_state[j]);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

int
tree_sequence_builder_dump_nodes(
    tree_sequence_builder_t *self, uint32_t *flags, double *time)
{
    int ret = 0;
    size_t j;

    for (j = 0; j < self->num_nodes; j++) {
        flags[j] = self->node_flags[j];
        time[j] = self->time[j];
    }
    return ret;
}

int
tree_sequence_builder_dump_edges(tree_sequence_builder_t *self, tsk_id_t *left,
    tsk_id_t *right, tsk_id_t *parent, tsk_id_t *child)
{
    int ret = 0;
    size_t j, u;
    indexed_edge_t *e;

    j = 0;
    for (u = 0; u < self->num_nodes; u++) {
        e = self->path[u];
        while (e != NULL) {
            left[j] = e->edge.left;
            right[j] = e->edge.right;
            parent[j] = e->edge.parent;
            child[j] = e->edge.child;
            e = e->next;
            j++;
        }
    }
    return ret;
}

int
tree_sequence_builder_dump_mutations(tree_sequence_builder_t *self, tsk_id_t *site,
    tsk_id_t *node, allele_t *derived_state, tsk_id_t *parent)
{
    int ret = 0;
    tsk_id_t l;
    mutation_list_node_t *u;
    tsk_id_t j = 0;

    for (l = 0; l < (tsk_id_t) self->num_sites; l++) {
        for (u = self->sites.mutations[l]; u != NULL; u = u->next) {
            site[j] = l;
            node[j] = u->node;
            derived_state[j] = u->derived_state;
            /* For now we don't compute the mutation parent. We'd need to
             * update the data structures in here to do this. */
            parent[j] = -1;
            j++;
        }
    }
    return ret;
}

size_t
tree_sequence_builder_get_num_nodes(tree_sequence_builder_t *self)
{
    return self->num_nodes;
}

size_t
tree_sequence_builder_get_num_edges(tree_sequence_builder_t *self)
{
    return avl_count(&self->left_index);
}

size_t
tree_sequence_builder_get_num_mutations(tree_sequence_builder_t *self)
{
    return self->num_mutations;
}
