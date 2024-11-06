/*
** Copyright (C) 2018 University of Oxford
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

#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "avl.h"

static int
cmp_time_map(const void *a, const void *b)
{
    const time_map_t *ia = (time_map_t const *) a;
    const time_map_t *ib = (time_map_t const *) b;
    return (ia->time > ib->time) - (ia->time < ib->time);
}

static int
cmp_pattern_map(const void *a, const void *b)
{
    const pattern_map_t *ia = (pattern_map_t const *) a;
    const pattern_map_t *ib = (pattern_map_t const *) b;
    int ret = memcmp(ia->genotypes, ib->genotypes, ia->num_samples * sizeof(allele_t));
    return ret;
}

static void
ancestor_builder_check_state(ancestor_builder_t *self)
{
    size_t count;
    avl_node_t *a, *b;
    pattern_map_t *pattern_map;
    time_map_t *time_map;
    site_list_t *s;

    for (a = self->time_map.head; a != NULL; a = a->next) {
        time_map = (time_map_t *) a->item;
        for (b = time_map->pattern_map.head; b != NULL; b = b->next) {
            pattern_map = (pattern_map_t *) b->item;
            count = 0;
            for (s = pattern_map->sites; s != NULL; s = s->next) {
                assert(self->sites[s->site].time == time_map->time);
                assert(self->sites[s->site].genotypes == pattern_map->genotypes);
                count++;
            }
            assert(pattern_map->num_sites == count);
        }
    }
}

int
ancestor_builder_print_state(ancestor_builder_t *self, FILE *out)
{
    size_t j, k;
    avl_node_t *a, *b;
    pattern_map_t *pattern_map;
    time_map_t *time_map;
    site_list_t *s;

    fprintf(out, "Ancestor builder\n");
    fprintf(out, "num_samples = %d\n", (int) self->num_samples);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);

    fprintf(out, "Sites:\n");
    for (j = 0; j < self->num_sites; j++) {
        fprintf(out, "%d\t%d\t%p\n", (int) j, (int) self->sites[j].time,
            (void *) self->sites[j].genotypes);
    }
    fprintf(out, "Time map:\n");

    for (a = self->time_map.head; a != NULL; a = a->next) {
        time_map = (time_map_t *) a->item;
        fprintf(out, "Epoch: time = %f: %d ancestors\n", time_map->time,
            avl_count(&time_map->pattern_map));
        for (b = time_map->pattern_map.head; b != NULL; b = b->next) {
            pattern_map = (pattern_map_t *) b->item;
            fprintf(out, "\t");
            for (k = 0; k < self->num_samples; k++) {
                fprintf(out, "%d", pattern_map->genotypes[k]);
            }
            fprintf(out, "\t");
            for (s = pattern_map->sites; s != NULL; s = s->next) {
                fprintf(out, "%d ", s->site);
            }
            fprintf(out, "\n");
        }
    }
    fprintf(out, "Descriptors:\n");
    for (j = 0; j < self->num_ancestors; j++) {
        fprintf(out, "%f\t%d: ", self->descriptors[j].time,
            (int) self->descriptors[j].num_focal_sites);
        for (k = 0; k < self->descriptors[j].num_focal_sites; k++) {
            fprintf(out, "%d, ", self->descriptors[j].focal_sites[k]);
        }
        fprintf(out, "\n");
    }
    tsi_blkalloc_print_state(&self->allocator, out);
    ancestor_builder_check_state(self);
    return 0;
}

int
ancestor_builder_alloc(
    ancestor_builder_t *self, size_t num_samples, size_t max_sites, int flags)
{
    int ret = 0;
    unsigned long max_size = 1024 * 1024;

    memset(self, 0, sizeof(ancestor_builder_t));
    if (num_samples <= 1) {
        ret = TSI_ERR_BAD_NUM_SAMPLES;
        goto out;
    }

    self->num_samples = num_samples;
    self->max_sites = max_sites;
    self->num_sites = 0;
    self->flags = flags;
    self->sites = calloc(max_sites, sizeof(site_t));
    self->descriptors = calloc(max_sites, sizeof(ancestor_descriptor_t));
    if (self->sites == NULL || self->descriptors == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    /* Pre-calculate the maximum sizes asked for in other methods when calling
     * tsi_blkalloc_get(&self->allocator, ...)  */
    max_size = TSK_MAX(self->num_samples * sizeof(allele_t), max_size);
    /* NB: using self->max_sites below is probably overkill: the real number should be
     * the maximum number of focal sites in a single ancestor, usually << max_sites */
    max_size = TSK_MAX(self->max_sites * sizeof(tsk_id_t), max_size);
    ret = tsi_blkalloc_init(&self->allocator, max_size);
    if (ret != 0) {
        goto out;
    }
    avl_init_tree(&self->time_map, cmp_time_map, NULL);
out:
    return ret;
}

int
ancestor_builder_free(ancestor_builder_t *self)
{
    tsi_safe_free(self->sites);
    tsi_safe_free(self->descriptors);
    tsi_blkalloc_free(&self->allocator);
    return 0;
}

static time_map_t *
ancestor_builder_get_time_map(ancestor_builder_t *self, double time)
{
    time_map_t *ret = NULL;
    time_map_t search, *time_map;
    avl_node_t *avl_node;

    search.time = time;
    avl_node = avl_search(&self->time_map, &search);
    if (avl_node == NULL) {
        avl_node = tsi_blkalloc_get(&self->allocator, sizeof(*avl_node));
        time_map = tsi_blkalloc_get(&self->allocator, sizeof(*time_map));
        if (avl_node == NULL || time_map == NULL) {
            goto out;
        }
        time_map->time = time;
        avl_init_tree(&time_map->pattern_map, cmp_pattern_map, NULL);
        avl_init_node(avl_node, time_map);
        avl_node = avl_insert_node(&self->time_map, avl_node);
        assert(avl_node != NULL);
        ret = time_map;
    } else {
        ret = (time_map_t *) avl_node->item;
    }
out:
    return ret;
}

static inline void
ancestor_builder_get_consistent_samples(
    ancestor_builder_t *self, tsk_id_t site, tsk_id_t *samples, size_t *num_samples)
{
    tsk_id_t j, k;
    allele_t *restrict genotypes = self->sites[site].genotypes;

    k = 0;
    for (j = 0; j < (tsk_id_t) self->num_samples; j++) {
        if (genotypes[j] == 1) {
            samples[k] = j;
            k++;
        }
    }
    *num_samples = (size_t) k;
}

static int
ancestor_builder_compute_ancestral_states(ancestor_builder_t *self, int direction,
    tsk_id_t focal_site, allele_t *ancestor, tsk_id_t *restrict sample_set,
    bool *restrict disagree, tsk_id_t *last_site_ret)
{
    int ret = 0;
    tsk_id_t last_site = focal_site;
    int64_t l;
    tsk_id_t u;
    size_t j, ones, zeros, tmp_size, sample_set_size, min_sample_set_size;
    double focal_site_time = self->sites[focal_site].time;
    const site_t *restrict sites = self->sites;
    const size_t num_sites = self->num_sites;
    const allele_t *restrict genotypes;
    allele_t consensus;

    ancestor_builder_get_consistent_samples(
        self, focal_site, sample_set, &sample_set_size);
    /* This can't happen because we've already tested for it in
     * ancestor_builder_compute_between_focal_sites */
    assert(sample_set_size > 0);
    memset(disagree, 0, self->num_samples * sizeof(*disagree));
    min_sample_set_size = sample_set_size / 2;

    /* printf("site=%d, direction=%d min_sample_size=%d\n", (int) focal_site, direction,
     */
    /*         (int) min_sample_set_size); */
    for (l = focal_site + direction; l >= 0 && l < (int64_t) num_sites; l += direction) {
        /* printf("\tl = %d\n", (int) l); */
        ancestor[l] = 0;
        last_site = (tsk_id_t) l;
        if (sites[l].time > focal_site_time) {

            /* printf("\t%d\t%d:", (int) l, (int) sample_set_size); */
            /* /1* for (j = 0; j < sample_set_size; j++) { *1/ */
            /* /1*     printf("%d, ", sample_set[j]); *1/ */
            /* /1* } *1/ */

            genotypes = self->sites[l].genotypes;
            ones = 0;
            zeros = 0;
            for (j = 0; j < sample_set_size; j++) {
                switch (genotypes[sample_set[j]]) {
                    case 0:
                        zeros++;
                        break;
                    case 1:
                        ones++;
                        break;
                }
            }
            if (ones + zeros == 0) {
                ancestor[l] = TSK_MISSING_DATA;
            } else {
                if (ones >= zeros) {
                    consensus = 1;
                } else {
                    consensus = 0;
                }
                /* printf("\t:ones=%d, consensus=%d\n", (int) ones, consensus); */
                /* fflush(stdout); */
                for (j = 0; j < sample_set_size; j++) {
                    u = sample_set[j];
                    if (disagree[u] && (genotypes[u] != consensus)
                        && (genotypes[u] != TSK_MISSING_DATA)) {
                        /* This sample has disagreed with consensus twice in a row,
                         * so remove it */
                        /* printf("\t\tremoving %d\n", sample_set[j]); */
                        sample_set[j] = -1;
                    }
                }
                ancestor[l] = consensus;
                /* Repack the sample set */
                tmp_size = 0;
                for (j = 0; j < sample_set_size; j++) {
                    if (sample_set[j] != -1) {
                        sample_set[tmp_size] = sample_set[j];
                        tmp_size++;
                    }
                }
                sample_set_size = tmp_size;
                if (sample_set_size <= min_sample_set_size) {
                    /* printf("BREAK\n"); */
                    break;
                }
                /* For the remaining sample set, set the disagree flags based
                 * on whether they agree with the consensus for this site. */
                for (j = 0; j < sample_set_size; j++) {
                    u = sample_set[j];
                    disagree[u] = ((genotypes[u] != consensus)
                                   && (genotypes[u] != TSK_MISSING_DATA));
                }
            }
        }
    }
    *last_site_ret = last_site;
    return ret;
}

static int
ancestor_builder_compute_between_focal_sites(ancestor_builder_t *self,
    size_t num_focal_sites, tsk_id_t *focal_sites, allele_t *ancestor,
    tsk_id_t *sample_set)
{
    int ret = 0;
    tsk_id_t l;
    size_t j, k, ones, zeros, sample_set_size;
    double focal_site_time;
    const site_t *restrict sites = self->sites;
    const allele_t *restrict genotypes;

    assert(num_focal_sites > 0);
    ancestor_builder_get_consistent_samples(
        self, focal_sites[0], sample_set, &sample_set_size);
    if (sample_set_size == 0) {
        ret = TSI_ERR_BAD_FOCAL_SITE;
        goto out;
    }
    focal_site_time = self->sites[focal_sites[0]].time;

    ancestor[focal_sites[0]] = 1;
    for (j = 1; j < num_focal_sites; j++) {
        ancestor[focal_sites[j]] = 1;
        for (l = focal_sites[j - 1] + 1; l < focal_sites[j]; l++) {
            ancestor[l] = 0;
            if (sites[l].time > focal_site_time) {
                /* printf("\t%d\t%d:", l, (int) sample_set_size); */
                /* for (k = 0; k < sample_set_size; k++) { */
                /*     printf("%d, ", sample_set[k]); */
                /* } */
                genotypes = self->sites[l].genotypes;
                ones = 0;
                zeros = 0;
                for (k = 0; k < sample_set_size; k++) {
                    switch (genotypes[sample_set[k]]) {
                        case 0:
                            zeros++;
                            break;
                        case 1:
                            ones++;
                            break;
                    }
                }
                if (ones + zeros == 0) {
                    ancestor[l] = TSK_MISSING_DATA;
                } else if (ones >= zeros) {
                    ancestor[l] = 1;
                }
            }
        }
    }
out:
    return ret;
}

/* Build the ancestors for sites in the specified focal sites */
int
ancestor_builder_make_ancestor(ancestor_builder_t *self, size_t num_focal_sites,
    tsk_id_t *focal_sites, tsk_id_t *ret_start, tsk_id_t *ret_end, allele_t *ancestor)
{
    int ret = 0;
    tsk_id_t focal_site, last_site;
    tsk_id_t *sample_set = malloc(self->num_samples * sizeof(tsk_id_t));
    bool *restrict disagree = calloc(self->num_samples, sizeof(*disagree));

    if (sample_set == NULL || disagree == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    memset(ancestor, 0xff, self->num_sites * sizeof(*ancestor));

    ret = ancestor_builder_compute_between_focal_sites(
        self, num_focal_sites, focal_sites, ancestor, sample_set);
    if (ret != 0) {
        goto out;
    }

    focal_site = focal_sites[num_focal_sites - 1];
    ret = ancestor_builder_compute_ancestral_states(
        self, +1, focal_site, ancestor, sample_set, disagree, &last_site);
    if (ret != 0) {
        goto out;
    }
    *ret_end = last_site + 1;

    focal_site = focal_sites[0];
    ret = ancestor_builder_compute_ancestral_states(
        self, -1, focal_site, ancestor, sample_set, disagree, &last_site);
    if (ret != 0) {
        goto out;
    }
    *ret_start = last_site;
out:
    tsi_safe_free(sample_set);
    tsi_safe_free(disagree);
    return ret;
}

int WARN_UNUSED
ancestor_builder_add_site(ancestor_builder_t *self, double time, allele_t *genotypes)
{
    int ret = 0;
    site_t *site;
    avl_node_t *avl_node;
    site_list_t *list_node;
    pattern_map_t search, *map_elem;
    avl_tree_t *pattern_map;
    tsk_id_t site_id = (tsk_id_t) self->num_sites;
    time_map_t *time_map = ancestor_builder_get_time_map(self, time);

    if (time_map == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    if (self->num_sites == self->max_sites) {
        ret = TSI_ERR_TOO_MANY_SITES;
        goto out;
    }
    self->num_sites++;
    pattern_map = &time_map->pattern_map;
    site = &self->sites[site_id];
    site->time = time;

    search.genotypes = genotypes;
    search.num_samples = self->num_samples;
    avl_node = avl_search(pattern_map, &search);
    if (avl_node == NULL) {
        avl_node = tsi_blkalloc_get(&self->allocator, sizeof(avl_node_t));
        map_elem = tsi_blkalloc_get(&self->allocator, sizeof(pattern_map_t));
        site->genotypes
            = tsi_blkalloc_get(&self->allocator, self->num_samples * sizeof(allele_t));
        if (avl_node == NULL || map_elem == NULL || site->genotypes == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        memcpy(site->genotypes, genotypes, self->num_samples * sizeof(allele_t));
        avl_init_node(avl_node, map_elem);
        map_elem->genotypes = site->genotypes;
        map_elem->num_samples = self->num_samples;
        map_elem->sites = NULL;
        map_elem->num_sites = 0;
        avl_node = avl_insert_node(pattern_map, avl_node);
        assert(avl_node != NULL);
        if (site->genotypes == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
    } else {
        map_elem = (pattern_map_t *) avl_node->item;
        site->genotypes = map_elem->genotypes;
    }
    map_elem->num_sites++;

    list_node = tsi_blkalloc_get(&self->allocator, sizeof(site_list_t));
    if (list_node == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    list_node->site = site_id;
    list_node->next = map_elem->sites;
    map_elem->sites = list_node;
out:
    return ret;
}

/* Returns true if we should break the an ancestor that spans from focal
 * site a to focal site b */
static bool
ancestor_builder_break_ancestor(ancestor_builder_t *self, tsk_id_t a, tsk_id_t b,
    tsk_id_t *restrict samples, size_t num_samples)
{
    bool ret = false;
    tsk_id_t j, k;
    size_t ones, missing;

    for (j = a + 1; j < b && !ret; j++) {
        if (self->sites[j].time > self->sites[a].time) {
            ones = 0;
            missing = 0;
            for (k = 0; k < (tsk_id_t) num_samples; k++) {
                switch (self->sites[j].genotypes[samples[k]]) {
                    case TSK_MISSING_DATA:
                        missing++;
                        break;
                    case 1:
                        ones++;
                        break;
                }
            }
            if (ones != (num_samples - missing) && ones != 0) {
                ret = true;
            }
        }
    }
    return ret;
}

int
ancestor_builder_finalise(ancestor_builder_t *self)
{
    int ret = 0;
    size_t j, num_consistent_samples;
    avl_node_t *a, *b;
    pattern_map_t *pattern_map;
    time_map_t *time_map;
    site_list_t *s;
    ancestor_descriptor_t *descriptor;
    tsk_id_t *focal_sites = NULL;
    tsk_id_t *p;
    tsk_id_t *consistent_samples = malloc(self->num_samples * sizeof(tsk_id_t));

    if (consistent_samples == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    num_consistent_samples = 0; /* Keep the compiler happy */
    self->num_ancestors = 0;

    /* Return the descriptors in *reverse* order */
    for (a = self->time_map.tail; a != NULL; a = a->prev) {
        time_map = (time_map_t *) a->item;
        for (b = time_map->pattern_map.head; b != NULL; b = b->next) {
            pattern_map = (pattern_map_t *) b->item;
            descriptor = self->descriptors + self->num_ancestors;
            self->num_ancestors++;
            descriptor->time = time_map->time;
            focal_sites = tsi_blkalloc_get(
                &self->allocator, pattern_map->num_sites * sizeof(tsk_id_t));
            if (focal_sites == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            descriptor->focal_sites = focal_sites;
            descriptor->num_focal_sites = pattern_map->num_sites;
            j = pattern_map->num_sites - 1;
            for (s = pattern_map->sites; s != NULL; s = s->next) {
                focal_sites[j] = s->site;
                j--;
            }
            /* Now check to see if we need to split this ancestor up
             * further */
            if (pattern_map->num_sites > 1) {
                ancestor_builder_get_consistent_samples(
                    self, focal_sites[0], consistent_samples, &num_consistent_samples);
            }
            for (j = 0; j < pattern_map->num_sites - 1; j++) {
                if (ancestor_builder_break_ancestor(self, focal_sites[j],
                        focal_sites[j + 1], consistent_samples,
                        num_consistent_samples)) {
                    p = focal_sites + j + 1;
                    descriptor->num_focal_sites = (size_t)(p - descriptor->focal_sites);
                    descriptor = self->descriptors + self->num_ancestors;
                    self->num_ancestors++;
                    descriptor->time = time_map->time;
                    descriptor->num_focal_sites = pattern_map->num_sites - j - 1;
                    descriptor->focal_sites = p;
                }
            }
        }
    }
out:
    tsi_safe_free(consistent_samples);
    return ret;
}
