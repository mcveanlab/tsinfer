/*
** Copyright (C) 2020 University of Oxford
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

#include "err.h"

const char *
tsi_strerror(int err)
{
    const char *ret = "Unknown error";

    switch (err) {
        case 0:
            ret = "Normal exit condition. This is not an error!";
            break;

        case TSI_ERR_GENERIC:
            ret = "Generic tsinfer error - please file a bug report.";
            break;
        case TSI_ERR_NO_MEMORY:
            ret = "Out of memory";
            break;
        case TSI_ERR_NONCONTIGUOUS_EDGES:
            ret = "Edges must be contiguous";
            break;
        case TSI_ERR_UNSORTED_EDGES:
            ret = "Edges must be sorted";
            break;
        case TSI_ERR_PC_ANCESTOR_TIME:
            ret = "Failure generating time for path compression ancestor";
            break;
        case TSI_ERR_BAD_PATH_CHILD:
            ret = "Bad path information: child node";
            break;
        case TSI_ERR_BAD_PATH_PARENT:
            ret = "Bad path information: parent node";
            break;
        case TSI_ERR_BAD_PATH_TIME:
            ret = "Bad path information: time";
            break;
        case TSI_ERR_BAD_PATH_INTERVAL:
            ret = "Bad path information: left >= right";
            break;
        case TSI_ERR_BAD_PATH_LEFT_LESS_ZERO:
            ret = "Bad path information: left < 0";
            break;
        case TSI_ERR_BAD_PATH_RIGHT_GREATER_NUM_SITES:
            ret = "Bad path information: right > num_sites";
            break;
        case TSI_ERR_MATCH_IMPOSSIBLE:
            ret = "Unexpected failure to find matching haplotype; please open "
                  "an issue on GitHub";
            break;
        case TSI_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA:
            ret = "Cannot find match: the specified mismatch probability is "
                  "0 or 1 and no matches are possible with these parameters";
            break;
        case TSI_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION:
            ret = "Cannot find match: the specified recombination probability is"
                  "zero and no matches could be found. Increasing the 'precision' "
                  "may help, but recombination values of 0 are not recommended.";
            break;
        case TSI_ERR_BAD_HAPLOTYPE_ALLELE:
            ret = "Input haplotype contains bad allele information.";
            break;
        case TSI_ERR_BAD_NUM_ALLELES:
            ret = "The number of alleles must be between 2 and 127";
            break;
        case TSI_ERR_BAD_MUTATION_NODE:
            ret = "Bad mutation information: node";
            break;
        case TSI_ERR_BAD_MUTATION_SITE:
            ret = "Bad mutation information: site";
            break;
        case TSI_ERR_BAD_MUTATION_DERIVED_STATE:
            ret = "Bad mutation information: derived state";
            break;
        case TSI_ERR_BAD_MUTATION_DUPLICATE_NODE:
            ret = "Bad mutation information: mutation already exists for this node.";
            break;
        case TSI_ERR_BAD_ANCESTRAL_STATE:
            ret = "Bad ancestral state for site";
            break;
        case TSI_ERR_BAD_NUM_SAMPLES:
            ret = "Must have at least 2 samples.";
            break;
        case TSI_ERR_TOO_MANY_SITES:
            ret = "Cannot add more sites than the specified maximum.";
            break;
        case TSI_ERR_BAD_FOCAL_SITE:
            ret = "Bad focal site.";
            break;
    }
    return ret;
}

/* Temporary hack. See notes in err.h for why this code is here. */

#include <stdlib.h>
#include <stdio.h>

void
tsi_blkalloc_print_state(tsi_blkalloc_t *self, FILE *out)
{
    fprintf(out, "Block allocator%p::\n", (void *) self);
    fprintf(out, "\ttop = %lld\n", (long long) self->top);
    fprintf(out, "\tchunk_size = %lld\n", (long long) self->chunk_size);
    fprintf(out, "\tnum_chunks = %lld\n", (long long) self->num_chunks);
    fprintf(out, "\ttotal_allocated = %lld\n", (long long) self->total_allocated);
    fprintf(out, "\ttotal_size = %lld\n", (long long) self->total_size);
}

int TSK_WARN_UNUSED
tsi_blkalloc_reset(tsi_blkalloc_t *self)
{
    int ret = 0;

    self->top = 0;
    self->current_chunk = 0;
    self->total_allocated = 0;
    return ret;
}

int TSK_WARN_UNUSED
tsi_blkalloc_init(tsi_blkalloc_t *self, size_t chunk_size)
{
    int ret = 0;

    tsk_memset(self, 0, sizeof(tsi_blkalloc_t));
    if (chunk_size < 1) {
        ret = TSK_ERR_BAD_PARAM_VALUE;
        goto out;
    }
    self->chunk_size = chunk_size;
    self->top = 0;
    self->current_chunk = 0;
    self->total_allocated = 0;
    self->total_size = 0;
    self->num_chunks = 0;
    self->mem_chunks = malloc(sizeof(char *));
    if (self->mem_chunks == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    self->mem_chunks[0] = malloc(chunk_size);
    if (self->mem_chunks[0] == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    self->num_chunks = 1;
    self->total_size = chunk_size + sizeof(void *);
out:
    return ret;
}

void *TSK_WARN_UNUSED
tsi_blkalloc_get(tsi_blkalloc_t *self, size_t size)
{
    void *ret = NULL;
    void *p;

    if (size > self->chunk_size) {
        goto out;
    }
    if ((self->top + size) > self->chunk_size) {
        if (self->current_chunk == (self->num_chunks - 1)) {
            p = realloc(self->mem_chunks, (self->num_chunks + 1) * sizeof(void *));
            if (p == NULL) {
                goto out;
            }
            self->mem_chunks = p;
            p = malloc(self->chunk_size);
            if (p == NULL) {
                goto out;
            }
            self->mem_chunks[self->num_chunks] = p;
            self->num_chunks++;
            self->total_size += self->chunk_size + sizeof(void *);
        }
        self->current_chunk++;
        self->top = 0;
    }
    ret = self->mem_chunks[self->current_chunk] + self->top;
    self->top += size;
    self->total_allocated += size;
out:
    return ret;
}

void
tsi_blkalloc_free(tsi_blkalloc_t *self)
{
    size_t j;

    for (j = 0; j < self->num_chunks; j++) {
        if (self->mem_chunks[j] != NULL) {
            free(self->mem_chunks[j]);
        }
    }
    if (self->mem_chunks != NULL) {
        free(self->mem_chunks);
    }
}
