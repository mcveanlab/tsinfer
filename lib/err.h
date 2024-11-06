#ifndef __ERR_H__
#define __ERR_H__

// clang-format off
#define TSI_ERR_GENERIC                                             -1
#define TSI_ERR_NO_MEMORY                                           -2
#define TSI_ERR_NONCONTIGUOUS_EDGES                                 -3
#define TSI_ERR_UNSORTED_EDGES                                      -4
#define TSI_ERR_PC_ANCESTOR_TIME                                    -5
#define TSI_ERR_BAD_PATH_CHILD                                      -6
#define TSI_ERR_BAD_PATH_PARENT                                     -7
#define TSI_ERR_BAD_PATH_TIME                                       -8
#define TSI_ERR_BAD_PATH_INTERVAL                                   -9
#define TSI_ERR_BAD_PATH_LEFT_LESS_ZERO                             -10
#define TSI_ERR_BAD_PATH_RIGHT_GREATER_NUM_SITES                    -11
#define TSI_ERR_MATCH_IMPOSSIBLE                                    -12
#define TSI_ERR_BAD_HAPLOTYPE_ALLELE                                -13
#define TSI_ERR_BAD_NUM_ALLELES                                     -14
#define TSI_ERR_BAD_MUTATION_NODE                                   -15
#define TSI_ERR_BAD_MUTATION_SITE                                   -16
#define TSI_ERR_BAD_MUTATION_DERIVED_STATE                          -17
#define TSI_ERR_BAD_MUTATION_DUPLICATE_NODE                         -18
#define TSI_ERR_BAD_NUM_SAMPLES                                     -19
#define TSI_ERR_TOO_MANY_SITES                                      -20
#define TSI_ERR_BAD_FOCAL_SITE                                      -21
#define TSI_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA             -22
#define TSI_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION              -23
#define TSI_ERR_BAD_ANCESTRAL_STATE                                 -24
// clang-format on

#ifdef __GNUC__
#define WARN_UNUSED __attribute__((warn_unused_result))
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)
#include <stdatomic.h>
#else
/* On windows we don't do any perf related stuff */
#define WARN_UNUSED
#define restrict
/* Although MSVS supports C11, it doesn't seem to include a working version of
 * stdatomic.h, so we can't use it portably. For this experiment we'll just
 * leave it out on Windows, as nobody is doing large-scale tsinfer'ing on
 * Windows. */
#define _Atomic
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

const char *tsi_strerror(int err);

/* FIXME! Including a custom version of the tsk_blkalloc struct here so that
 * we can use c11 atomics on the total_size attribute. Including it in this
 * file and err.c as this is the least noisy place to put it, for now
 * See https://github.com/jeromekelleher/sc2ts/issues/381 for reasoning.
 */

#include "tskit.h"

typedef struct {
    size_t chunk_size; /* number of bytes per chunk */
    size_t top;        /* the offset of the next available byte in the current chunk */
    size_t current_chunk;      /* the index of the chunk currently being used */
    _Atomic size_t total_size; /* the total number of bytes allocated + overhead. */
    size_t total_allocated;    /* the total number of bytes allocated. */
    size_t num_chunks;         /* the number of memory chunks. */
    char **mem_chunks;         /* the memory chunks */
} tsi_blkalloc_t;

extern void tsi_blkalloc_print_state(tsi_blkalloc_t *self, FILE *out);
extern int tsi_blkalloc_reset(tsi_blkalloc_t *self);
extern int tsi_blkalloc_init(tsi_blkalloc_t *self, size_t chunk_size);
extern void *tsi_blkalloc_get(tsi_blkalloc_t *self, size_t size);
extern void tsi_blkalloc_free(tsi_blkalloc_t *self);

#endif /*__ERR_H__*/
