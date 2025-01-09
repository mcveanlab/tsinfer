# Changelog

## [0.4.0a3] - ****-**-**

**Features**

- Document that the `zarr-vcf` dataset can be either a path or an in-memory zarr group.
  (feature introduced in {pr}`966`, documented in {pr}`974`, {user}`hyanwong`)

- Allow a contig to be selected by name (`contig_id`), and get the `sequence_length`
  of the contig associated with the unmasked sites, if contig lengths are provided
  ({pr}`964`, {user}`hyanwong`)

**Fixes**

- Properly account for "N" as an unknown ancestral state, and ban "" from being
  set as an ancestral state ({pr}`963`, {user}`hyanwong`)

## [0.4.0a2] - 2024-09-06

2nd Alpha release of tsinfer 0.4.0

**Features**

- Add batch ancestor and sample matching APIs for splitting work across many independent jobs.
  ({pr}`954`, {pr}`917`, {user}`benjeffery`)
  

## [0.4.0a1] - 2024-07-26

Alpha release of tsinfer 0.4.0

**Features**

- `tsinfer` now supports inferring data from an `vcf-zarr` dataset. This allows users 
  to infer from VCFs via the optimised and parallel VCF parsing in `bio2zarr`.
- The `VariantData` class can be used to load the vcf-data and be used for inference.
- `vcf-zarr` `sample_ids` are inserted into individual metadata as `variant_data_sample_id`
  if this key does not already exist.

**Breaking Changes**

- Remove the `uuid` field from SampleData. SampleData equality is now purely based
  on data. ({pr}`748`, {user}`benjeffery`)

**Performance improvements**

- Reduce memory usage when running `match_samples` against large cohorts
  containing sequences with substantial amounts of error.
  ({pr}`761`, {user}`jeromekelleher`)

- `truncate_ancestors` no longer requires loading all the ancestors into RAM.
  ({pr}`811`, {user}`benjeffery`)

- Reduce memory requirements of the `generate_ancestors` function by providing
  the `genotype_encoding` ({pr}`809`) and `mmap_temp_dir` ({pr}`808`) options
  ({user}`jeromekelleher`).

- Increase parallelisation of `match_ancestors` by generating parallel groups from
  their implied dependency graph. ({pr}`828`, {issue}`147`,  {user}`benjeffery`)

## [0.3.3] - 2024-07-17

**Fixes**
  - Bug fix release for numpy 2 ({issue}`937`).

**Breaking Changes**
  - A permissive json schema is now set on node table metadata
    ({issue}`416` {pr}`931`, {user}`hyanwong`).

## [0.3.2] - 2024-07-16

**Features**

  - `tsinfer` now supports numpy2 (and 1.XX) and python 3.12.

**Breaking Changes**

  - tsinfer now requires Python 3.9 or later

## [0.3.1] - 2023-04-19
 
- Bug fix release for a bad dependency specification.


## [0.3.0] - 2022-10-25

**Features**

- When calling `sample_data.add_site()` the ancestral state does not need to be the
  first allele (index 0): alternatively, an ancestral allele index can be given
  (and if `MISSING_DATA`, the ancestral state will be imputed). ({pr}`718`,
  {issue}`686` {user}`hyanwong`)

- The CLI interface now allows recombination rate (or rate maps) and mismatch ratios
  to be specified ({pr}`731`, {issue}`435` {user}`hyanwong`)

- The calls to match-ancestors and match-samples via the CLI are now logged
  in the provenance entries of the output tree sequence ({pr}`732` and `741`,
  {issue}`730` {user}`hyanwong`)

- The CLI interface allows `--no-post-process` to be specified (for details of post-
  processing, see "Breaking changes" below) ({pr}`727`, {issue}`721` {user}`hyanwong`)

- matching routines warn if no inference sites
  ({pr}`685`, {issue}`683` {user}`hyanwong`)

**Fixes**

- `sample_data.subset()` now accepts a sequence_length  ({pr}`681`, {user}`hyanwong`)

- `verify` no longer raises error when comparing a genotype to missingness.
  ({pr}`716`, {issue}`625`, {user}`benjeffery`)

**Breaking changes**:

- The `simplify` parameter is now deprecated in favour of `post_process`, which
  prior to simplification, removes the "virtual-root-like" ancestor (inserted by
  tsinfer to aid the matching process) then splits the ultimate ancestor into separate
  pieces. If splitting is not required, the `post_process` step can also be called as a
  separate function with the parameter `split_ultimate=False` ({pr}`687`, {pr}`750`,
  {issue}`673`, {user}`hyanwong`)

- Post-processing by default erases tree topology that exists before the first site
  and one unit after the last site, to avoid extrapolating into regions with no data.
  This can be disabled by calling `post_process` step as a separate function with the
  parameter `erase_flanks=False` ({pr}`720`, {issue}`483`, {user}`hyanwong`)

- Inference now sets time_units on both ancestor and final tree sequences to
  tskit.TIME_UNITS_UNCALIBRATED, stopping accidental use of branch length
  calculations on the ts. ({pr}`680`, {user}`hyanwong`)

## [0.2.3] - 2022-04-08

**Features**

- Added `ancestor(id)` to `AncestorData` class.
  ({pr}`570`, {issue}`569`, {user}`hyanwong`)

**Fixes**

- Mark zarr 2.11.0, 2.11.1 and 2.11.2 as incompatible due to `zarr-python`
  bugs #965 and #967.
  ({issue}`643`, {pr}`657`, {user}`benjeffery`)

## [0.2.2] - 2022-02-23

**Bugfixes**:

- Mutations at non-inference sites are now guaranteed to be fully parsimonious.
  Previous versions required a mutation above the root when the input ancestral state
  disagreed with the ancestral state produced by the parsimony algorithm. Now fixed by
  using the new map_mutations code from tskit 0.3.7 ({pr}`557`, {user}`hyanwong`)

**New Features**:

**Breaking changes**:

- Oldest nodes in a standard inferred tree sequence are no longer set to frequencies ~2
  and ~3 (i.e. 2 or 3 times as old as all the other nodes), but are spaced above the
  others by the mean time between unique ancestor ages ({pr}`485`, {user}`hyanwong`)

- The `tsinfer.SampleData.from_tree_sequence()` function now defaults to setting
  `use_sites_time` and `use_individuals_time` to `False` rather than `True`
  ({pr}`599`, {user}`hyanwong`)

## [0.2.1] - 2021-05-26

Bugfix release

**Bugfixes**:

- Fix a bug in the core LS matching algorithm in which the rate of recombination
  was being incorrectly computed ({issue}`493`, {pr}`514`, {user}`jeromekelleher`,
  {user}`hyanwong`).

- `tsinfer.verify()` no longer requires that non-ancestral alleles in a SampleData
  and Tree Sequence file are in the same order ({issue}`490`, {pr}`492`,
  {user}`hyanwong`).

**New Features**:

- Inferred ancestral haplotypes may be truncated via
  `AncestorData.truncate_ancestors()` to improve performance when inferring large
  datasets ({issue}`276`, {pr}`467`, {user}`awohns`).

**Breaking changes**:

- tsinfer now requires Python 3.7


## [0.2.0] - 2020-12-18

Major feature release, including some incompatible file format and API updates.

**New features**:

- Mismatch and recombination parameters can now be specified via the
  recombination_rate and mismatch_ratio arguments in the Python API.

- Missing data can be accomodated in SampleData using the tskit.MISSING_DATA
  value in input genotypes. Missing data will be imputed in the output
  tree sequence.

- Metadata schemas for population, individual, site and tree sequence metadata
  can now we be specified in the SampleData format. These will be included
  in the final tree sequence and allow for automatic decoding of JSON metadata.

- Map non-inference sites onto the tree by using the tskit `map_mutations`
  parsimony method. This allows us to support sites with > 2 alleles.

- Historical (non-contemporaneous) samples can now be accommodated in inference,
  assuming that the true dates of ancestors have been set, by using the concept
  of "proxy samples". This is done via the new function
  `AncestorData.insert_proxy_samples()`, then setting the new
  parameter `force_sample_times=True` when matching samples.

- The default tree sequence returned after inference when `simplify=True` retains
  unary nodes (i.e. simplify is done with `keep_unary=True`.


**Breaking changes**:

- The ancestors tree sequence now contains the real alleles and not
  0/1 values as before.

- Times for undated sites now use frequencies (0..1), not as counts (1..num_samples),
  and are now stored as `tskit.UNKNOWN_TIME`, then calculated on the fly in the
  variants() iterator.

- The SampleData file no longer accepts the `inference` argument to add_site.
  This functionality has been replaced by the `exclude_positions` argument
  to the `infer` and `generate_ancestors` functions.

- The SampleData format is now at version 5, and older versions cannot be read.
  Users should rerun their data ingest pipelines.

- Users can specify variant ages, via `sample_data.add_sites(... , time=user_time)`.
  If not `None`, this overrides the default time position of an ancestor, otherwise
  ancestors are ordered in time by using the frequency of the derived variant (#143).

- Change "age" to "time" to match tskit/msprime notation, and to avoid confusion
  with the age since birth of an individual (#149). Together with the 2 changes below,
  this addition bumped the file format to 3.0.

- Add the ability to record user-specified times for individuals, and therefore
  the samples contained in them (currently ignored during inference). Times are
  added using `sample_data.add_individual(... , time=user_time)` (#190).

- Change `tsinfer.UNKNOWN_ALLELE` to `tskit.MISSING_DATA` for marking unknown regions
  of ancestral haplotypes (#188) . This also involves changing the allele storage to a
  signed int from `np.uint8` which matches the tskit v0.2 format for allele storage
  (see https://github.com/tskit-dev/tskit/issues/144).

**Bugfixes**:

- Individuals and populations in the SampleData file are kept in the returned tree
  sequence, even if they are not referenced by any sample. The individual and population
  ids are therefore guaranteed to stay the same between the sample data file and the
  inferred tree sequence. ({pr}`348`)

## [0.1.4] - 2018-12-12

Bugfix release.

- Fix issue caused by upstream changes in numcodecs (#136).

## [0.1.3] - 2018-11-02

Release corresponding to code used in the preprint.

## [0.1.2] - 2018-06-18

Minor update to take advantage of msprime 0.6.0's Population and Individual
objects and fix various bugs.


**Breaking changes**:

- Bumped SampleData file format version to 1.0 because of the addition
  of individuals and populations. Older SampleData files will not be
  readable and must be regenerated.

- Changed the order of the `alleles` and `genotypes` arguments to
  SampleData.add_site.

**New features**:

- Sample and individual metadata now handled correctly.

- Added --no-simplify option to CLI and simplify=True option to infer function.

- Better handling of missing files (raises correct exceptions).

- tsinfer list now presents basic information for .trees files.

**Bug fixes**:

- Degenerate examples with zero inference sites are now rooted (#44)

- Simplify=False results in tree sequence with correct sample nodes.
