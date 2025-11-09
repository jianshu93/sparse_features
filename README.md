# Simulate sparse features based on feature name, sparsity for any number of samples

## Usage
```bash
Simulate sparse OTU/feature tables from a Newick tree and write BIOM (HDF5 CSR+CSC)

Usage: sparse_features [OPTIONS] --tree <tree> --samples <samples>

Options:
  -t, --tree <tree>
          Input Newick tree file (taxa from leaf names)
  -n, --samples <samples>
          Number of samples (columns)
      --mean-sparsity <mean_sparsity>
          Average per-sample nonzero fraction (e.g., 0.02 = 2%) [default: 0.02]
      --std-sparsity <std_sparsity>
          Stddev of per-sample sparsity (Normal, then clipped) [default: 0.005]
      --min-sparsity <min_sparsity>
          Lower clip for per-sample sparsity [default: 0.0]
      --max-sparsity <max_sparsity>
          Upper clip for per-sample sparsity [default: 0.5]
      --min-count <min_count>
          Minimum count for a nonzero entry [default: 1]
      --max-count <max_count>
          Maximum count for a nonzero entry [default: 100]
      --chunk-size <chunk_size>
          Number of taxa to process per chunk (parallelized) [default: 2048]
      --max-nz-per-row <max_nz_per_row>
          Optional hard cap of nonzeros per taxon (prevents giant rows)
      --seed <seed>
          Base RNG seed (u64) for reproducibility [default: 42]
  -o, --output <output>
          Output BIOM (HDF5) path [default: simulated.biom]
      --sample-prefix <sample_prefix>
          Sample ID prefix (IDs will be prefix1..prefixN) [default: sample]
  -h, --help
          Print help
  -V, --version
```

## Install
```bash
git clone https://github.com/jianshu93/sparse_features
cd sparse_features
cargo build --release
./target/release/sparse_features -h
```
## Generating feature tables
```bash
sparse_features -t ../../data/ASVs_aligned.tre -o ./ASVs_otu.biom --samples 100
```
