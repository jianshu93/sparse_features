// Simulate sparse OTU/feature tables from a Newick tree into BIOM (HDF5, CSR+CSC).
// - Clap 4.3 CLI
// - Newick parsing & sanitization
// - Per-sample sparsity ~ Normal(mean, std) clipped to [min, max]
// - Parallel row simulation with Rayon
// - BIOM 2.1 writer compatible with `biom 2.1.16` CLI

use clap::{Arg, Command};
use hdf5::{File as H5File, Result as H5Result, types::VarLenUnicode};
use newick::{Newick, NewickTree, one_from_string};
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

//  Newick helpers
fn sanitize_newick_drop_internal_labels_and_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                i += 1;
                let mut depth = 1;
                while i < bytes.len() && depth > 0 {
                    match bytes[i] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        _ => {}
                    }
                    i += 1;
                }
            }
            b')' => {
                out.push(')');
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                // Optional internal label after ')'
                if i < bytes.len() && bytes[i] == b'\'' {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() {
                            i += 2;
                            continue;
                        }
                        if bytes[i] == b'\'' {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                } else {
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c.is_ascii_whitespace()
                            || matches!(c, b':' | b',' | b')' | b'(' | b';' | b'[')
                        {
                            break;
                        }
                        i += 1;
                    }
                }
            }
            _ => {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    out
}

fn load_newick_taxa(tree_path: &str) -> anyhow::Result<Vec<String>> {
    let raw = std::fs::read_to_string(tree_path)?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    let t: NewickTree = one_from_string(&sanitized)?;

    let mut taxa = Vec::<String>::new();
    for n in t.nodes() {
        if t[n].is_leaf() {
            let nm = t
                .name(n)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("L{n}"));
            taxa.push(nm);
        }
    }
    // Deduplicate (suffix on repeats)
    let mut seen: HashMap<String, usize> = HashMap::new();
    for s in &mut taxa {
        match seen.get_mut(s) {
            None => {
                seen.insert(s.clone(), 1);
            }
            Some(cnt) => {
                let new = format!("{}__{}", s, *cnt);
                *cnt += 1;
                *s = new;
            }
        }
    }
    Ok(taxa)
}

// Alias sampler for weighted sampling

struct AliasSampler {
    prob: Vec<f64>,
    alias: Vec<usize>,
}

impl AliasSampler {
    fn new(weights: &[f64]) -> Self {
        let n = weights.len();
        let sum = weights.iter().fold(0.0, |a, &b| a + b.max(0.0));
        let mut prob = vec![0.0; n];
        let mut alias = vec![0usize; n];

        if n == 0 || sum <= 0.0 {
            return Self { prob, alias };
        }
        let mut scaled: Vec<f64> = weights
            .iter()
            .map(|&w| (w.max(0.0)) * (n as f64) / sum)
            .collect();
        let mut small = Vec::<usize>::new();
        let mut large = Vec::<usize>::new();
        for (i, &p) in scaled.iter().enumerate() {
            if p < 1.0 {
                small.push(i)
            } else {
                large.push(i)
            }
        }
        while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
            prob[s] = scaled[s];
            alias[s] = l;
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            if scaled[l] < 1.0 {
                small.push(l)
            } else {
                large.push(l)
            }
        }
        for i in large.into_iter().chain(small.into_iter()) {
            prob[i] = 1.0;
            alias[i] = i;
        }
        Self { prob, alias }
    }

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let uni = rand::distributions::Uniform::new(0.0f64, 1.0f64);
        let n = self.prob.len();
        if n == 0 {
            return 0;
        }
        let i = rng.gen_range(0..n);
        let u: f64 = uni.sample(rng);
        if u < self.prob[i] { i } else { self.alias[i] }
    }
}

// SplitMix64 for stable per-row seeds

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// CSR to CSC

fn csr_to_csc(
    n_rows: usize,
    n_cols: usize,
    indptr: &[i32],
    indices: &[i32],
    data: &[f64],
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let nnz = indices.len();
    let mut csc_indptr = vec![0i32; n_cols + 1];
    for &j in indices {
        csc_indptr[(j as usize) + 1] += 1;
    }
    for c in 0..n_cols {
        csc_indptr[c + 1] += csc_indptr[c];
    }
    let mut next = csc_indptr.clone();
    let mut csc_indices = vec![0i32; nnz];
    let mut csc_data = vec![0f64; nnz];
    for r in 0..n_rows {
        let start = indptr[r] as usize;
        let end = indptr[r + 1] as usize;
        for p in start..end {
            let j = indices[p] as usize;
            let dst = next[j] as usize;
            csc_indices[dst] = r as i32;
            csc_data[dst] = data[p];
            next[j] += 1;
        }
    }
    (csc_indptr, csc_indices, csc_data)
}

// BIOM writer (2.1)

#[inline]
fn as_vlen_vec(strings: &[String]) -> Vec<VarLenUnicode> {
    strings
        .iter()
        .map(|s| unsafe { VarLenUnicode::from_str_unchecked(s.as_str()) })
        .collect()
}

fn write_biom_hdf5(
    out_path: &str,
    taxa_ids: &[String],
    sample_ids: &[String],
    indptr_u32: &[u32], // CSR (observation-oriented)
    indices_u32: &[u32],
    data: &[f64],
) -> H5Result<()> {
    let n_rows = taxa_ids.len();
    let n_cols = sample_ids.len();
    let nnz = indices_u32.len();

    // Cast to BIOM-required dtypes
    let obs_indptr: Vec<i32> = indptr_u32.iter().map(|&x| x as i32).collect();
    let obs_indices: Vec<i32> = indices_u32.iter().map(|&x| x as i32).collect();

    // Build CSC for sample/matrix (float64 for BIOM)
    let (samp_indptr, samp_indices, samp_data) =
        csr_to_csc(n_rows, n_cols, &obs_indptr, &obs_indices, data);

    let f = H5File::create(out_path)?;

    // groups
    let obs = f.create_group("observation")?;
    let _ = obs.create_group("metadata")?;
    let _ = obs.create_group("group-metadata")?;
    let obs_mat = obs.create_group("matrix")?;

    let samp = f.create_group("sample")?;
    let _ = samp.create_group("metadata")?;
    let _ = samp.create_group("group-metadata")?;
    let samp_mat = samp.create_group("matrix")?;

    // ids (vlen unicode)
    let taxa_v = as_vlen_vec(taxa_ids);
    obs.new_dataset_builder().with_data(&taxa_v).create("ids")?;

    let samp_v = as_vlen_vec(sample_ids);
    samp.new_dataset_builder()
        .with_data(&samp_v)
        .create("ids")?;

    // observation/matrix (CSR, float64) 
    obs_mat
        .new_dataset_builder()
        .with_data(data)
        .create("data")?;
    obs_mat
        .new_dataset_builder()
        .with_data(&obs_indices)
        .create("indices")?;
    obs_mat
        .new_dataset_builder()
        .with_data(&obs_indptr)
        .create("indptr")?;

    // sample/matrix (CSC, float64)
    samp_mat
        .new_dataset_builder()
        .with_data(&samp_data)
        .create("data")?;
    samp_mat
        .new_dataset_builder()
        .with_data(&samp_indices)
        .create("indices")?;
    samp_mat
        .new_dataset_builder()
        .with_data(&samp_indptr)
        .create("indptr")?;

    // top-level required attributes

    // id : <string or null>
    {
        let attr = f.new_attr::<VarLenUnicode>().create("id")?;
        let v: VarLenUnicode = "No Table ID".parse().unwrap();
        attr.write_scalar(&v)?;
    }

    // format : <string> The name and version of the current biom format
    {
        let attr = f.new_attr::<VarLenUnicode>().create("format")?;
        let v: VarLenUnicode = "Biological Observation Matrix 2.1.0".parse().unwrap();
        attr.write_scalar(&v)?;
    }

    // format-url : <url> static URL providing format details
    {
        let attr = f.new_attr::<VarLenUnicode>().create("format-url")?;
        let v: VarLenUnicode = "http://biom-format.org".parse().unwrap();
        attr.write_scalar(&v)?;
    }

    // type : <string> Table type
    {
        let attr = f.new_attr::<VarLenUnicode>().create("type")?;
        let v: VarLenUnicode = "OTU table".parse().unwrap();
        attr.write_scalar(&v)?;
    }

    // generated-by : <string>
    {
        let attr = f.new_attr::<VarLenUnicode>().create("generated-by")?;
        let v: VarLenUnicode = "sparse_features 0.1.1".parse().unwrap();
        attr.write_scalar(&v)?;
    }

    // creation-date : <datetime> any ISO8601 is fine
    {
        let attr = f.new_attr::<VarLenUnicode>().create("creation-date")?;
        let v: VarLenUnicode = "1970-01-01T00:00:00".parse().unwrap();
        attr.write_scalar(&v)?;
    }

    // format-version : [major, minor] (as ints)
    let fmt_ver: [i32; 2] = [2, 1];
    f.new_attr_builder()
        .with_data(&fmt_ver)
        .create("format-version")?;

    // shape : [N, M]
    let shape_i32: [i32; 2] = [n_rows as i32, n_cols as i32];
    f.new_attr_builder().with_data(&shape_i32).create("shape")?;

    // nnz : number of non-zero elements
    let nnz_i32: [i32; 1] = [nnz as i32];
    f.new_attr_builder().with_data(&nnz_i32).create("nnz")?;

    // Make sure everything hits disk
    f.flush()?;
    Ok(())
}

// Simulation core

#[derive(Clone, Debug)]
struct SimParams {
    nsamp: usize,
    mean_sparsity: f64,
    std_sparsity: f64,
    min_sparsity: f64,
    max_sparsity: f64,
    min_count: u32,
    max_count: u32,
    chunk_size: usize,
    max_nz_per_row: Option<usize>,
    seed: u64,
}

#[derive(Clone)]
struct RowCsr {
    indices: Vec<u32>,
    data: Vec<f64>,
}

fn main() -> anyhow::Result<()> {
    let m = Command::new("simulate-biom-from-newick")
        .version("0.1.3")
        .about(
            "Simulate sparse OTU/feature tables from a Newick tree and write BIOM (HDF5 CSR+CSC)",
        )
        .arg(
            Arg::new("tree")
                .short('t')
                .long("tree")
                .required(true)
                .help("Input Newick tree file (taxa from leaf names)"),
        )
        .arg(
            Arg::new("samples")
                .short('n')
                .long("samples")
                .required(true)
                .value_parser(clap::value_parser!(usize))
                .help("Number of samples (columns)"),
        )
        .arg(
            Arg::new("mean_sparsity")
                .long("mean-sparsity")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.02")
                .help("Average per-sample nonzero fraction (e.g., 0.02 = 2%)"),
        )
        .arg(
            Arg::new("std_sparsity")
                .long("std-sparsity")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.005")
                .help("Stddev of per-sample sparsity (Normal, then clipped)"),
        )
        .arg(
            Arg::new("min_sparsity")
                .long("min-sparsity")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.0")
                .help("Lower clip for per-sample sparsity"),
        )
        .arg(
            Arg::new("max_sparsity")
                .long("max-sparsity")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.5")
                .help("Upper clip for per-sample sparsity"),
        )
        .arg(
            Arg::new("min_count")
                .long("min-count")
                .value_parser(clap::value_parser!(u32))
                .default_value("1")
                .help("Minimum count for a nonzero entry"),
        )
        .arg(
            Arg::new("max_count")
                .long("max-count")
                .value_parser(clap::value_parser!(u32))
                .default_value("100")
                .help("Maximum count for a nonzero entry"),
        )
        .arg(
            Arg::new("chunk_size")
                .long("chunk-size")
                .value_parser(clap::value_parser!(usize))
                .default_value("2048")
                .help("Number of taxa to process per chunk (parallelized)"),
        )
        .arg(
            Arg::new("max_nz_per_row")
                .long("max-nz-per-row")
                .value_parser(clap::value_parser!(usize))
                .required(false)
                .help("Optional hard cap of nonzeros per taxon (prevents giant rows)"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_parser(clap::value_parser!(u64))
                .default_value("42")
                .help("Base RNG seed (u64) for reproducibility"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .default_value("simulated.biom")
                .help("Output BIOM (HDF5) path"),
        )
        .arg(
            Arg::new("sample_prefix")
                .long("sample-prefix")
                .default_value("sample")
                .help("Sample ID prefix (IDs will be prefix1..prefixN)"),
        )
        .get_matches();

    let tree_path = m.get_one::<String>("tree").unwrap();
    let nsamp = *m.get_one::<usize>("samples").unwrap();
    let out_path = m.get_one::<String>("output").unwrap();
    let sample_prefix = m.get_one::<String>("sample_prefix").unwrap();

    let params = SimParams {
        nsamp,
        mean_sparsity: *m.get_one::<f64>("mean_sparsity").unwrap(),
        std_sparsity: *m.get_one::<f64>("std_sparsity").unwrap(),
        min_sparsity: *m.get_one::<f64>("min_sparsity").unwrap(),
        max_sparsity: *m.get_one::<f64>("max_sparsity").unwrap(),
        min_count: *m.get_one::<u32>("min_count").unwrap(),
        max_count: *m.get_one::<u32>("max_count").unwrap(),
        chunk_size: *m.get_one::<usize>("chunk_size").unwrap(),
        max_nz_per_row: m.get_one::<usize>("max_nz_per_row").cloned(),
        seed: *m.get_one::<u64>("seed").unwrap(),
    };

    // Parse tree to taxa (leaf names)
    let t0 = Instant::now();
    let taxa = load_newick_taxa(tree_path)?;
    if taxa.is_empty() {
        anyhow::bail!("No leaves/taxa found in the tree.");
    }
    let n_taxa = taxa.len();
    eprintln!(
        "Loaded tree with {} taxa (leaves). Elapsed: {} ms",
        n_taxa,
        t0.elapsed().as_millis()
    );

    // Build sample IDs
    let samples: Vec<String> = (1..=nsamp)
        .map(|i| format!("{}{}", sample_prefix, i))
        .collect();

    // Draw per-sample sparsities ~ Normal(mean, std), clipped to [min, max]
    let t1 = Instant::now();
    let sparsities = {
        let normal = Normal::new(params.mean_sparsity, params.std_sparsity.max(0.0))
            .map_err(|_| anyhow::anyhow!("Invalid Normal(mean, std)"))?;
        let mut rng = ChaCha20Rng::seed_from_u64(params.seed ^ 0xC001_FEED_BAAD_F00D);
        (0..nsamp)
            .map(|_| {
                let mut p = normal.sample(&mut rng);
                if p.is_nan() {
                    p = params.mean_sparsity;
                }
                if p < params.min_sparsity {
                    p = params.min_sparsity;
                }
                if p > params.max_sparsity {
                    p = params.max_sparsity;
                }
                if p <= 0.0 { 1e-12 } else { p }
            })
            .collect::<Vec<f64>>()
    };
    let sum_p: f64 = sparsities.iter().sum();
    let p_bar = sum_p / (nsamp as f64);
    eprintln!(
        "Per-sample sparsities: mean={:.6}, std≈{:.6}, sum_p={:.3}, elapsed={} ms",
        p_bar,
        {
            let m = p_bar;
            let v = sparsities.iter().map(|&x| (x - m) * (x - m)).sum::<f64>()
                / (sparsities.len().max(1) as f64);
            v.sqrt()
        },
        sum_p,
        t1.elapsed().as_millis()
    );

    // Alias sampler across samples
    let alias = AliasSampler::new(&sparsities);

    // Prepare CSR containers
    let mut indptr: Vec<u32> = Vec::with_capacity(n_taxa + 1);
    indptr.push(0);
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f64> = Vec::new();

    // Chunked, parallel row simulation
    let t2 = Instant::now();
    let chunk = params.chunk_size.max(1);
    let poisson = Poisson::new(sum_p.max(1e-12)).unwrap(); // expected nonzeros per row
    let count_dist =
        Uniform::new_inclusive(params.min_count, params.max_count.max(params.min_count));

    let mut start = 0usize;
    while start < n_taxa {
        let end = (start + chunk).min(n_taxa);
        let local: Vec<(usize, &String)> = taxa[start..end]
            .iter()
            .enumerate()
            .map(|(i, s)| (start + i, s))
            .collect();

        let rows: Vec<RowCsr> = local
            .par_iter()
            .map(|(global_row_idx, _name)| {
                let seed = splitmix64(params.seed ^ (*global_row_idx as u64));
                let mut rng = ChaCha20Rng::seed_from_u64(seed);

                let mut k = poisson.sample(&mut rng) as usize;
                if let Some(cap) = params.max_nz_per_row {
                    if k > cap {
                        k = cap;
                    }
                }
                if k == 0 {
                    k = 1;
                }

                let mut set = HashSet::<u32>::with_capacity(k * 2);
                while set.len() < k {
                    let j = alias.sample(&mut rng) as u32;
                    set.insert(j);
                    if set.len() >= params.nsamp {
                        break;
                    }
                }
                if set.is_empty() {
                    set.insert(
                        (alias.sample(&mut rng) as u32).min(params.nsamp.saturating_sub(1) as u32),
                    );
                }

                let mut idx: Vec<u32> = set.into_iter().collect();
                idx.sort_unstable();

                let mut vals: Vec<f64> = Vec::with_capacity(idx.len());
                for _ in &idx {
                    let c = count_dist.sample(&mut rng) as f64;
                    vals.push(c);
                }

                RowCsr {
                    indices: idx,
                    data: vals,
                }
            })
            .collect();

        for row in rows {
            indices.extend_from_slice(&row.indices);
            data.extend_from_slice(&row.data);
            let last = *indptr.last().unwrap();
            indptr.push(last + (row.indices.len() as u32));
        }

        start = end;
    }
    eprintln!(
        "Simulated CSR with nnz={} in {} ms",
        indices.len(),
        t2.elapsed().as_millis()
    );

    // Write BIOM (HDF5, 2.1)
    let t3 = Instant::now();
    write_biom_hdf5(out_path, &taxa, &samples, &indptr, &indices, &data)?;
    eprintln!(
        "Wrote BIOM to '{}' (rows={}, cols={}, nnz={}) in {} ms",
        out_path,
        taxa.len(),
        samples.len(),
        indices.len(),
        t3.elapsed().as_millis()
    );

    Ok(())
}