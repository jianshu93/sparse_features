use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rayon::prelude::*;
use std::{
    env,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    process,
};

fn main() -> io::Result<()> {
    // We expect exactly 2 arguments:
    //   1) Path to the file with feature names (one per line)
    //   2) Number of samples to generate (columns)
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <features_file_path> <num_samples>", args[0]);
        process::exit(1);
    }

    let features_file_path = &args[1];
    let num_samples: usize = match args[2].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Could not parse <num_samples> as an integer.");
            process::exit(1);
        }
    };


    // Read the list of features from file
    let features = read_features(features_file_path)?;
    if features.is_empty() {
        eprintln!("No features found in '{}'.", features_file_path);
        process::exit(1);
    }
    let num_features = features.len();
    println!(
        "Read {} features from '{}'. Generating {} samples (columns).",
        num_features, features_file_path, num_samples
    );

    // Prepare Output (Tab-Separated)

    // We'll create an output file named "sparse_dataset.tsv".
    // Each row:  FeatureName [TAB] val1 [TAB] val2 [TAB] ... [TAB] valN
    let output_path = "sparse_dataset.tsv";
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Write a header row:
    //   "Feature    sample1    sample2    ...    sampleN"
    write_header(&mut writer, num_samples)?;


    // Row Generation Strategy
    // - We define a sparsity fraction: fraction of columns (samples) that get a
    //   non-zero integer value per feature/row.
    // - We pick ~ (sparsity * num_samples) distinct columns to fill for that row.
    // - Each chosen column gets a random integer in, e.g., [1..101).
    // - We then "fix" the row so that no single column exceeds 10% of the row's sum.
    // - We write the row out as tab-separated floats (e.g. "12.0").
    //
    // Because we want to preserve the order of features in the output, yet also
    // exploit parallelism, we process features in chunks.

    let sparsity = 0.02; // e.g., 2% of the columns are non-zero per row

    // We'll define a chunk size (number of features to process at once).
    let chunk_size = 1000; 
    let mut start_idx = 0;

    while start_idx < num_features {
        let end_idx = (start_idx + chunk_size).min(num_features);

        // Slice of features for this chunk
        let feature_chunk = &features[start_idx..end_idx];

        // Process each feature in parallel using Rayon
        let rows_data: Vec<String> = feature_chunk
            .par_iter()
            .map(|feature_name| {
                // Generate one row's worth of data (num_samples columns).
                let mut row = generate_one_row(num_samples, sparsity);

                // Ensure no single value > 10% of row's sum
                fix_row_values(&mut row);

                // Format the row as "FeatureName [TAB] 12.0 [TAB] 0.0 [TAB] ... \n"
                format_one_row(feature_name, &row)
            })
            .collect();
        // Write the rows in the correct (original) order
        for row_str in rows_data {
            writer.write_all(row_str.as_bytes())?;
        }

        start_idx = end_idx;
    }

    println!(
        "Done writing '{}' with {} features × {} samples.",
        output_path, num_features, num_samples
    );

    Ok(())
}

/// Reads non-empty lines (feature names) from the given file.
fn read_features(file_path: &str) -> io::Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut features = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            features.push(trimmed.to_string());
        }
    }
    Ok(features)
}

/// Write the header line to the TSV file:
/// Feature [TAB] sample1 [TAB] sample2 [TAB] ... [TAB] sampleN
fn write_header<W: Write>(writer: &mut W, num_samples: usize) -> io::Result<()> {
    write!(writer, "Feature")?;
    for i in 1..=num_samples {
        write!(writer, "\tsample{}", i)?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Generate one row of data with `num_samples` columns, using integer values
/// (in f32 form). We pick about (sparsity * num_samples) columns to fill
/// with a random integer, ensuring at least one non-zero.
fn generate_one_row(num_samples: usize, sparsity: f64) -> Vec<f32> {
    let picks_count = (num_samples as f64 * sparsity).round() as usize;
    let picks_count = picks_count.max(1); // ensure at least 1 pick

    // Initialize all columns as 0.0
    let mut row = vec![0.0_f32; num_samples];

    let mut rng = rand::thread_rng();
    let sample_dist = Uniform::from(0..num_samples);
    // We'll pick integer values in [1..101), then store them as f32
    let value_dist = Uniform::from(1..101);

    use std::collections::HashSet;
    let mut chosen_columns = HashSet::with_capacity(picks_count);

    // Randomly select distinct columns
    while chosen_columns.len() < picks_count {
        let col_idx = sample_dist.sample(&mut rng);
        chosen_columns.insert(col_idx);
    }

    // Assign random integer values as f32
    for &col_idx in &chosen_columns {
        let val = value_dist.sample(&mut rng);
        row[col_idx] = val as f32;
    }

    row
}

/// Ensures that no single column in this row exceeds 10% of the total sum.
/// We do this iteratively: if any value is above `0.1 * row_sum`, cap it,
/// then repeat until stable.
fn fix_row_values(row: &mut [f32]) {
    loop {
        let sum: f32 = row.iter().sum();
        if sum <= 0.0 {
            // If sum is zero (all zero), nothing to fix
            break;
        }

        let mut changed = false;
        let limit = 0.1 * sum;

        // Scan for any values exceeding limit
        for val in row.iter_mut() {
            if *val > limit {
                *val = limit;
                changed = true;
            }
        }

        // If we changed anything, we must re-check because the sum changed
        if changed {
            continue;
        } else {
            break;
        }
    }
}

/// Format a single row as:
/// FeatureName [TAB] val1 [TAB] val2 [TAB] ... [TAB] valN [NEWLINE]
/// Example:
/// "GeneABC\t12.0\t0.0\t5.0\t...\n"
fn format_one_row(feature_name: &str, row_values: &[f32]) -> String {
    let mut line = String::new();
    line.push_str(feature_name);

    for &val in row_values {
        // Print integer-looking float, e.g. 12.0
        // You can adjust the decimal places if you wish
        line.push('\t');
        line.push_str(&format!("{:.1}", val));
    }
    line.push('\n');

    line
}

