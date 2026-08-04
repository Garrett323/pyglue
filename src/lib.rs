use ndarray::ArrayView2;
use rayon::prelude::*;
mod python;
use std::collections::HashMap;

struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use python::{StringEncoding, arr_to_out, pyany_to_vec};
pub use python::{check_feature_mismatch, raise_if_nan_col, raise_not_fitted};

pub fn all_empty_column(arr: ArrayView2<f64>) -> Result<(), Vec<usize>> {
    let all_nan_cols: Vec<usize> = (0..arr.ncols())
        .into_par_iter()
        .filter_map(|i| {
            for v in arr.column(i) {
                if !v.is_nan() {
                    return None;
                }
            }
            Some(i)
        })
        .collect();
    if all_nan_cols.len() == 0 {
        Ok(())
    } else {
        Err(all_nan_cols)
    }
}

pub fn label_encode(values: &[String]) -> (Vec<f64>, HashMap<String, Option<u64>>) {
    // Collect unique labels in sorted order.
    let mut unique: Vec<&str> = values.iter().map(String::as_str).collect();
    unique.par_sort_unstable();
    unique.dedup();
    let mut counter = 0;
    let map: HashMap<String, Option<u64>> = unique
        .iter()
        .map(|&s| match s {
            "nan" | "NaN" | "<NA>" => (s.to_owned(), None),
            _ => (s.to_owned(), {
                let e = Some(counter);
                counter += 1;
                e
            }),
        })
        .collect();

    let encoded: Vec<f64> = values
        .par_iter()
        .map(|s| match map[s] {
            None => f64::NAN,
            Some(x) => x as f64,
        })
        .collect();
    (encoded, map)
}
