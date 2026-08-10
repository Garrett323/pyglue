use ndarray::ArrayView2;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use rayon::prelude::*;
mod python;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub use python::{StringEncoding, arr_to_out, pyany_to_vec};
pub use python::{check_feature_mismatch, raise_if_nan_col, raise_not_fitted};

/// A Python module implemented in Rust.
#[pymodule]
mod pyglue {
    use super::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<Encoder>()?;
        Ok(())
    }
}

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

#[pyclass(module = "pyglue")]
#[derive(Serialize, Deserialize)]
struct Encoder {
    encoding_info: Option<python::EncodingInfo>,
    encoding_strat: StringEncoding,
}
#[pymethods]
impl Encoder {
    #[new]
    #[pyo3(signature = (encoding=None))]
    pub fn new(encoding: Option<&str>) -> PyResult<Encoder> {
        let encoding = if let None = encoding {
            StringEncoding::LabelEncoding
        } else {
            match encoding.unwrap_or("label") {
                "label" => StringEncoding::LabelEncoding,
                s => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "{} is not a supported encoding strategy for strings!",
                        s
                    )));
                }
            }
        };
        Ok(Encoder {
            encoding_info: None,
            encoding_strat: encoding,
        })
    }
    pub fn encode<'py>(
        &mut self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (arr, out, enc_info) = pyany_to_vec(obj, &Some(self.encoding_strat.clone()))?;
        self.encoding_info = enc_info;
        arr_to_out(py, &arr, out, None)
    }
    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (arr, out, _) = pyany_to_vec(obj, &Some(self.encoding_strat.clone()))?;
        arr_to_out(py, &arr, out, self.encoding_info.as_ref())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Serialize whatever fields matter into bytes
        let bytes = bincode::serialize(&self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).into())
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let decoded: Encoder = bincode::deserialize(state.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("failed to unpickle KnnImputerRS: {e}"))
        })?;
        self.encoding_info = decoded.encoding_info;
        self.encoding_strat = decoded.encoding_strat;
        Ok(())
    }
}
