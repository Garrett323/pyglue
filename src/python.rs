use super::{SendPtr, label_encode};
use ndarray::{Array2, ArrayView2};
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyString};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const SUPPORTED_TYPES: &str = "numpy.ndarray, pandas.DataFrame";
pub enum OUT {
    Numpy,
    DataFrame(Vec<String>),
}

#[derive(Debug)]
pub struct EncodingInfo {
    pub string_column_indices: Vec<usize>,
    pub _label_maps: HashMap<usize, HashMap<String, Option<u64>>>,
    pub reverse_maps: HashMap<usize, HashMap<u64, String>>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum StringEncoding {
    LabelEncoding,
}

pub fn raise_not_fitted(py: Python<'_>) -> PyErr {
    let validation = py.import("sklearn.exceptions").unwrap();

    let exc = validation.getattr("NotFittedError").unwrap();

    PyErr::from_type(
        exc.cast_into().unwrap(),
        "This estimator is not fitted yet. Call 'fit' before using this estimator.",
    )
}

pub fn raise_if_nan_col(arr: ArrayView2<f64>) -> Result<(), PyErr> {
    match super::all_empty_column(arr) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "The following columns contain only NaNs can {:?}",
            e
        ))),
    }
}

pub fn check_feature_mismatch(expected: usize, actual: usize) -> Result<(), PyErr> {
    if expected != actual {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "The estimator was fitted on {} features, but {} were provided!",
            expected, actual
        )));
    }
    Ok(())
}

pub fn pyany_to_vec(
    obj: &Bound<'_, PyAny>,
    string_encoding: &Option<StringEncoding>,
) -> PyResult<(Array2<f64>, OUT, Option<EncodingInfo>)> {
    let typ = obj.get_type().name()?;
    let out = match typ.to_string().as_str() {
        "ndarray" | "memmap" => OUT::Numpy,
        "DataFrame" => {
            let columns = obj.getattr("columns")?;
            let columns: Vec<String> = match columns.extract::<Vec<String>>() {
                Ok(cols) => cols,
                Err(_) => columns
                    .call_method0("tolist")?
                    .extract::<Vec<i64>>()?
                    .into_iter()
                    .map(|i| i.to_string())
                    .collect(),
            };
            OUT::DataFrame(columns)
        }
        _ => {
            println!("{}", typ);
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Unsupported type: '{}'. Supported types are: {}",
                typ, SUPPORTED_TYPES
            )));
        }
    };
    // happy path not categorical values
    if let Ok(arr) = obj.cast() {
        Ok((arr.readonly().to_owned_array(), out, None))
    }
    // need to deal with categories
    else {
        match string_encoding {
            None => Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                "Provide a way to encode String!".to_string(),
            )),
            Some(enc) => {
                let (data, enc_info) = encode_object_array(obj, enc, &out);
                Ok((data, out, Some(enc_info)))
            }
        }
    }
}

fn encode_object_array(
    arr: &Bound<'_, PyAny>,
    enc: &StringEncoding,
    out: &OUT,
) -> (Array2<f64>, EncodingInfo) {
    let (nrows, ncols) = arr.getattr("shape").unwrap().extract().unwrap();
    let mut data = vec![0f64; nrows * ncols];
    let ptr = std::sync::Arc::new(SendPtr(data.as_mut_ptr()));
    let mut string_column_indices: Vec<usize> = Vec::new();
    let mut label_maps = HashMap::new();
    let mut reverse_maps: HashMap<usize, HashMap<u64, String>> = HashMap::new();
    let values;
    let arr = if let OUT::DataFrame(_) = out {
        values = arr.getattr("values").unwrap();
        values.as_ref()
    } else {
        arr
    };

    for col_idx in 0..ncols {
        let mut numeric: Vec<f64> = Vec::with_capacity(nrows);
        let mut strings: Vec<String> = Vec::new();
        let mut is_string_col = false;

        for row_idx in 0..nrows {
            // numpy supports `arr[(row, col)]` integer tuple indexing.
            let elem = arr.get_item((row_idx, col_idx)).unwrap();

            if is_string_col {
                strings.push(elem.str().expect("cant convert to str").to_string());
            } else if let Ok(v) = elem.extract::<f64>() {
                numeric.push(v);
            } else {
                // First non-numeric element found — retroactively convert all
                // previously-seen numeric values to strings so the column is
                // encoded uniformly.
                is_string_col = true;
                strings = numeric.drain(..).map(|v| v.to_string()).collect();
                strings.push(elem.str().expect("cant convert to str").to_string());
            }
        }

        if is_string_col {
            let (encoded, map) = match enc {
                StringEncoding::LabelEncoding => label_encode(&strings),
            };
            // for (row_idx, val) in
            encoded
                .into_par_iter()
                .enumerate()
                .for_each(|(row_idx, val)| unsafe {
                    *ptr.0.add(row_idx * ncols + col_idx) = val;
                });
            string_column_indices.push(col_idx);
            let reverse: HashMap<u64, String> = map
                .par_iter()
                .filter_map(|(k, v)| v.map(|val| (val, k.clone())))
                .collect();
            label_maps.insert(col_idx, map);
            reverse_maps.insert(col_idx, reverse);
        } else {
            // Purely numeric column — write collected values directly.
            numeric
                .into_par_iter()
                .enumerate()
                .for_each(|(row_idx, val)| unsafe {
                    *ptr.0.add(row_idx * ncols + col_idx) = val;
                });
        }
    }

    (
        Array2::from_shape_vec([nrows, ncols], data).expect("couldn't convert to ndarray"),
        EncodingInfo {
            string_column_indices,
            _label_maps: label_maps,
            reverse_maps,
        },
    )
}

pub fn arr_to_out<'py>(
    py: Python<'py>,
    arr: &Array2<f64>,
    out: OUT,
    enc_info: Option<EncodingInfo>,
) -> PyResult<Bound<'py, PyAny>> {
    match out {
        OUT::Numpy => {
            let mut out = arr.view().to_pyarray(py).into_any();
            if let Some(enc) = enc_info {
                // cast to object array then overwrite string columns
                let np = py.import("numpy")?;
                let obj_arr = np
                    .call_method1("array", (&out,))?
                    .call_method1("astype", ("object",))?;
                for &col_idx in &enc.string_column_indices {
                    let rev = &enc.reverse_maps[&col_idx];
                    for row_idx in 0..arr.nrows() {
                        let encoded = arr[(row_idx, col_idx)] as u64;
                        let label = rev.get(&encoded).map(String::as_str).unwrap_or("NaN");
                        obj_arr.call_method1("__setitem__", ((row_idx, col_idx), label))?;
                    }
                }
                out = obj_arr.into_any();
            }
            Ok(out)
        }
        OUT::DataFrame(columns) => {
            let pd = PyModule::import(py, "pandas")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("columns", &columns)?;
            let df = pd
                .getattr("DataFrame")?
                .call((arr.view().to_pyarray(py),), Some(&kwargs))?;

            if let Some(enc) = enc_info {
                for &col_idx in &enc.string_column_indices {
                    let rev = &enc.reverse_maps[&col_idx];
                    let decoded: Vec<Py<PyAny>> = (0..arr.nrows())
                        .map(|r| {
                            let v = arr[(r, col_idx)];
                            if v.is_nan() {
                                py.None()
                            } else {
                                rev.get(&(v as u64))
                                    .map(|s| PyString::new(py, s).into_any().unbind())
                                    .unwrap_or_else(|| py.None())
                            }
                        })
                        .collect();
                    // cast column to object dtype first
                    df.call_method1(
                        "__setitem__",
                        (
                            &columns[col_idx],
                            df.get_item(&columns[col_idx])?
                                .call_method1("astype", ("object",))?,
                        ),
                    )?;

                    // now assign strings
                    let loc = df.getattr("loc")?;
                    loc.set_item((pyo3::types::PySlice::full(py), &columns[col_idx]), decoded)?;
                }
            }
            Ok(df.into_any())
        }
    }
}
