use super::{SendPtr, label_encode};
use crate::errors::Errors;
use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyIndexError;
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

#[derive(Debug, Serialize, Deserialize)]
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
            let e = Err(Errors::UnsupportedType {
                unsupported: typ.to_string(),
                supported: Some(SUPPORTED_TYPES.to_string()),
            }
            .into());
            return e;
        }
    };
    // For DataFrames, inspect the underlying NumPy array. Casting the DataFrame
    // itself always fails and incorrectly sends an all-numeric frame through
    // the object-array encoder.
    let values;
    let array_obj = if let OUT::DataFrame(_) = &out {
        values = obj.getattr("values")?;
        values.as_ref()
    } else {
        obj
    };

    // happy path not categorical values
    if let Ok(arr) = array_obj.cast::<PyArray2<f64>>() {
        Ok((arr.readonly().to_owned_array(), out, None))
    }
    // need to deal with categories
    else {
        match string_encoding {
            None => Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                "Provide a way to encode String!".to_string(),
            )),
            Some(enc) => {
                let (data, enc_info) = encode_object_array(obj, enc, &out)?;
                Ok((data, out, Some(enc_info)))
            }
        }
    }
}

fn encode_object_array(
    arr: &Bound<'_, PyAny>,
    enc: &StringEncoding,
    out: &OUT,
) -> PyResult<(Array2<f64>, EncodingInfo)> {
    let (nrows, ncols) = arr.getattr("shape").unwrap().extract().unwrap();
    let mut data = vec![0f64; nrows * ncols];
    let ptr = std::sync::Arc::new(SendPtr(data.as_mut_ptr()));
    let mut label_maps = HashMap::with_capacity(ncols);
    let mut reverse_maps: HashMap<usize, HashMap<u64, String>> = HashMap::with_capacity(ncols);
    let values;
    let arr = if let OUT::DataFrame(_) = out {
        values = arr.getattr("values").unwrap();
        values.as_ref()
    } else {
        arr
    };
    // NumPy string arrays normally use a fixed-width Unicode dtype rather than
    // object dtype. Normalize those arrays so their elements can be handled as
    // Python objects alongside mixed object arrays from pandas.
    let object_arr;
    let arr = match arr.cast::<PyArray2<Py<PyAny>>>() {
        Ok(arr) => arr,
        Err(_) => {
            object_arr = arr.call_method1("astype", ("object",))?;
            object_arr.cast::<PyArray2<Py<PyAny>>>()?
        }
    };
    let arr = arr.try_readonly()?;
    let mut string_cols = vec![false; ncols];
    let mut numeric: Vec<Vec<f64>> = vec![Vec::with_capacity(nrows); ncols];
    let mut strings: Vec<Vec<String>> = vec![Vec::with_capacity(nrows); ncols];
    let res: PyResult<()> = (0..ncols)
        .into_iter()
        .map(|col_idx| {
            (0..nrows)
                .into_iter()
                .map(|row_idx| {
                    // for row_idx in 0..nrows {
                    // numpy supports `arr[(row, col)]` integer tuple indexing.
                    Python::attach(|py| {
                        let elem = arr
                            .get((row_idx, col_idx))
                            .ok_or_else(|| PyIndexError::new_err("indexing Error"))?;
                        // .ok_or_else(|| PyIndexError::new_err("index out of bounds"))?
                        // .bind(py);
                        if string_cols[col_idx] {
                            strings[col_idx].push(
                                elem.bind(py)
                                    .str()?
                                    // .expect("cant convert to str")
                                    .to_string(),
                            );
                        } else if let Ok(v) = elem.bind(py).extract::<f64>() {
                            numeric[col_idx].push(v);
                        } else {
                            // First non-numeric element found — retroactively convert all
                            // previously-seen numeric values to strings so the column is
                            // encoded uniformly.
                            string_cols[col_idx] = true;
                            strings[col_idx] =
                                numeric[col_idx].drain(..).map(|v| v.to_string()).collect();
                            strings[col_idx].push(elem.bind(py).str()?.to_string());
                        }
                        Ok(())
                    })
                })
                .collect()
        })
        .collect();
    res?;

    let maps: Vec<(usize, HashMap<String, Option<u64>>, HashMap<u64, String>)> = (0..ncols)
        .into_par_iter()
        .filter_map(|col_idx| {
            if string_cols[col_idx] {
                let (encoded, map) = match enc {
                    StringEncoding::LabelEncoding => label_encode(&strings[col_idx]),
                };
                // for (row_idx, val) in
                encoded
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(row_idx, val)| unsafe {
                        *ptr.0.add(row_idx * ncols + col_idx) = val;
                    });
                let reverse: HashMap<u64, String> = map
                    .par_iter()
                    .filter_map(|(k, v)| v.map(|val| (val, k.clone())))
                    .collect();
                Some((col_idx, map, reverse))
            } else {
                // Purely numeric column — write collected values directly.
                numeric[col_idx]
                    .par_iter()
                    .enumerate()
                    .for_each(|(row_idx, val)| unsafe {
                        *ptr.0.add(row_idx * ncols + col_idx) = *val;
                    });
                None
            }
        })
        .collect();
    maps.iter().for_each(|(col_idx, map, reverse)| {
        label_maps.insert(*col_idx, map.clone());
        reverse_maps.insert(*col_idx, reverse.clone());
    });
    Ok((
        Array2::from_shape_vec([nrows, ncols], data).expect("couldn't convert to ndarray"),
        EncodingInfo {
            string_column_indices: string_cols
                .iter()
                .enumerate()
                .filter_map(|(c, b)| if *b { Some(c) } else { None })
                .collect(),
            _label_maps: label_maps,
            reverse_maps,
        },
    ))
}

pub fn arr_to_out<'py>(
    py: Python<'py>,
    arr: &Array2<f64>,
    out: OUT,
    enc_info: Option<&EncodingInfo>,
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
