use pyo3::{exceptions::PyTypeError, prelude::*};
use std::fmt;

#[derive(Debug)]
pub enum Errors {
    UnsupportedType {
        unsupported: String,
        supported: Option<String>,
    },
}

impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Errors::UnsupportedType {
                unsupported,
                supported,
            } => {
                write!(f, "Unsupported Type: {}", unsupported)?;
                if let Some(values) = supported {
                    write!(f, "\nSupported Types are: {}", values)?
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for Errors {}

impl From<Errors> for PyErr {
    fn from(err: Errors) -> PyErr {
        match err {
            Errors::UnsupportedType { .. } => PyTypeError::new_err(err.to_string()),
        }
    }
}
