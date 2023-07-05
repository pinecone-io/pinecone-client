use client_sdk::utils::errors as core_errors;
use pyo3::create_exception;
use pyo3::exceptions;
use pyo3::prelude::*;

create_exception!(
    pinecone_client,
    PineconeOpError,
    pyo3::exceptions::PyException
);

pub struct PineconeClientError {
    inner: core_errors::PineconeClientError,
}

impl From<core_errors::PineconeClientError> for PineconeClientError {
    fn from(error: core_errors::PineconeClientError) -> PineconeClientError {
        PineconeClientError { inner: error }
    }
}

impl From<PineconeClientError> for PyErr {
    fn from(err: PineconeClientError) -> PyErr {
        match err.inner {
            core_errors::PineconeClientError::ArgumentError { .. } => {
                exceptions::PyValueError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::ControlConnectionError { .. } => {
                exceptions::PyConnectionError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::IndexConnectionError { .. } => {
                exceptions::PyConnectionError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::DataplaneOperationError(_) => {
                PineconeOpError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::IoError(_) => {
                exceptions::PyIOError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::MetadataValueError { .. } => {
                exceptions::PyValueError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::MetadataError { .. } => {
                exceptions::PyValueError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::Other(_) => {
                exceptions::PyRuntimeError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::ControlPlaneOperationError { .. } => {
                PineconeOpError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::ControlPlaneParsingError { .. } => {
                PineconeOpError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::DeserializationError(_) => {
                PineconeOpError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::ValueError(_) => {
                exceptions::PyValueError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::UpsertValueError { .. } => {
                exceptions::PyValueError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::UpsertKeyError { .. } => {
                exceptions::PyValueError::new_err(err.inner.to_string())
            }
            core_errors::PineconeClientError::KeyboardInterrupt(_) => {
                exceptions::PyKeyboardInterrupt::new_err(err.inner.to_string())
            }
        }
    }
}

pub type PineconeResult<T> = Result<T, PineconeClientError>;
