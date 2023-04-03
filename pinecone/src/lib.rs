// pub extern crate client_sdk;
//

use pyo3::prelude::*;

pub mod client;
pub mod data_types;
pub mod index;
pub mod utils;

use crate::index::Index;
use client::Client;
use client_sdk::data_types as core_data_types;
use utils::errors;

#[pymodule]
fn pinecone(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<core_data_types::Vector>()?;
    m.add_class::<core_data_types::SparseValues>()?;
    m.add_class::<core_data_types::QueryResult>()?;
    m.add_class::<core_data_types::NamespaceStats>()?;
    m.add_class::<core_data_types::IndexStats>()?;
    m.add(
        "PineconeOpError",
        <errors::PineconeOpError as pyo3::PyTypeInfo>::type_object(_py),
    )?;
    m.add_class::<Index>()?;
    Ok(())
}
