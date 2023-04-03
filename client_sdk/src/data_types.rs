use derivative::Derivative;

use pyo3::types::{PyDict, PyList};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::vec::Vec;

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

const SHORT_PRINT_LEN: usize = 5;

#[derive(Debug, Default, Clone)]
#[pyclass]
#[pyo3(get_all)]
#[pyo3(text_signature = "(indices, values)")]
pub struct SparseValues {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[pymethods]
impl SparseValues {
    #[new]
    #[pyo3(signature = (indices, values))]
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self { indices, values }
    }

    pub fn __repr__(&self) -> Result<String, PyErr> {
        Ok(format!(
            "SparseValue:\n  indices: {indices:?}...\n  values: {values:?} ...",
            indices = &self.indices.chunks(5).next().unwrap_or(&Vec::<u32>::new()),
            values = &self.values.chunks(5).next().unwrap_or(&Vec::<f32>::new())
        ))
    }
}

#[derive(Debug, Default, Clone)]
#[pyclass]
#[pyo3(get_all)]
#[pyo3(text_signature = "(id, values, sparse_values=None, metadata=None)")]
pub struct Vector {
    pub id: String,
    pub values: Vec<f32>,
    pub sparse_values: Option<SparseValues>,
    pub metadata: Option<BTreeMap<String, MetadataValue>>,
}

#[pymethods]
impl Vector {
    #[new]
    #[pyo3(signature = (id, values, sparse_values=None, metadata=None))]
    pub fn new(
        id: String,
        values: Vec<f32>,
        sparse_values: Option<SparseValues>,
        metadata: Option<BTreeMap<String, MetadataValue>>,
    ) -> Self {
        Self {
            id,
            values,
            sparse_values,
            metadata,
        }
    }

    pub fn __repr__(&self, py: Python) -> Result<String, PyErr> {
        Ok("Vector:\n".to_string() + pretty_print_dict(self.to_dict(py), 2)?.as_str())
    }

    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("id", self.id.to_object(py)),
            ("values", self.values.to_object(py)),
            ("sparse_values", self.sparse_values.to_object(py)),
            ("metadata", self.metadata.to_object(py)),
        ];
        key_vals.into_py_dict(py)
    }
}

#[derive(Debug)]
#[pyclass]
#[pyo3(get_all)]
pub struct UpsertResponse {
    pub upserted_count: u32,
}

#[pymethods]
impl UpsertResponse {
    pub fn __repr__(&self, py: Python) -> Result<String, PyErr> {
        Ok("UpsertResponse:\n".to_string() + pretty_print_dict(self.to_dict(py), 2)?.as_str())
    }

    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> =
            vec![("upserted_count", self.upserted_count.to_object(py))];
        key_vals.into_py_dict(py)
    }
}

#[derive(Debug)]
#[pyclass]
#[pyo3(get_all, mapping)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    pub values: Option<Vec<f32>>,
    pub sparse_values: Option<SparseValues>,
    pub metadata: Option<BTreeMap<String, MetadataValue>>,
}

#[pymethods]
impl QueryResult {
    pub fn __repr__(&self, py: Python) -> Result<String, PyErr> {
        Ok("QueryResult:\n".to_string() + pretty_print_dict(self.to_dict(py), 2)?.as_str())
    }

    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("id", self.id.to_object(py)),
            ("score", self.score.to_object(py)),
            ("values", self.values.to_object(py)),
            ("sparse_values", self.sparse_values.to_object(py)),
            ("metadata", self.metadata.to_object(py)),
        ];
        key_vals.into_py_dict(py)
    }
}

#[derive(Deserialize, Debug)]
pub struct WhoamiResponse {
    pub project_name: String,
    pub user_label: String,
    pub user_name: String,
}

#[derive(Deserialize, Debug, Clone)]
#[pyclass]
#[pyo3(get_all)]
pub struct NamespaceStats {
    pub vector_count: u32,
}

#[pymethods]
impl NamespaceStats {
    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> =
            vec![("vector_count", self.vector_count.to_object(py))];
        key_vals.into_py_dict(py)
    }
}

#[derive(Deserialize, Debug)]
#[pyclass]
#[pyo3(get_all)]
pub struct IndexStats {
    pub namespaces: HashMap<String, NamespaceStats>,
    pub dimension: u32,
    pub index_fullness: f32,
    pub total_vector_count: u32,
}

#[pymethods]
impl IndexStats {
    pub fn __repr__(&self, py: Python) -> Result<String, PyErr> {
        Ok("Index statistics:\n".to_string() + pretty_print_dict(self.to_dict(py), 2)?.as_str())
    }

    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("namespaces", self.namespaces.to_object(py)),
            ("dimension", self.dimension.to_object(py)),
            ("index_fullness", self.index_fullness.to_object(py)),
            ("total_vector_count", self.total_vector_count.to_object(py)),
        ];
        key_vals.into_py_dict(py)
    }
}

#[derive(FromPyObject, Debug, Clone)]
pub enum MetadataValue {
    StringVal(String),
    BoolVal(bool),
    NumberVal(f64),
    ListVal(Vec<MetadataValue>),
    DictVal(BTreeMap<String, MetadataValue>),
}

#[derive(Derivative, Default, Debug, Clone)]
#[pyclass]
#[pyo3(get_all, mapping)]
pub struct Db {
    pub name: String,
    pub dimension: i32,
    pub metric: Option<String>,
    pub replicas: Option<i32>,
    pub shards: Option<i32>,
    pub pods: Option<i32>,
    pub source_collection: Option<String>,
    pub metadata_config: Option<BTreeMap<String, Vec<String>>>,
    pub pod_type: Option<String>,
    pub status: Option<String>,
}

#[derive(Derivative, Default, Debug, Clone)]
#[pyclass]
#[pyo3(get_all, mapping)]
pub struct Collection {
    pub name: String,
    pub source: String,
    pub vector_count: Option<i32>,
    pub size: Option<i32>,
    pub status: Option<String>,
}

#[pymethods]
impl Db {
    pub fn __repr__(&self, py: Python) -> Result<String, PyErr> {
        Ok("Index config:\n".to_string() + pretty_print_dict(self.to_dict(py), 2)?.as_str())
    }

    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("name", self.name.to_object(py)),
            ("dimension", self.dimension.to_object(py)),
            ("replicas", self.replicas.to_object(py)),
            ("shards", self.shards.to_object(py)),
            ("pod_type", self.pod_type.to_object(py)),
            ("metric", self.metric.to_object(py)),
            ("pods", self.pods.to_object(py)),
            ("source_collection", self.source_collection.to_object(py)),
            ("metadata_config", self.metadata_config.to_object(py)),
            ("status", self.status.to_object(py)),
        ];
        key_vals.into_py_dict(py)
    }
}

#[pymethods]
impl Collection {
    pub fn __repr__(&self, py: Python) -> Result<String, PyErr> {
        Ok("Collection:\n".to_string() + pretty_print_dict(self.to_dict(py), 2)?.as_str())
    }

    pub fn to_dict<'a>(&self, py: Python<'a>) -> &'a PyDict {
        let key_vals: Vec<(&str, PyObject)> = vec![
            ("name", self.name.to_object(py)),
            ("source", self.source.to_object(py)),
            ("vector_count", self.vector_count.to_object(py)),
            ("size", self.size.to_object(py)),
            ("status", self.status.to_object(py)),
        ];
        key_vals.into_py_dict(py)
    }
}

fn pretty_print_dict(dict: &PyDict, indent: usize) -> Result<String, PyErr> {
    let mut msg = String::new();
    for (k, v) in dict.into_iter() {
        if let Ok(dict_val) = v.downcast::<PyDict>() {
            let inner_msg = pretty_print_dict(dict_val, indent + 2)?;
            msg += format!(
                "{:indent$}{key}:\n{val}",
                "",
                key = k,
                val = inner_msg,
                indent = indent
            )
            .as_str();
        } else if let Ok(list_val) = v.downcast::<PyList>() {
            let short_list = list_val.as_sequence().get_slice(0, SHORT_PRINT_LEN)?;
            let ellipsis = if list_val.len() > SHORT_PRINT_LEN {
                "..."
            } else {
                ""
            };
            msg += format!(
                "{:indent$}{key}: {short_list:.3}{elipsis}\n",
                "",
                key = k,
                short_list = short_list,
                elipsis = ellipsis,
                indent = indent
            )
            .as_str();
        } else {
            msg += format!(
                "{:indent$}{key}: {val:.3}\n",
                "",
                key = k,
                val = v,
                indent = indent
            )
            .as_str();
        }
    }
    Ok(msg)
}
