use crate::data_types::{MetadataValue, NamespaceStats, SparseValues, Vector};
use crate::utils::errors::PineconeClientError;
use pyo3::types::{IntoPyDict, PyDict};
use pyo3::{IntoPy, PyObject, Python, ToPyObject};
use std::collections::{BTreeMap, HashSet};

const SPARSE_KEYS: &[&str] = &["indices", "values"];
const VECTOR_KEYS: &[&str] = &["id", "values", "sparse_values", "metadata"];

impl TryFrom<&PyDict> for SparseValues {
    type Error = PineconeClientError;

    fn try_from(dict: &PyDict) -> Result<Self, Self::Error> {
        let allowed_keys: HashSet<String> = SPARSE_KEYS.iter().map(|x| (*x).into()).collect();
        let actual_keys: HashSet<String> = dict
            .keys()
            .into_iter()
            .map(|x| x.extract::<String>())
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|_| {
                PineconeClientError::ValueError("Couldn't retrieve dictionary keys".into())
            })?;

        let excess_keys = actual_keys
            .difference(&allowed_keys)
            .collect::<Vec<&String>>();
        if !excess_keys.is_empty() {
            return Err(PineconeClientError::ValueError(format!(
                "Found unexpected keys: {excess_keys:?}",
                excess_keys = excess_keys
            )));
        }

        let indices = match dict.get_item("indices") {
            None => {
                return Err(PineconeClientError::UpsertKeyError {
                    key: "indices".into(),
                    vec_num: 0,
                })
            }
            Some(v) => {
                v.extract::<Vec<u32>>()
                    .map_err(|_| PineconeClientError::UpsertValueError {
                        key: "indices".into(),
                        vec_num: 0,
                        expected_type: "List[int]".into(),
                        actual: format!("{:?}", v),
                    })?
            }
        };

        let values = match dict.get_item("values") {
            None => {
                return Err(PineconeClientError::UpsertKeyError {
                    key: "values".into(),
                    vec_num: 0,
                })
            }
            Some(v) => {
                v.extract::<Vec<f32>>()
                    .map_err(|_| PineconeClientError::UpsertValueError {
                        key: "values".into(),
                        vec_num: 0,
                        expected_type: "List[float]".into(),
                        actual: format!("{:?}", v),
                    })?
            }
        };

        Ok(SparseValues { indices, values })
    }
}

impl ToPyObject for SparseValues {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = [
            ("indices", self.indices.to_object(py)),
            ("values", self.values.to_object(py)),
        ]
        .into_py_dict(py);
        dict.to_object(py)
    }
}

impl TryFrom<&PyDict> for Vector {
    type Error = PineconeClientError;

    fn try_from(dict: &PyDict) -> Result<Self, Self::Error> {
        let allowed_keys: HashSet<String> = VECTOR_KEYS.iter().map(|x| (*x).into()).collect();
        let actual_keys: HashSet<String> = dict
            .keys()
            .into_iter()
            .map(|x| x.extract::<String>())
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|_| {
                PineconeClientError::ValueError("Couldn't retrieve dictionary keys".into())
            })?;

        let excess_keys = actual_keys
            .difference(&allowed_keys)
            .collect::<Vec<&String>>();
        if !excess_keys.is_empty() {
            return Err(PineconeClientError::ValueError(format!(
                "Found unexpected keys: {excess_keys:?}",
                excess_keys = excess_keys
            )));
        }

        Ok(Vector {
            id: match dict.get_item("id") {
                None => {
                    return Err(PineconeClientError::UpsertKeyError {
                        key: "id".into(),
                        vec_num: 0,
                    })
                }
                Some(id) => {
                    id.extract::<String>()
                        .map_err(|_| PineconeClientError::UpsertValueError {
                            key: "id".into(),
                            vec_num: 0,
                            expected_type: "String".into(),
                            actual: format!("{:?}", id),
                        })
                }
            }?,
            values: match dict.get_item("values") {
                None => {
                    return Err(PineconeClientError::UpsertKeyError {
                        key: "values".into(),
                        vec_num: 0,
                    })
                }
                Some(values) => values.extract::<Vec<f32>>().map_err(|_| {
                    PineconeClientError::UpsertValueError {
                        key: "values".into(),
                        vec_num: 0,
                        expected_type: "List[float]".into(),
                        actual: format!("{:?}", values),
                    }
                })?,
            },
            sparse_values: dict
                .get_item("sparse_values")
                .map(|val| {
                    let val = val.extract::<&PyDict>().map_err(|_| {
                        PineconeClientError::UpsertValueError {
                            key: "sparse_values".into(),
                            vec_num: 0,
                            expected_type: "dict".into(),
                            actual: format!("{:?}", val),
                        }
                    })?;
                    val.try_into().map_err(|e| match e {
                        PineconeClientError::UpsertKeyError { key, vec_num } => {
                            PineconeClientError::UpsertKeyError {
                                key: format!("sparse_values: {key}", key = key),
                                vec_num,
                            }
                        }
                        PineconeClientError::UpsertValueError {
                            key,
                            vec_num,
                            actual,
                            expected_type,
                        } => PineconeClientError::UpsertValueError {
                            key: format!("sparse_values: {key}", key = key),
                            vec_num,
                            actual,
                            expected_type,
                        },
                        _ => PineconeClientError::ValueError(format!(
                            "Error in 'sparse_values: {e}",
                            e = e
                        )),
                    })
                })
                .transpose()?,
            metadata: dict
                .get_item("metadata")
                .map(|val| {
                    val.extract::<BTreeMap<String, MetadataValue>>()
                        .map_err(|_| PineconeClientError::UpsertValueError {
                            key: "metadata".into(),
                            vec_num: 0,
                            expected_type: "dict".into(),
                            actual: format!("{:?}", val),
                        })
                })
                .transpose()?,
        })
    }
}

impl ToPyObject for NamespaceStats {
    fn to_object(&self, py: Python) -> PyObject {
        self.to_dict(py).to_object(py)
    }
}

impl ToPyObject for MetadataValue {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            MetadataValue::StringVal(v) => v.to_object(py),
            MetadataValue::NumberVal(v) => v.to_object(py),
            MetadataValue::BoolVal(v) => v.to_object(py),
            MetadataValue::ListVal(v) => v.to_object(py),
            MetadataValue::DictVal(v) => v.to_object(py),
        }
    }
}

impl IntoPy<PyObject> for MetadataValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            MetadataValue::StringVal(v) => v.to_object(py),
            MetadataValue::NumberVal(v) => v.to_object(py),
            MetadataValue::ListVal(v) => v.to_object(py),
            MetadataValue::BoolVal(v) => v.to_object(py),
            MetadataValue::DictVal(v) => v.to_object(py),
        }
    }
}
