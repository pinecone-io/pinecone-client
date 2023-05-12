use pyo3::types::PyDict;
use pyo3::{FromPyObject, PyAny};
use std::collections::BTreeMap;

use crate::utils::errors::{PineconeClientError, PineconeResult};
use client_sdk::data_types as core_data_types;
use client_sdk::utils::errors::PineconeClientError as core_error;

#[derive(FromPyObject, Debug, Clone)]
pub enum UpsertRecord<'a> {
    Vector(core_data_types::Vector),
    TwoTuple((String, Vec<f32>)),
    ThreeTuple(
        (
            String,
            Vec<f32>,
            BTreeMap<String, core_data_types::MetadataValue>,
        ),
    ),
    Dict(&'a PyDict),
    #[pyo3(transparent)]
    Other(&'a PyAny), // This extraction never fails
}

pub fn convert_upsert_enum_to_vectors(
    vectors: Vec<UpsertRecord>,
) -> PineconeResult<Vec<core_data_types::Vector>> {
    let vectors_to_upsert: Vec<core_data_types::Vector> = vectors.into_iter().enumerate().map(|(i, vec)| {
            let new_vec: PineconeResult<core_data_types::Vector> = match vec.to_owned() {
                UpsertRecord::Vector(v) => Ok(v),
                UpsertRecord::TwoTuple(t) => Ok(core_data_types::Vector{ id: t.0, values: t.1 , ..Default::default()}),
                UpsertRecord::ThreeTuple(t) => Ok(core_data_types::Vector{ id: t.0, values: t.1 , metadata: Some(t.2),  ..Default::default()}),
                UpsertRecord::Dict(d) => Ok(
                    d.try_into()
                        .map_err(|e| match e{
                            core_error::UpsertKeyError { key, vec_num: _ } =>
                                core_error::UpsertKeyError {key, vec_num: i},
                            core_error::UpsertValueError { key, vec_num: _, actual, expected_type} =>
                                core_error::UpsertValueError {key, vec_num: i, actual, expected_type},
                            _ => core_error::ValueError(format!("Error in vector number {i}: {e}", i=i, e=e))
                        })?
                ),
                // TODO: add a dedicated error type, then format this error message in pinecone (the error message is pythonic)
                UpsertRecord::Other(val) => Err(PineconeClientError::from(
                    core_error::ValueError(format!("Error in vector number {i}: Found unexpected value: {val}.\n\
                    Allowed types are: Vector; Tuple[str, List[float]]; Tuple[str, List[float], dict]; Dict[str, Any]", i=i, val=val))
                ))

            };
            new_vec

        }).collect::<Result<Vec<_>, _>>()?;
    Ok(vectors_to_upsert)
}
