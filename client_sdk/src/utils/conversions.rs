use crate::client::grpc::{GrpcScoredVector, GrpcSparseValues, GrpcVector};
use crate::data_types::{Collection, Db, MetadataValue, QueryResult, SparseValues, Vector};
use crate::utils::errors::PineconeClientError::{MetadataError, MetadataValueError};
use crate::utils::errors::{PineconeClientError, PineconeResult};
use index_service::models::IndexMetaStatus;
use index_service::models::{
    CollectionMeta, CreateCollectionRequest, CreateRequest, CreateRequestMetadataConfig, IndexMeta,
};
use prost_types::value::Kind;
use prost_types::{ListValue as ProstListValue, Struct, Value as ProstValue};
use std::collections::BTreeMap;

impl From<SparseValues> for GrpcSparseValues {
    fn from(value: SparseValues) -> Self {
        GrpcSparseValues {
            indices: value.indices,
            values: value.values,
        }
    }
}

impl From<GrpcSparseValues> for SparseValues {
    fn from(value: GrpcSparseValues) -> Self {
        SparseValues {
            indices: value.indices,
            values: value.values,
        }
    }
}

impl TryFrom<ProstValue> for MetadataValue {
    type Error = PineconeClientError;

    fn try_from(val: ProstValue) -> Result<Self, Self::Error> {
        if let Some(kind) = val.kind {
            match kind {
                Kind::NumberValue(v) => Ok(MetadataValue::NumberVal(v)),
                Kind::StringValue(v) => Ok(MetadataValue::StringVal(v)),
                Kind::BoolValue(v) => Ok(MetadataValue::BoolVal(v)),
                Kind::ListValue(v) => {
                    let mut inners = Vec::new();
                    for item in v.values.into_iter() {
                        let new_val = item.try_into().map_err(|e| match e {
                            MetadataValueError { val_type } => MetadataValueError {
                                val_type: format!("{val_type} value in a list"),
                            },
                            _ => e,
                        })?;
                        inners.push(new_val);
                    }
                    Ok(MetadataValue::ListVal(inners))
                }
                Kind::NullValue(_) => Err(MetadataValueError {
                    val_type: "None".into(),
                }),
                Kind::StructValue(s) => {
                    let mut inners = BTreeMap::new();
                    for (k, v) in s.fields {
                        let new_val = v.try_into().map_err(|e| match e {
                            MetadataValueError { val_type } => MetadataValueError {
                                val_type: format!("{val_type} value in a dict"),
                            },
                            MetadataError { key, val_type } => MetadataError {
                                key: format!("{k}: {key}"),
                                val_type: format!("{val_type} value in a dict"),
                            },
                            _ => e,
                        })?;
                        inners.insert(k, new_val);
                    }
                    Ok(MetadataValue::DictVal(inners))
                }
            }
        } else {
            Err(MetadataValueError {
                val_type: "empty".into(),
            })
        }
    }
}

impl From<MetadataValue> for ProstValue {
    fn from(val: MetadataValue) -> Self {
        match val {
            MetadataValue::StringVal(v) => ProstValue {
                kind: Some(Kind::StringValue(v)),
            },
            MetadataValue::NumberVal(v) => ProstValue {
                kind: Some(Kind::NumberValue(v)),
            },
            MetadataValue::BoolVal(v) => ProstValue {
                kind: Some(Kind::BoolValue(v)),
            },
            MetadataValue::ListVal(v) => {
                let new_list = v.into_iter().map(|x| x.into()).collect();
                ProstValue {
                    kind: Some(Kind::ListValue(ProstListValue { values: new_list })),
                }
            }
            MetadataValue::DictVal(v) => {
                let mut new_dict: BTreeMap<String, ProstValue> = BTreeMap::new();
                for (k, v) in v.into_iter() {
                    new_dict.insert(k, v.into());
                }
                let new_struct = Struct { fields: new_dict };
                ProstValue {
                    kind: Some(Kind::StructValue(new_struct)),
                }
            }
        }
    }
}

impl From<Db> for CreateRequest {
    fn from(index: Db) -> Self {
        CreateRequest {
            name: index.name,
            dimension: index.dimension,
            replicas: index.replicas,
            pod_type: index.pod_type,
            metric: index.metric,
            pods: index.pods,
            shards: index.shards,
            source_collection: index.source_collection,
            metadata_config: index.metadata_config.map(|config| {
                Some(Box::new(CreateRequestMetadataConfig {
                    indexed: config.get("indexed").map(|v| v.to_vec()),
                }))
            }),
            ..Default::default()
        }
    }
}

impl TryFrom<IndexMeta> for Db {
    type Error = PineconeClientError;
    fn try_from(index_meta: IndexMeta) -> Result<Self, Self::Error> {
        let db = index_meta.database;
        let status = index_meta.status;
        let state = status.and_then(|inner_box| {
            let inner_struct: IndexMetaStatus = *inner_box;
            inner_struct.state
        });
        match db {
            Some(db) => {
                let name = db.name.ok_or_else(|| {
                    PineconeClientError::ValueError("Failed to parse db name".to_string())
                })?;
                let replicas = db.replicas;
                let shards = db.shards;
                let pod_type = db.pod_type;
                let dimension = db.dimension.ok_or_else(|| {
                    PineconeClientError::Other("Failed to parse db dimension".to_string())
                })?;
                let metric = db.metric;
                let pods = db.pods;
                let source_collection = db.source_collection;
                let metadata_config = db.metadata_config.map(|config| {
                    let indexed = config.indexed.unwrap_or_default();
                    let mut map = BTreeMap::new();
                    map.insert("indexed".to_string(), indexed);
                    map
                });
                let status = state;
                Ok(Db {
                    name,
                    dimension,
                    replicas,
                    shards,
                    pod_type,
                    metric,
                    pods,
                    source_collection,
                    metadata_config,
                    status,
                })
            }
            None => Err(PineconeClientError::Other("Failed to parse db".to_string())),
        }
    }
}

impl From<Collection> for CreateCollectionRequest {
    fn from(collection: Collection) -> Self {
        CreateCollectionRequest {
            name: collection.name,
            source: collection.source,
        }
    }
}

impl From<CollectionMeta> for Collection {
    fn from(collection_meta: CollectionMeta) -> Self {
        Collection {
            name: collection_meta.name.unwrap(),
            source: "".to_string(),
            vector_count: None,
            size: collection_meta.size,
            status: collection_meta.status,
        }
    }
}

pub fn hashmap_to_prost_struct(dict: BTreeMap<String, MetadataValue>) -> Struct {
    let mut fields = BTreeMap::new();
    for (k, v) in dict.into_iter() {
        fields.insert(k, v.into());
    }
    Struct { fields }
}

pub fn prost_struct_to_hashmap(dict: Struct) -> PineconeResult<BTreeMap<String, MetadataValue>> {
    let mut fields: BTreeMap<String, MetadataValue> = BTreeMap::new();
    for (k, v) in dict.fields.into_iter() {
        let new_val = v.try_into().map_err(|e| match e {
            PineconeClientError::MetadataValueError { val_type } => {
                PineconeClientError::MetadataError {
                    key: k.clone(),
                    val_type,
                }
            }
            _ => e,
        })?;
        fields.insert(k, new_val);
    }
    Ok(fields)
}

impl From<Vector> for GrpcVector {
    fn from(grpc_vector: Vector) -> Self {
        GrpcVector {
            id: grpc_vector.id,
            values: grpc_vector.values,
            sparse_values: grpc_vector
                .sparse_values
                .map(|sparse_vector| sparse_vector.into()),
            metadata: grpc_vector.metadata.map(hashmap_to_prost_struct),
        }
    }
}

impl TryFrom<GrpcVector> for Vector {
    type Error = PineconeClientError;

    fn try_from(grpc_vector: GrpcVector) -> Result<Self, Self::Error> {
        Ok(Vector {
            id: grpc_vector.id,
            values: grpc_vector.values,
            sparse_values: grpc_vector
                .sparse_values
                .map(|sparse_vector| sparse_vector.into()),
            metadata: grpc_vector
                .metadata
                .map(prost_struct_to_hashmap)
                .transpose()?,
        })
    }
}

impl TryFrom<GrpcScoredVector> for QueryResult {
    type Error = PineconeClientError;

    fn try_from(grpc_vector: GrpcScoredVector) -> Result<Self, Self::Error> {
        Ok(QueryResult {
            id: grpc_vector.id,
            score: grpc_vector.score,
            values: if grpc_vector.values.is_empty() {
                None
            } else {
                Some(grpc_vector.values)
            },
            sparse_values: grpc_vector
                .sparse_values
                .map(|sparse_vector| sparse_vector.into()),
            metadata: grpc_vector
                .metadata
                .map(prost_struct_to_hashmap)
                .transpose()?,
        })
    }
}
