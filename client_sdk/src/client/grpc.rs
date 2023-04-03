use self::dataplane_client::UpdateResponse;
pub use self::dataplane_client::{
    ScoredVector as GrpcScoredVector, SparseValues as GrpcSparseValues, Vector as GrpcVector,
};
use crate::data_types::{
    IndexStats, MetadataValue, NamespaceStats, QueryResult, SparseValues, Vector,
};
use crate::utils::conversions;
use crate::utils::errors::PineconeResult;
use dataplane_client::vector_service_client::VectorServiceClient;
use dataplane_client::{DescribeIndexStatsRequest, QueryRequest, UpsertRequest};
use std::collections::{BTreeMap, HashMap};
use tonic::metadata::Ascii;
use tonic::{
    metadata::MetadataValue as TonicMetadataVal, service::interceptor::InterceptedService,
    service::Interceptor, transport::Channel, Request, Status,
};

mod dataplane_client {
    tonic::include_proto!("_");
}

#[derive(Debug, Clone)]
pub struct DataplaneGrpcClient {
    inner: VectorServiceClient<InterceptedService<Channel, ApiKeyInterceptor>>,
}

impl DataplaneGrpcClient {
    // TODO: this method shouldn't be public or exposed to python
    pub async fn connect(
        index_endpoint_url: String,
        api_key: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let channel = Channel::from_shared(index_endpoint_url)?.connect().await?;
        let token: TonicMetadataVal<_> = api_key.parse()?;
        let add_api_key_interceptor = ApiKeyInterceptor { api_token: token };
        let inner = VectorServiceClient::with_interceptor(channel, add_api_key_interceptor);

        Ok(Self { inner })
    }

    pub async fn upsert(
        &mut self,
        namespace: &str,
        vectors: &[Vector],
    ) -> Result<u32, tonic::Status> {
        let grpc_vectors: Vec<GrpcVector> = vectors.iter().map(|v| v.clone().into()).collect();
        let res = self
            .inner
            .upsert(UpsertRequest {
                namespace: namespace.to_string(),
                vectors: grpc_vectors,
            })
            .await?;
        Ok(res.into_inner().upserted_count)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn query(
        &mut self,
        namespace: &str,
        id: Option<String>,
        values: Option<Vec<f32>>,
        sparse_values: Option<SparseValues>,
        top_k: u32,
        filter: Option<BTreeMap<String, MetadataValue>>,
        include_values: bool,
        include_metadata: bool,
    ) -> PineconeResult<Vec<QueryResult>> {
        let sparse_vectors = sparse_values.map(|sparse_vector| sparse_vector.into());
        let res = self
            .inner
            .query(QueryRequest {
                namespace: namespace.to_string(),
                id: id.unwrap_or_default(),
                vector: values.unwrap_or_default(),
                sparse_vector: sparse_vectors,
                top_k,
                filter: filter.map(conversions::hashmap_to_prost_struct),
                include_values,
                include_metadata,
                queries: Vec::default(), // Deprecated
            })
            .await?;

        res.into_inner()
            .matches
            .into_iter()
            .map(|sv| sv.try_into())
            .collect()
    }

    pub async fn describe_index_stats(
        &mut self,
        filter: Option<BTreeMap<String, MetadataValue>>,
    ) -> Result<IndexStats, tonic::Status> {
        let res = self
            .inner
            .describe_index_stats(DescribeIndexStatsRequest {
                filter: filter.map(conversions::hashmap_to_prost_struct),
            })
            .await?
            .into_inner();
        let ns_summaries = res.namespaces;
        let mut ns_map: HashMap<String, NamespaceStats> =
            HashMap::with_capacity(ns_summaries.len());
        for (ns_name, ns_summary) in ns_summaries {
            ns_map.insert(
                ns_name,
                NamespaceStats {
                    vector_count: ns_summary.vector_count,
                },
            );
        }
        let stats: IndexStats = IndexStats {
            namespaces: ns_map,
            total_vector_count: res.total_vector_count,
            index_fullness: res.index_fullness,
            dimension: res.dimension,
        };
        Ok(stats)
    }

    pub async fn fetch(
        &mut self,
        namespace: &str,
        ids: &[String],
    ) -> PineconeResult<HashMap<String, Vector>> {
        let res = self
            .inner
            .fetch(dataplane_client::FetchRequest {
                namespace: namespace.to_string(),
                ids: ids.to_owned(),
            })
            .await?;
        let fetch_response = res.into_inner();
        let vectors = fetch_response.vectors;
        let mut fetch_vectors: HashMap<String, Vector> = HashMap::with_capacity(vectors.len());
        for (id, vector) in vectors {
            fetch_vectors.insert(id, vector.try_into()?);
        }
        Ok(fetch_vectors)
    }

    pub async fn delete(
        &mut self,
        ids: Option<Vec<String>>,
        namespace: &str,
        filter: Option<BTreeMap<String, MetadataValue>>,
        delete_all: bool,
    ) -> Result<(), tonic::Status> {
        self.inner
            .delete(dataplane_client::DeleteRequest {
                namespace: namespace.into(),
                ids: ids.unwrap_or_default(),
                delete_all,
                filter: filter.map(conversions::hashmap_to_prost_struct),
            })
            .await?;
        Ok(())
    }

    pub async fn update(
        &mut self,
        id: &str,
        vector: Option<&Vec<f32>>,
        sparse_values: Option<SparseValues>,
        set_metadata: Option<BTreeMap<String, MetadataValue>>,
        namespace: &str,
    ) -> Result<UpdateResponse, tonic::Status> {
        let res = self
            .inner
            .update(dataplane_client::UpdateRequest {
                id: id.into(),
                values: match vector {
                    Some(vec) => vec.clone(),
                    None => Vec::new(),
                },
                sparse_values: sparse_values.map(|sparse_values| sparse_values.into()),
                set_metadata: set_metadata.map(conversions::hashmap_to_prost_struct),
                namespace: namespace.into(),
            })
            .await?;
        Ok(res.into_inner())
    }
}

#[derive(Debug, Clone)]
pub struct ApiKeyInterceptor {
    api_token: TonicMetadataVal<Ascii>,
}

impl Interceptor for ApiKeyInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // TODO: replace `api_token` with an `Option`, and do a proper `if_some`.
        if !self.api_token.is_empty() {
            request
                .metadata_mut()
                .insert("api-key", self.api_token.clone());
        }
        Ok(request)
    }
}

/// Get internal grpc client
/// This client would only work from within a pinecone region to the internal endpoint address/
/// It is meant to be used by internal services within the region that need to communicate with the Index GRPC API
/// TODO: this function shouldn't be exposed by the python client
pub async fn get_internal_grpc_client(
    index_endpoint_url: String,
) -> Result<DataplaneGrpcClient, Box<dyn std::error::Error>> {
    // TODO: Theoretically this method could have simply been one line:
    // VectorServiceClient::connect(index_endpoint_url).await?
    // But than the return type would be different, DataplaneGrpcClient would need to be Generic.
    // so TODO: Find a better way to expose an inner stateless, authentication-less, gRPC client

    let channel = Channel::from_shared(index_endpoint_url)?.connect().await?;
    let token: TonicMetadataVal<_> = "".parse()?;
    let add_api_key_interceptor = ApiKeyInterceptor { api_token: token };
    let inner = VectorServiceClient::with_interceptor(channel, add_api_key_interceptor);
    Ok(DataplaneGrpcClient { inner })
}

// todo: add better tests
#[cfg(test)]
mod tests {
    use crate::data_types::SparseValues;

    use super::DataplaneGrpcClient;
    const INDEX_ENDPOINT: &str = "";
    const KEY: &str = "";

    fn gen_random_dense_vectors(count: usize, dimension: i32) -> Vec<super::Vector> {
        let mut vectors = Vec::new();
        for i in 0..count {
            let values = vec![0.1; dimension as usize];

            vectors.push(super::Vector {
                id: i.to_string(),
                values,
                sparse_values: None,
                metadata: None,
            });
        }
        vectors
    }

    fn gen_random_mixed_vectors(count: usize, dimension: i32) -> Vec<super::Vector> {
        let mut vectors = Vec::new();
        for i in 0..count {
            let values = vec![0.1; dimension as usize];
            let sparse_values = SparseValues {
                indices: vec![0; dimension as usize],
                values: vec![0.1; dimension as usize],
            };
            vectors.push(super::Vector {
                id: i.to_string(),
                values,
                sparse_values: Some(sparse_values),
                metadata: None,
            });
        }
        vectors
    }

    #[tokio::test]
    async fn test_upsert() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let vectors = gen_random_dense_vectors(10, 1024);
        let res = client.upsert("ns", &vectors).await;
        assert!(res.unwrap() == 10)
    }

    #[tokio::test]
    async fn test_stats() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client.describe_index_stats(None).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_fetch() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client.fetch("ns", &["1".to_string()]).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_delete() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client
            .delete(Some(vec![("2".to_string())]), "ns", None, false)
            .await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_delete_all() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client.delete(None, "ns", None, true).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_update() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client
            .update("1", Some(&vec![0.4; 128]), None, None, "ns")
            .await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_mixed_upsert() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let vectors = gen_random_mixed_vectors(10, 128);
        let res = client.upsert("ns", &vectors).await;
        assert!(res.unwrap() == 10)
    }

    #[tokio::test]
    async fn test_fetch_non_existent() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client.fetch("ns", &["100".to_string()]).await;
        assert!(res.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_delete_non_existent() {
        let mut client = DataplaneGrpcClient::connect(INDEX_ENDPOINT.to_string(), KEY)
            .await
            .unwrap();
        let res = client
            .delete(Some(vec!["100".to_string()]), "ns", None, false)
            .await;
        assert!(res.is_ok());
    }
}
