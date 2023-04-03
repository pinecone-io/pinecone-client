use crate::client::grpc::DataplaneGrpcClient;
use crate::data_types::MetadataValue;
use crate::data_types::{QueryResult, UpsertResponse, Vector};
use crate::utils::errors::{PineconeClientError, PineconeResult};
use std::collections::{BTreeMap, HashMap};

use crate::data_types::{IndexStats, SparseValues};

#[derive(Clone)]
pub struct Index {
    pub name: String,
    dataplane_client: DataplaneGrpcClient,
}

impl Index {
    pub fn new(index_name: String, dataplane_client: DataplaneGrpcClient) -> Self {
        Index {
            name: index_name,
            dataplane_client,
        }
    }

    /// The `Upsert` operation writes vectors into a namespace.
    /// If a new value is upserted for an existing vector id, it will overwrite the previous value.
    ///
    /// # Arguments
    /// - `namespace` - the name of the namespace to which data will be upserted
    /// - `vectors` - a list of vectors to be upserted to the index.
    ///
    /// # Returns
    /// `Ok(list_ids)` with a list of vector ids that were successfully upserted to the Index, or the underlying gRPC error on failure.

    pub async fn upsert(
        &mut self,
        namespace: &str,
        vectors: &[Vector],
        batch_size: Option<u32>,
    ) -> PineconeResult<UpsertResponse> {
        if batch_size.is_some() {
            todo!("Add proper upsert batching")
        }

        let upserted_count = self.dataplane_client.upsert(namespace, vectors).await?;

        if upserted_count != vectors.len() as u32 {
            return Err(PineconeClientError::Other(format!(
                "Failed to upsert all vectors. Upserted {} out of {} vectors",
                upserted_count,
                vectors.len()
            )));
        }

        Ok(UpsertResponse { upserted_count })
    }

    /// Query
    ///
    /// The `Query` operation searches a namespace, using a query vector.
    /// It retrieves the ids of the most similar items in a namespace, along with their similarity scores.
    /// To query by the id of already upserted vector, use `Index.query_by_id()`
    ///
    /// # Arguments
    /// - `namespace` - the name of the namespace in which vectors will be queried
    /// - `values` - The values for a new, unseen query vector. This should be the same length as the dimension of the index being queried. The results will be the `top_k` vectors closest to the given vector. Can not be used together with `id`
    /// - `sparse_values` - The query vector's sparse values.
    /// - `top_k` - The number of results to return for each query.
    /// - `filter` - The filter to apply. You can use vector metadata to limit your search. See <https://www.pinecone.io/docs/metadata-filtering/`>
    /// - `include_values` - Indicates whether vector values are included in the response.
    /// - `include_metadata` - Indicates whether metadata is included in the response as well as the ids.
    ///
    /// # Returns
    /// A list of QueryResults
    #[allow(clippy::too_many_arguments)]
    pub async fn query(
        &mut self,
        namespace: &str,
        values: Option<Vec<f32>>,
        sparse_values: Option<SparseValues>,
        top_k: u32,
        filter: Option<BTreeMap<String, MetadataValue>>,
        include_values: bool,
        include_metadata: bool,
    ) -> PineconeResult<Vec<QueryResult>> {
        let res = self
            .dataplane_client
            .query(
                namespace,
                None,
                values,
                sparse_values,
                top_k,
                filter,
                include_values,
                include_metadata,
            )
            .await?;

        Ok(res)
    }

    /// Query by id
    ///
    /// The `Query by id` operation searches a namespace given the `id` of a vector already residing in the Index.
    /// It retrieves the ids of the most similar items in a namespace, along with their similarity scores.
    /// To query by new unseen vector use `Index.query()`
    ///
    /// # Arguments
    /// - `namespace` - the name of the namespace in which vectors will be queried
    /// - `id` - An id of a vector already upserted to the relevant namespace. The results will be the `top_k` nearest neighbours of the vector with the given id. Can not be used together with `values`.
    /// - `top_k` - The number of results to return for each query.
    /// - `filter` - The filter to apply. You can use vector metadata to limit your search. See <https://www.pinecone.io/docs/metadata-filtering/`>
    /// - `include_values` - Indicates whether vector values are included in the response.
    /// - `include_metadata` - Indicates whether metadata is included in the response as well as the ids.
    ///
    /// # Returns
    /// A list QueryResults
    pub async fn query_by_id(
        &mut self,
        namespace: &str,
        id: &str,
        top_k: u32,
        filter: Option<BTreeMap<String, MetadataValue>>,
        include_values: bool,
        include_metadata: bool,
    ) -> PineconeResult<Vec<QueryResult>> {
        let res = self
            .dataplane_client
            .query(
                namespace,
                Some(id.into()),
                None,
                None,
                top_k,
                filter,
                include_values,
                include_metadata,
            )
            .await?;

        Ok(res)
    }

    /// Describe index stats
    ///
    /// The DescribeIndexStats operation returns the number of vectors present in the index, for all the namespaces
    /// and the fullness of the index. Can also accept a filter to count the number of vectors matching the filter.
    ///
    /// # Arguments
    /// - `filter` - Optional filter to apply to the stats call. When applied, the stats only refer to matching vectors.
    ///
    /// # Returns
    /// A map of number of vectors per namespace, total vectors and the index fulness.
    pub async fn describe_index_stats(
        &mut self,
        filter: Option<BTreeMap<String, MetadataValue>>,
    ) -> PineconeResult<IndexStats> {
        let res = self.dataplane_client.describe_index_stats(filter).await?;
        Ok(res)
    }

    /// Fetch
    ///
    /// The Fetch operation retrieves the vectors with the given ids from the index.
    ///
    /// # Arguments
    /// - `namespace` - the name of the namespace in which vectors will be fetched
    /// - `ids` - A list of ids of vectors already upserted to the relevant namespace.
    ///
    pub async fn fetch(
        &mut self,
        namespace: &str,
        ids: &[String],
    ) -> PineconeResult<HashMap<String, Vector>> {
        let res = self.dataplane_client.fetch(namespace, ids).await?;
        Ok(res)
    }

    /// Update
    /// The update operation updates a single vector in the index.
    ///
    /// # Arguments
    /// - `id` - The id of the vector to be updated
    /// - `values` - Optional new values for the vector
    /// - `set_metadata` - Optional new metadata keys and values to be updated
    /// - `namespace` - The name of the namespace in which vectors will be updated
    ///
    pub async fn update(
        &mut self,
        id: &str,
        values: Option<&Vec<f32>>,
        sparse_values: Option<SparseValues>,
        set_metadata: Option<BTreeMap<String, MetadataValue>>,
        namespace: &str,
    ) -> PineconeResult<()> {
        self.dataplane_client
            .update(id, values, sparse_values, set_metadata, namespace)
            .await?;
        Ok(())
    }

    /// Delete
    /// The delete operation deletes a list of vectors from a given namespace.
    ///
    /// # Arguments
    /// - `ids` - ids of the vectors to be deleted
    /// - `namespace` - the name of the namespace in which vectors will be deleted
    ///
    pub async fn delete(&mut self, ids: Vec<String>, namespace: &str) -> PineconeResult<()> {
        self.dataplane_client
            .delete(Some(ids), namespace, None, false)
            .await?;
        Ok(())
    }

    /// Delete by filter
    /// The delete by filter operation deletes a list of vectors from a given namespace that match the filter.
    ///
    /// # Arguments
    /// - `filter` - filter to be applied to delete the vectors
    /// - `namespace` - the name of the namespace in which vectors will be deleted
    ///
    pub async fn delete_by_metadata(
        &mut self,
        filter: Option<BTreeMap<String, MetadataValue>>,
        namespace: &str,
    ) -> PineconeResult<()> {
        self.dataplane_client
            .delete(None, namespace, filter, false)
            .await?;
        Ok(())
    }

    /// Delete all
    /// The delete all operation deletes all the vectors from a given namespace.
    ///
    /// # Arguments
    /// - `namespace` - the name of the namespace in which vectors will be deleted
    ///
    pub async fn delete_all(&mut self, namespace: &str) -> PineconeResult<()> {
        self.dataplane_client
            .delete(None, namespace, None, true)
            .await?;
        Ok(())
    }
}
