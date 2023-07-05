use crate::data_types::convert_upsert_enum_to_vectors;
use crate::data_types::UpsertRecord;
use crate::utils::errors::{PineconeClientError, PineconeResult};
use client_sdk::data_types as core_data_types;
use client_sdk::index as core_index;
use client_sdk::utils::errors::PineconeClientError as core_error;
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};
use tokio::runtime::Handle;

#[pyclass]
pub struct Index {
    inner: core_index::Index,
    runtime: Handle,
}

impl Index {
    pub fn new(inner: core_index::Index, runtime: Handle) -> Self {
        Self { inner, runtime }
    }
}

#[pymethods]
impl Index {
    pub fn __repr__(&self) -> String {
        format!("Index: \"{name}\"", name = self.inner.name)
    }

    #[pyo3(signature = (vectors, namespace="", async_req=false))]
    #[pyo3(text_signature = "(vectors, namespace='', async_req=False)")]
    /// The `Upsert` operation writes vectors into a namespace.
    /// If a new value is upserted for an existing vector id, it will overwrite the previous value.
    ///
    /// Args:
    ///     vectors (Union[List[Tuple[str, List[float]]], List[Dict[str, Any]], List[Vector]]): A list of vectors to upsert.
    ///         A vector can be represented by:
    ///         - A `Vector` object.
    ///         - A tuple of the form (id: str, vector: List[float]) or (id: str, vector: List[float], metadata: Dict[str, Union[str, float, int, bool, List[str]]]])
    ///         - A dictionary with the keys 'id' (str), 'values' (List[float]), 'sparse_values' (optional dict in the format {'indices': List[int], 'values': List[float]}), 'metadata' (Optional[Dict[str, Any]])
    ///         Note: sparse values are not supported when using a tuple. Please use a dictionary or a `Vector` object instead.
    ///
    ///     namespace (Optional[str]): Optional namespace to which data will be upserted.
    ///     async_req (bool): When set to True, the upsert request will be performed asynchronously, and a "future" (asyncio coroutine) will be returned.
    ///
    /// Examples:
    ///     ```python
    ///     index.upsert([ Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
    ///                    Vector(id='id3', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4])) ])
    ///
    ///     index.upsert([ ('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
    ///                    ('id2', [1.0, 2.0, 3.0]) ])
    ///
    ///     index.upsert([ {'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
    ///                        {'id': 'id2', 'values': [1.0, 2.0, 3.0], 'sparse_values': {'indices': [1, 2], 'values': [0.2, 0.4]}} ])
    ///
    ///     # Mixing different vector representations is also allowed
    ///     index.upsert([ {'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}, 'sparse_values': {'indices': [1, 2], 'values': [0.2, 0.4]}},
    ///                    ('id2', [1.0, 2.0, 3.0]), ])
    ///     ```
    ///
    /// Returns:
    ///     - If `async_req=False`:
    ///         UpsertResponse: An upsert response object. Currently has an 'upserted_count' field with vector count. Might be extended in the future.
    ///     - If `async_req=True`:
    ///         An `asyncio` coroutine that can be awaited using `await` or `asyncio.gather()`.
    pub fn upsert<'a>(
        &mut self,
        py: Python<'a>,
        vectors: Vec<UpsertRecord>,
        namespace: &'a str,
        async_req: bool,
    ) -> PyResult<&'a PyAny> {
        // According to tonic's documentation, cloning the generated client is actually quite cheap,
        // and that's the recommended behavior: https://docs.rs/tonic/latest/tonic/transport/struct.Channel.html#multiplexing-requests
        let mut inner_index = self.inner.clone();

        let namespace = namespace.to_owned();
        let vectors_to_upsert =
            convert_upsert_enum_to_vectors(vectors).map_err(PineconeClientError::from)?;

        if async_req {
            pyo3_asyncio::tokio::future_into_py(py, async move {
                let res = inner_index
                    .upsert(&namespace, &vectors_to_upsert, None)
                    .await
                    .map_err(PineconeClientError::from)?;
                Ok(res)
            })
        } else {
            pyo3_asyncio::tokio::get_runtime().block_on(async move {
                let res = inner_index
                    .upsert(&namespace, &vectors_to_upsert, None)
                    .await
                    .map_err(PineconeClientError::from)?;
                Ok(res.into_py(py).into_ref(py))
            })
        }
    }

    #[pyo3(signature = (top_k, values=None, sparse_values=None, namespace="", filter=None, include_values=false, include_metadata=false))]
    #[pyo3(
        text_signature = "($self, top_k, values=None, sparse_values=None, namespace='', filter=None, include_values=False, include_metadata=False)"
    )]
    /// Query
    ///
    /// The `Query` operation searches a namespace, using a query vector.
    /// It retrieves the ids of the most similar items in a namespace, along with their similarity scores.
    /// To query by the id of already upserted vector, use `Index.query_by_id()`
    ///
    /// Args:
    ///     top_k (int): The number of results to return for each query.
    ///     values (Optional[List[float]]): The values for a new, unseen query vector. This should be the same length as the dimension of the index being queried. The results will be the `top_k` vectors closest to the given vector. Can not be used together with `id`.
    ///     sparse_values (Optional[SparseValues]): The query vector's sparse values.
    ///     namespace (Optional[str]): Optional namespace in which vectors will be queried.
    ///     filter (Optional[dict]): The filter to apply. You can use vector metadata to limit your search. See <https://www.pinecone.io/docs/metadata-filtering/>
    ///     include_values (bool): Indicates whether vector values are included in the response.
    ///     include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
    ///
    /// Returns:
    ///     list of QueryResults
    #[allow(clippy::too_many_arguments)]
    pub fn query(
        &mut self,
        top_k: i32,
        values: Option<Vec<f32>>,
        sparse_values: Option<core_data_types::SparseValues>,
        namespace: &str,
        filter: Option<BTreeMap<String, core_data_types::MetadataValue>>,
        include_values: bool,
        include_metadata: bool,
    ) -> PineconeResult<Vec<core_data_types::QueryResult>> {
        if top_k < 1 {
            return Err(core_error::ValueError("top_k must be greater than 0".to_string()).into());
        }
        let res = self.runtime.block_on(self.inner.query(
            namespace,
            values,
            sparse_values,
            top_k as u32,
            filter,
            include_values,
            include_metadata,
        ))?;
        Ok(res)
    }

    #[pyo3(signature = (id, top_k, namespace="", filter=None, include_values=false, include_metadata=false))]
    #[pyo3(
        text_signature = "($self, id, top_k, namespace='', filter=None, include_values=False, include_metadata=False)"
    )]
    /// Query by id
    ///
    /// The `Query by id` operation searches a namespace given the `id` of a vector already residing in the Index.
    /// It retrieves the ids of the most similar items in a namespace, along with their similarity scores.
    /// To query by new unseen vector use `Index.query()`
    ///
    /// Args:
    ///     id (str): An id of a vector already upserted to the relevant namespace. The results will be the `top_k` nearest neighbours of the vector with the given id. Cannot be used together with `values`.
    ///     top_k (int): The number of results to return for each query.
    ///     namespace (Optional[str]): Optional namespace in which vectors will be queried.
    ///     filter (Optional[dict]): The filter to apply. You can use vector metadata to limit your search. See <https://www.pinecone.io/docs/metadata-filtering/>
    ///     include_values (bool): Indicates whether vector values are included in the response.
    ///     include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
    ///
    /// Returns:
    ///     list of QueryResults
    pub fn query_by_id(
        &mut self,
        id: &str,
        top_k: i32,
        namespace: &str,
        filter: Option<BTreeMap<String, core_data_types::MetadataValue>>,
        include_values: bool,
        include_metadata: bool,
    ) -> PineconeResult<Vec<core_data_types::QueryResult>> {
        if top_k < 1 {
            return Err(core_error::ValueError("top_k must be greater than 0".to_string()).into());
        }
        let res = self.runtime.block_on(self.inner.query_by_id(
            namespace,
            id,
            top_k as u32,
            filter,
            include_values,
            include_metadata,
        ))?;
        Ok(res)
    }

    #[pyo3(signature = (filter=None))]
    #[pyo3(text_signature = "(filter=None)")]
    /// Describe index stats.
    ///
    /// The `DescribeIndexStats` operation returns the number of vectors present in the index, for all the namespaces
    /// and the fullness of the index. Can also accept a filter to count the number of vectors matching the filter.
    ///
    /// Args:
    ///     filter (Dict[str, Union[str, float, int, bool, List, dict]]):
    ///     If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
    ///     See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
    ///
    /// Returns:
    ///     An `IndexStats` object containing index statistics.
    pub fn describe_index_stats(
        &mut self,
        filter: Option<BTreeMap<String, core_data_types::MetadataValue>>,
    ) -> PineconeResult<core_data_types::IndexStats> {
        let res = self
            .runtime
            .block_on(self.inner.describe_index_stats(filter))?;
        Ok(res)
    }

    #[pyo3(signature = (ids, namespace=""))]
    #[pyo3(text_signature = "($self, ids, namespace='')")]
    /// Fetch
    ///
    /// The fetch operation looks up and returns vectors, by ID, from a single namespace.
    /// The returned vectors include the vector data and/or metadata.
    ///
    /// Args:
    ///     ids (List[str]): The vector IDs to fetch.
    ///     namespace (str): The namespace to fetch vectors from.
    ///                      If not specified, the default namespace is used. [optional]
    ///
    /// Examples:
    ///     >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
    ///     >>> index.fetch(ids=['id1', 'id2'])
    ///
    /// Returns: a dictionary of vector IDs to the fetched vectors.
    pub fn fetch(
        &mut self,
        ids: Vec<String>,
        namespace: &str,
    ) -> PineconeResult<HashMap<String, core_data_types::Vector>> {
        let res = self.runtime.block_on(self.inner.fetch(namespace, &ids))?;
        Ok(res)
    }

    #[pyo3(signature = (id, values=None, sparse_values=None, set_metadata=None, namespace=""))]
    #[pyo3(
        text_signature = "($self, id, values=None, sparse_values=None, set_metadata=None, namespace='')"
    )]
    /// Update
    /// The Update operation updates vector in a namespace.
    /// If a value is included, it will overwrite the previous value.
    /// If a set_metadata is included,
    /// the values of the fields specified in it will be added or overwrite the previous value.
    ///
    /// Examples:
    ///     >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
    ///     >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace')
    ///     >>> index.update(id='id1', values=[1, 2, 3], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
    ///                      namespace='my_namespace')
    ///
    /// Args:
    ///     id (str): Vector's unique id.
    ///     values (List[float]): vector values to set. [optional]
    ///     sparse_values: (SparseValues): sparse values to update for the vector.
    ///     set_metadata (Dict[str, Union[str, float, int, bool, List[str]]]]): metadata to set for vector. [optional]
    ///     namespace (str): Namespace name where to update the vector.. [optional]
    ///
    pub fn update(
        &mut self,
        id: &str,
        values: Option<Vec<f32>>,
        sparse_values: Option<core_data_types::SparseValues>,
        set_metadata: Option<BTreeMap<String, core_data_types::MetadataValue>>,
        namespace: &str,
    ) -> PineconeResult<()> {
        self.runtime.block_on(self.inner.update(
            id,
            values.as_ref(),
            sparse_values,
            set_metadata,
            namespace,
        ))?;
        Ok(())
    }

    #[pyo3(signature = (ids, namespace=""))]
    #[pyo3(text_signature = "($self, ids, namespace='')")]
    /// Delete
    /// Delete vectors by ID from a given namespace.
    ///
    /// Args:
    ///     ids (List[str]): A list of IDs for vectors to be deleted.
    ///     namespace (str): The name of the namespace from which vectors will be deleted. If None, the default namespace will be used.
    ///
    /// Returns:
    ///    None
    pub fn delete(&mut self, ids: Vec<String>, namespace: &str) -> PineconeResult<()> {
        self.runtime.block_on(self.inner.delete(ids, namespace))?;
        Ok(())
    }

    #[pyo3(signature = (filter, namespace=""))]
    #[pyo3(text_signature = "($self, filter, namespace='')")]
    /// Delete by filter
    /// The delete by filter operation deletes a list of vectors from a given namespace that match the filter.
    ///
    /// Args:
    ///     filter (Dict[str, Union[str, float, int, bool, List, dict]]): filter to be applied to delete the vectors. See https://www.pinecone.io/docs/metadata-filtering/
    ///     namespace (Optional[str]): The name of the namespace from which vectors will be deleted. If None, the default namespace will be used.
    ///
    /// Returns:
    ///    None
    pub fn delete_by_metadata(
        &mut self,
        filter: Option<BTreeMap<String, core_data_types::MetadataValue>>,
        namespace: &str,
    ) -> PineconeResult<()> {
        self.runtime
            .block_on(self.inner.delete_by_metadata(filter, namespace))?;
        Ok(())
    }

    #[pyo3(signature = (namespace=""))]
    #[pyo3(text_signature = "($self, namespace='')")]
    /// Delete all
    /// The delete all operation deletes all the vectors from a given namespace.
    ///
    /// Args:
    ///     namespace (str): The name of the namespace from which vectors will be deleted. If None, the default namespace will be used.
    ///
    /// Returns:
    ///    None
    pub fn delete_all(&mut self, namespace: &str) -> PineconeResult<()> {
        self.runtime.block_on(self.inner.delete_all(namespace))?;
        Ok(())
    }
}
