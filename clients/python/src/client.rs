use std::collections::BTreeMap;

use client_sdk::data_types::{Collection, Db};
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::index::Index;
use crate::utils::errors::{PineconeClientError, PineconeResult};
use client_sdk::client::pinecone_client as core_client;
use client_sdk::utils::errors::{self as core_errors};

#[pyclass]
#[pyo3(text_signature = "(api_key=None, region=None, project_id=None)")]
pub struct Client {
    inner: core_client::PineconeClient,
    runtime: Runtime,
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (api_key=None, region=None, project_id=None))]
    /// Creates a Pinecone client instance.
    /// Configuration parameters are usually set as environment variables. If you want to override the environment variables, you can pass them as arguments to the constructor.
    ///
    /// Args:
    ///     api_key (str, optional): The API key to use for authentication. Defaults to the value of the `PINECONE_API_KEY` environment variable. See more info here: https://docs.pinecone.io/docs/quickstart#2-get-and-verify-your-pinecone-api-key
    ///     region (str, optional): The pinecone region to use. Defaults to the value of the `PINECONE_REGION` environment variable, or to `us-west1-gcp` if the environment variable is not set.
    ///     project_id (str, optional): By default, the client will use project id associated with the API key. If you want to use a different project id, you can pass it as an argument to the constructor.
    ///
    /// Returns:
    ///    Client: A Pinecone client instance.
    pub fn new(
        api_key: Option<&str>,
        region: Option<&str>,
        project_id: Option<&str>,
    ) -> PineconeResult<Self> {
        let rt = Runtime::new().map_err(core_errors::PineconeClientError::IoError)?;
        let client = rt.block_on(core_client::PineconeClient::new(
            api_key, region, project_id,
        ))?;

        Ok(Self {
            inner: client,
            runtime: rt,
        })
    }

    pub fn __repr__(&self) -> String {
        let api_key = self.inner.api_key.split('-').last().unwrap_or("None");
        format!(
            "Client:\n  API key: ****************-{api_key}\n  region: {region}\n  project_id: {project_id}",
            api_key = api_key,
            region = self.inner.region,
            project_id = self.inner.project_id
        )
    }

    /// Index
    ///
    /// The Index is the main object for interacting with a Pinecone index. It is used to insert, update, fetch and query vectors.
    /// You can create an Index object by calling the `get_index` method on the Pinecone client.
    /// You can also create an Index object by calling the `create_index` method on the Pinecone client.
    /// This method is a shortcut for `get_index` for backwards compatibility and will eventually be deprecated.
    ///
    /// Args:
    ///     name (str): The name an existing Pinecone index to connect to.
    ///
    /// Returns:
    ///    Index: The index object.
    #[allow(non_snake_case)]
    pub fn Index(&self, name: &str) -> PineconeResult<Index> {
        self.get_index(name)
    }

    /// Get an Index object for interacting with a Pinecone index.
    ///
    /// The Index is the main object for interacting with a Pinecone index. It is used to insert, update, fetch and query vectors.
    /// You can create an Index object by calling the `get_index` method on the Pinecone client.
    /// You can also create an Index object by calling the `create_index` method on the Pinecone client.
    ///
    /// Args:
    ///     name (str): The name an existing Pinecone index to connect to.
    ///
    /// Returns:
    ///    Index: The index object.
    pub fn get_index(&self, index_name: &str) -> PineconeResult<Index> {
        let inner_index = self.runtime.block_on(self.inner.get_index(index_name))?;
        Ok(Index::new(inner_index, self.runtime.handle().clone()))
    }

    /// Creates a new Pinecone index.
    ///
    /// Args:
    ///     name (str): The name of the index to be created. The maximum length is 45 characters.
    ///     dimension (int): The dimensions of the vectors to be inserted in the index.
    ///     metric (str, optional): The distance metric to be used for similarity search. You can use 'euclidean', 'cosine', or 'dotproduct'. Defaults to 'cosine'.
    ///     replicas (int, optional): The number of replicas. Replicas duplicate your index. They provide higher availability and throughput. Defaults to 1.
    ///     shards (int, optional): The number of shards to be used in the index. Defaults to 1.
    ///     pods (int, optional): The number of pods for the index to use,including replicas. Defaults to 1.
    ///     pod_type (str, optional): The type of pod to use. One of `s1`, `p1`, or `p2` appended with `.` and one of `x1`, `x2`, `x4`, or `x8`. Defaults to p1.x1.
    ///     metadata_config (dict, optional): Configuration for the behavior of Pinecone's internal metadata index. By default, all metadata is indexed; when `metadata_config` is present, only specified metadata fields are indexed. To specify metadata fields to index, provide a JSON object of the following form: {"indexed": ["example_metadata_field"]}.
    ///     source_collection (str, optional): The name of the collection to create an index from.
    ///     timeout (int, optional): The number of seconds to wait for the index to be created. Defaults to 300 seconds. Pass -1 to avoid waiting for the index to be created.
    ///
    /// Returns:
    ///     Index: The index object, if successfully created.
    #[pyo3(signature = (name, dimension, metric=None, replicas=None, shards=None, pods=None, pod_type=None, metadata_config=None, source_collection=None, timeout=None))]
    #[pyo3(
        text_signature = "($self, name, dimension, metric=None, replicas=None, shards=None, pods=None, pod_type=None, metadata_config=None, source_collection=None)"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn create_index(
        &self,
        name: &str,
        py: Python<'_>,
        dimension: i32,
        metric: Option<String>,
        replicas: Option<i32>,
        shards: Option<i32>,
        pods: Option<i32>,
        pod_type: Option<String>,
        metadata_config: Option<BTreeMap<String, Vec<String>>>,
        source_collection: Option<String>,
        timeout: Option<i32>,
    ) -> PineconeResult<Index> {
        let db = Db {
            name: name.into(),
            dimension,
            metric,
            replicas,
            shards,
            pods,
            pod_type,
            metadata_config,
            source_collection,
            ..Default::default()
        };
        self.runtime
            .block_on(self.inner.create_index(db, timeout, Some(py)))?;
        // If successful return an Index object
        self.get_index(name)
    }

    /// Delete an index.
    ///
    /// Args:
    ///     name (str): The name of the index to delete.
    ///     timeout (int, optional): The number of seconds to wait for the index to be deleted. Defaults to 300 seconds. Pass -1 to avoid waiting for the index to be deleted.
    ///
    /// Returns:
    ///     None
    pub fn delete_index(&self, name: &str, timeout: Option<i32>) -> PineconeResult<()> {
        self.runtime
            .block_on(self.inner.delete_index(name, timeout))?;
        Ok(())
    }

    /// List all indexes
    ///
    /// Returns:
    ///  List[str]: A list of all indexes in the project
    pub fn list_indexes(&self) -> PineconeResult<Vec<String>> {
        let res = self.runtime.block_on(self.inner.list_indexes())?;
        Ok(res)
    }

    ///  Describe an index.
    ///
    ///  Args:
    ///      name (str): The name of the index to describe.
    ///
    ///  Returns:
    ///      DB: An object describing the index configuration.
    pub fn describe_index(&self, name: &str) -> PineconeResult<Db> {
        let res = self.runtime.block_on(self.inner.describe_index(name))?;
        Ok(res)
    }

    #[pyo3(signature = (name, replicas=None, pod_type=None))]
    #[pyo3(text_signature = "($self, name, replicas=None, pod_type=None)")]
    /// Configure an index.
    ///
    /// Args:
    ///     name (str): The name of the index to rescale or configure.
    ///     replicas (int): The number of replicas to use for the index.
    ///     pod_type (str): The type of pod to use for the index.
    ///
    /// Returns:
    ///     None
    pub fn scale_index(
        &self,
        name: &str,
        replicas: Option<i32>,
        pod_type: Option<String>,
    ) -> PineconeResult<()> {
        // at least one of replicas or pod_type must be set
        if replicas.is_none() && pod_type.is_none() {
            return Err(PineconeClientError::from(
                core_errors::PineconeClientError::ValueError(
                    "At least one of replicas or pod_type must be set".into(),
                ),
            ));
        }
        self.runtime
            .block_on(self.inner.configure_index(name, pod_type, replicas))?;
        Ok(())
    }

    /// Create a new collection.
    ///
    /// Args:
    ///     name (str): The name of the collection to create.
    ///     source_index (str): The name of the index to use as the source for the collection.
    ///
    /// Returns:
    ///     None
    pub fn create_collection(
        &self,
        name: &str,
        source_index: &str,
    ) -> Result<(), PineconeClientError> {
        self.runtime
            .block_on(self.inner.create_collection(name, source_index))?;
        Ok(())
    }

    /// Describe a collection
    ///
    /// Args:
    ///     name (str): The name of the collection to describe
    ///
    /// Returns:
    ///     Collection: The collection description
    pub fn describe_collection(&self, name: &str) -> Result<Collection, PineconeClientError> {
        let res = self
            .runtime
            .block_on(self.inner.describe_collection(name))?;
        Ok(res)
    }

    /// List all collections
    ///
    /// Returns:
    ///     List[str] - A list of all collections
    pub fn list_collections(&self) -> PineconeResult<Vec<String>> {
        let res = self.runtime.block_on(self.inner.list_collections())?;
        Ok(res)
    }

    /// Delete a collection
    ///
    /// Args:
    ///     name (str): The name of the collection to delete.
    ///
    /// Returns:
    ///     None
    pub fn delete_collection(&self, name: &str) -> Result<(), PineconeClientError> {
        self.runtime.block_on(self.inner.delete_collection(name))?;
        Ok(())
    }
}
