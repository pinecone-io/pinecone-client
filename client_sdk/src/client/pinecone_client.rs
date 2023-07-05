use pyo3::Python;
use std::io::Write;
use std::time::{Duration, Instant};
use std::{env, io};

use super::control_plane::ControlPlaneClient;
use super::grpc::DataplaneGrpcClient;
use crate::data_types::{Collection, Db};
use crate::index::Index;
use crate::utils::errors::PineconeClientError::IndexConnectionError;
use crate::utils::errors::{PineconeClientError, PineconeResult};

const DEAULT_PINECONE_REGION: &str = "us-west1-gcp";

#[derive(Debug)]
pub struct PineconeClient {
    pub api_key: String,
    pub region: String,
    pub project_id: String,
    control_plane_client: ControlPlaneClient,
}

impl PineconeClient {
    pub async fn new(
        api_key: Option<&str>,
        region: Option<&str>,
        project_id: Option<&str>,
    ) -> PineconeResult<Self> {
        let api_key = match api_key {
                Some(s) => Ok(s.to_string()),
                None => env::var("PINECONE_API_KEY").map_err(|_| PineconeClientError::ValueError("Please provide a valid API key or set the 'PINECONE_API_KEY' environment variable".to_string()))
            }?;
        let region = match region {
            Some(s) => s.to_string(),
            None => {
                env::var("PINECONE_REGION").unwrap_or_else(|_| DEAULT_PINECONE_REGION.to_string())
            }
        };
        // Check if region is empty. For cases where the user sets the region to an empty string
        if region.is_empty() {
            return Err(PineconeClientError::ValueError(
                "Please provide a valid region or set the 'PINECONE_REGION' environment variable"
                    .to_string(),
            ));
        }
        let control_plane_client =
            ControlPlaneClient::new(&PineconeClient::get_controller_url(&region), &api_key);
        let project_id = match project_id {
            Some(id) => id.to_string(),
            None => PineconeClient::get_project_id(&control_plane_client)
                .await
                .map_err(|e| match e {
                    PineconeClientError::ControlPlaneConnectionError { err, .. } => {
                        PineconeClientError::ControlPlaneConnectionError {
                            err,
                            region: region.clone(),
                        }
                    }
                    _ => e,
                })?,
        };

        Ok(PineconeClient {
            api_key,
            region,
            project_id,
            control_plane_client,
        })
    }

    fn get_index_url(&self, index_name: &str) -> String {
        let output = format!(
            "https://{index_name}-{project_id}.svc.{region}.pinecone.io:443",
            index_name = index_name,
            project_id = self.project_id,
            region = self.region
        );
        output
    }

    fn get_controller_url(region: &str) -> String {
        let output = format!("https://controller.{}.pinecone.io", region);
        output
    }

    async fn get_dataplane_grpc_client(
        &self,
        index_name: &str,
    ) -> PineconeResult<DataplaneGrpcClient> {
        let index_endpoint_url = self.get_index_url(index_name);
        let client = DataplaneGrpcClient::connect(index_endpoint_url, &self.api_key)
            .await
            .map_err(|e| IndexConnectionError {
                index: index_name.to_string(),
                err: e.to_string(),
            })?;
        Ok(client)
    }

    async fn get_project_id(control_plane_client: &ControlPlaneClient) -> PineconeResult<String> {
        let whoami_response = control_plane_client.whoami().await?;
        Ok(whoami_response.project_name)
    }

    pub async fn create_index(
        &self,
        db: Db,
        timeout: Option<i32>,
        py: Option<Python<'_>>,
    ) -> PineconeResult<()> {
        // If timeout is -ve and not -1 throw an error
        let name = db.name.clone();
        // If timeout is -ve and not -1 throw an error
        if timeout.is_some() && timeout.unwrap() < -1 {
            return Err(PineconeClientError::ValueError(
                "Timeout must be -1 or a positive integer".to_string(),
            ));
        }
        self.control_plane_client.create_index(db).await?;
        // If -1 then don't wait for index to be ready
        if timeout == Some(-1) {
            return Ok(());
        }
        // block until index is ready
        let mut new_index = self.describe_index(&name).await?;
        let start_time = Instant::now();
        let max_timeout = Duration::from_secs(timeout.unwrap_or(300) as u64);
        if let Some(py) = py {
            py.run(
                "print(\"Waiting for index to be ready...\", flush=True)",
                None,
                None,
            )
            .map_err(|_| PineconeClientError::Other("Failed to print to stdout".to_string()))?;
        } else {
            println!("Waiting for index to be ready...");
            io::stdout().flush()?;
        }
        while new_index.status != Some("Ready".to_string()) {
            if let Some(py) = py {
                Python::check_signals(py)
                    .map_err(|_| {
                        let msg = "Interrupted. Index status unknown. Please call describe_index() to check status";
                        println!("{}", msg);
                        io::stdout().flush().unwrap();
                        PineconeClientError::KeyboardInterrupt(
                            msg.into(),
                        )
                    })?;
            }
            if start_time.elapsed() > max_timeout {
                return Err(PineconeClientError::Other(
                    "Index creation timed out. Please call describe_index() to check status."
                        .to_string(),
                ));
            }
            new_index = self.describe_index(&name).await?;
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
        Ok(())
    }

    pub async fn get_index(&self, index_name: &str) -> PineconeResult<Index> {
        Ok(Index::new(
            index_name.to_string(),
            self.get_dataplane_grpc_client(index_name).await?,
        ))
    }

    pub async fn describe_index(&self, index_name: &str) -> PineconeResult<Db> {
        self.control_plane_client.describe_index(index_name).await
    }

    pub async fn list_indexes(&self) -> PineconeResult<Vec<String>> {
        self.control_plane_client.list_indexes().await
    }

    pub async fn delete_index(&self, index_name: &str, timeout: Option<i32>) -> PineconeResult<()> {
        // If timeout is -ve and not -1 throw an error
        if timeout.is_some() && timeout.unwrap() < -1 {
            return Err(PineconeClientError::ValueError(
                "Timeout must be -1 or a positive integer".to_string(),
            ));
        }
        self.control_plane_client.delete_index(index_name).await?;
        if timeout == Some(-1) {
            return Ok(());
        }
        // block until index is deleted
        println!("Verifying delete...");
        let start_time = Instant::now();
        let max_timeout = Duration::from_secs(timeout.unwrap_or(300) as u64);
        while self.list_indexes().await?.contains(&index_name.to_string()) {
            if start_time.elapsed() > max_timeout {
                return Err(PineconeClientError::Other(
                    "Index deletion timed out. Please call describe_index to check status."
                        .to_string(),
                ));
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
        Ok(())
    }

    pub async fn configure_index(
        &self,
        index_name: &str,
        pod_type: Option<String>,
        replicas: Option<i32>,
    ) -> PineconeResult<()> {
        self.control_plane_client
            .configure_index(index_name, pod_type, replicas)
            .await
    }

    pub async fn create_collection(
        &self,
        collection_name: &str,
        source_index: &str,
    ) -> PineconeResult<()> {
        let collection = Collection {
            name: collection_name.to_string(),
            source: source_index.to_string(),
            ..Default::default()
        };
        self.control_plane_client
            .create_collection(collection)
            .await
    }

    pub async fn describe_collection(&self, collection_name: &str) -> PineconeResult<Collection> {
        self.control_plane_client
            .describe_collection(collection_name)
            .await
    }

    pub async fn list_collections(&self) -> PineconeResult<Vec<String>> {
        self.control_plane_client.list_collections().await
    }

    pub async fn delete_collection(&self, collection_name: &str) -> PineconeResult<()> {
        self.control_plane_client
            .delete_collection(collection_name)
            .await
    }
}

mod tests {
    #[tokio::test]
    async fn test_env_vars() {
        use super::*;
        env::set_var("PINECONE_API_KEY", "");
        env::set_var("PINECONE_REGION", "");
        let client = PineconeClient::new(None, None, None).await.unwrap();
        println!("{:?}", client);
    }
}
