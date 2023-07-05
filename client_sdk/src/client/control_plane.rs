use crate::data_types::Collection;
use crate::data_types::Db;
use crate::data_types::WhoamiResponse;
use crate::utils::errors::PineconeClientError;
use crate::utils::errors::PineconeResult;
use index_service::apis::configuration;
use index_service::apis::index_operations_api;
use index_service::apis::index_operations_api::{
    DescribeCollectionSuccess, DescribeIndexSuccess, ListCollectionsSuccess, ListIndexesSuccess,
};
use index_service::models::CreateCollectionRequest;
use index_service::models::PatchRequest;

#[derive(Debug)]
pub struct ControlPlaneClient {
    controller_url: String,
    configuration: configuration::Configuration,
}

impl ControlPlaneClient {
    pub fn new(controller_url: &str, api_key: &str) -> ControlPlaneClient {
        let mut config = configuration::Configuration::new();
        config.base_path = controller_url.to_string();
        config.api_key = Some(configuration::ApiKey {
            prefix: None,
            key: api_key.to_string(),
        });
        config.user_agent = Some("pinecone-rust-client/0.1".to_string());
        // can pass a custom client here
        config.client = reqwest::Client::new();
        ControlPlaneClient {
            controller_url: controller_url.to_string(),
            configuration: config,
        }
    }

    pub async fn create_index(&self, index: Db) -> PineconeResult<()> {
        index_operations_api::create_index(&self.configuration, Some(index.into())).await?;
        Ok(())
    }

    pub async fn delete_index(&self, name: &str) -> PineconeResult<()> {
        index_operations_api::delete_index(&self.configuration, name).await?;
        Ok(())
    }

    pub async fn describe_index(&self, name: &str) -> PineconeResult<Db> {
        let response = index_operations_api::describe_index(&self.configuration, name).await?;
        match response
            .entity
            .ok_or(PineconeClientError::ControlPlaneParsingError {})?
        {
            DescribeIndexSuccess::Status200(entity) => Db::try_from(entity),
            DescribeIndexSuccess::UnknownValue(val) => {
                Err(PineconeClientError::Other(val.to_string()))
            }
        }
    }

    pub async fn list_indexes(&self) -> PineconeResult<Vec<String>> {
        let response = index_operations_api::list_indexes(&self.configuration).await?;
        match response
            .entity
            .ok_or(PineconeClientError::ControlPlaneParsingError {})?
        {
            ListIndexesSuccess::Status200(entity) => Ok(entity),
            ListIndexesSuccess::UnknownValue(val) => {
                Err(PineconeClientError::Other(val.to_string()))
            }
        }
    }

    pub async fn configure_index(
        &self,
        name: &str,
        pod_type: Option<String>,
        replicas: Option<i32>,
    ) -> PineconeResult<()> {
        let patch_request = PatchRequest { pod_type, replicas };
        index_operations_api::configure_index(&self.configuration, name, Some(patch_request))
            .await?;
        Ok(())
    }

    pub async fn create_collection(&self, collection: Collection) -> PineconeResult<()> {
        let collection_request = CreateCollectionRequest::from(collection);
        index_operations_api::create_collection(&self.configuration, Some(collection_request))
            .await?;
        Ok(())
    }

    pub async fn describe_collection(&self, collection_name: &str) -> PineconeResult<Collection> {
        let response =
            index_operations_api::describe_collection(&self.configuration, collection_name).await?;
        match response
            .entity
            .ok_or(PineconeClientError::ControlPlaneParsingError {})?
        {
            DescribeCollectionSuccess::Status200(entity) => Ok(Collection::from(entity)),
            DescribeCollectionSuccess::UnknownValue(val) => {
                Err(PineconeClientError::Other(val.to_string()))
            }
        }
    }

    pub async fn delete_collection(&self, collection_name: &str) -> PineconeResult<()> {
        index_operations_api::delete_collection(&self.configuration, collection_name).await?;
        Ok(())
    }

    pub async fn list_collections(&self) -> PineconeResult<Vec<String>> {
        let response = index_operations_api::list_collections(&self.configuration).await?;
        match response
            .entity
            .ok_or(PineconeClientError::ControlPlaneParsingError {})?
        {
            ListCollectionsSuccess::Status200(entity) => Ok(entity),
            ListCollectionsSuccess::UnknownValue(val) => {
                Err(PineconeClientError::Other(val.to_string()))
            }
        }
    }

    pub async fn whoami(&self) -> PineconeResult<WhoamiResponse> {
        let rq_client = self.configuration.client.clone();
        let api_key = self
            .configuration
            .api_key
            .as_ref()
            .ok_or_else(|| PineconeClientError::ValueError("Error parsing Api Key".into()))?
            .key
            .as_str();
        if api_key.is_empty() {
            return Err(PineconeClientError::ValueError(
                "Api key empty or not provided".into(),
            ));
        }
        let response = rq_client
            .get(&format!("{}/actions/whoami", self.controller_url))
            .header("Api-Key", api_key)
            .send()
            .await
            .map_err(|e| PineconeClientError::ControlPlaneConnectionError {
                region: " ".to_string(),
                err: e.to_string(),
            })?;
        let json_repsonse = response.json::<WhoamiResponse>().await.map_err(|e| {
            PineconeClientError::ControlPlaneConnectionError {
                region: " ".to_string(),
                err: e.to_string(),
            }
        })?;
        Ok(json_repsonse)
    }
}

#[cfg(test)]
mod control_plane_tests {
    use std::collections::BTreeMap;

    use super::ControlPlaneClient;
    use crate::data_types::Collection;
    use crate::data_types::Db;
    use std::env;

    struct ClientContext {
        client: ControlPlaneClient,
    }
    impl ClientContext {
        fn new() -> Self {
            let controller_uri = format!(
                "https://controller.{}.pinecone.io",
                env::var("PINECONE_REGION").unwrap_or_else(|_| "internal-beta".to_string())
            );
            let api_key = env::var("PINECONE_API_KEY").unwrap_or_else(|_| "".to_string());
            let client = ControlPlaneClient::new(controller_uri.as_str(), api_key.as_str());
            ClientContext { client }
        }
    }

    #[tokio::test]
    async fn test_create() {
        let context = ClientContext::new();
        let index = Db {
            name: "test-index".to_string(),
            dimension: 128,
            metadata_config: Some(
                [(
                    "indexed".to_string(),
                    vec!["value1".to_string(), "value2".to_string()],
                )]
                .iter()
                .cloned()
                .collect::<BTreeMap<String, Vec<String>>>(),
            ),
            ..Default::default()
        };
        let response = context.client.create_index(index).await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_get() {
        let context = ClientContext::new();
        let response = context.client.describe_index("test-index").await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_list() {
        let context = ClientContext::new();
        let response = context.client.list_indexes().await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_update() {
        let context = ClientContext::new();
        let response = context
            .client
            .configure_index("test-index", None, Some(2))
            .await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_create_collection() {
        let context = ClientContext::new();
        let collection: Collection = Collection {
            name: "test-collection".to_string(),
            source: "test-index".to_string(),
            ..Default::default()
        };
        let response = context.client.create_collection(collection).await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_list_collection() {
        let context = ClientContext::new();
        let response = context.client.list_collections().await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_describe_collection() {
        let context = ClientContext::new();
        let response = context.client.describe_collection("test-collection").await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_delete_collection() {
        let context = ClientContext::new();
        let response = context.client.delete_collection("test-collection").await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_delete_index() {
        let context = ClientContext::new();
        let response = context.client.delete_index("test-index").await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_delete_invalid_timeout() {
        let context = ClientContext::new();
        let response = context.client.delete_index("test-index").await;
        println!("{:?}", response);
    }

    #[tokio::test]
    async fn test_whoami() {
        let context = ClientContext::new();
        let response = context.client.whoami().await;
        println!("{:?}", response);
        assert!(response.is_ok());
    }
}
