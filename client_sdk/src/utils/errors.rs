use std::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PineconeClientError {
    #[error("Invalid value for argument {name}: {found:?})")]
    ArgumentError { name: String, found: String },

    #[error("`{0}`")]
    ValueError(String),

    #[error("Error in vector number {vec_num}: Missing key '{key}'")]
    UpsertKeyError { key: String, vec_num: usize },

    #[error("Error in vector number {vec_num}: Found unexpected value for '{key}'. Expected a {expected_type}, found: {actual}")]
    UpsertValueError {
        key: String,
        vec_num: usize,
        expected_type: String,
        actual: String,
    },

    #[error("Failed to connect to Pinecone's controller on region {region}. Please verify client configuration: API key, region and project_id. \
        See more info: https://docs.pinecone.io/docs/quickstart#2-get-and-verify-your-pinecone-api-key\n\
        Underlying Error: {err}")]
    ControlPlaneConnectionError { region: String, err: String },

    #[error("Failed to connect to index '{index}'. Please verify that an index with that name exists using `client.list_indexes()`. \n\
        Underlying Error: {err}")]
    IndexConnectionError { index: String, err: String },

    #[error(transparent)]
    DataplaneOperationError(#[from] tonic::Status),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("Unsupported metadata value {val_type}. \
    Please see https://docs.pinecone.io/docs/metadata-filtering#supported-metadata-types for allowed metadata types")]
    MetadataValueError { val_type: String },

    #[error("Unsupported metadata value for key {key}: found value of type {val_type}. \
    Please see https://docs.pinecone.io/docs/metadata-filtering#supported-metadata-types for allowed metadata types")]
    MetadataError { key: String, val_type: String },

    #[error("`{0}`")]
    Other(String),

    #[error("Operation failed with error code {status_code }. \nUnderlying Error: {err}")]
    ControlPlaneOperationError { err: String, status_code: String },

    #[error("Failed to parse response contents")]
    ControlPlaneParsingError {},

    #[error(transparent)]
    DeserializationError(#[from] serde_json::Error),

    #[error("`{0}`")]
    KeyboardInterrupt(String),
}

// TODO: Decide if we want to print the full formatted error on dubug
// impl std::fmt::Debug for PineconeClientError {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.to_string())
//     }
// }

pub type PineconeResult<T> = Result<T, PineconeClientError>;

impl<T> From<index_service::apis::Error<T>> for PineconeClientError {
    fn from(err: index_service::apis::Error<T>) -> Self {
        match err {
            index_service::apis::Error::ResponseError(response_error) => {
                PineconeClientError::ControlPlaneOperationError {
                    err: response_error.content,
                    status_code: response_error.status.to_string(),
                }
            }
            index_service::apis::Error::Reqwest(reqwest_error) => {
                if reqwest_error.is_connect() {
                    PineconeClientError::ControlPlaneConnectionError {
                        region: "".into(),
                        err: reqwest_error.to_string(),
                    }
                } else {
                    PineconeClientError::ControlPlaneOperationError {
                        err: reqwest_error.to_string(),
                        status_code: match reqwest_error.status() {
                            None => "unknown".into(),
                            Some(c) => c.to_string(),
                        },
                    }
                }
            }
            err => PineconeClientError::Other(err.to_string()),
        }
    }
}
