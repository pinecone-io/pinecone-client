# Building

## Install dependencies

1. **Install [Rust compiler](https://www.rust-lang.org/tools/install)**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
2. **Install [protobuf compiler](https://grpc.io/docs/protoc-installation/)**
   ```bash
   # Linux
    sudo apt-get install protobuf-compiler
   
   # mac OS
   brew install protobuf
   ```
   Or alternatively install from source: https://github.com/protocolbuffers/protobuf/releases/tag/v22.2

   **Note:** If you are still getting an error like ``Could not find protoc installation`` - set the `PROTOC` environment variable to the `protoc` binary you just installed.
   ```bash
   export PROTOC=/path/to/protoc
   ```

3. **Install lib-ssl**

   If you are getting an error like `Could not find directory of OpenSSL installation`, you need to install lib-ssl.
    #### linux
   ```bash
    sudo apt-get update && sudo apt-get install -y  pkg-config libssl-dev
    ```
    #### mac OS
   ```bash
    brew install openssl
    ```
   
4. **Generate OpenAPI client** (Optional, usually done automatically at build time)
   
   Pinecone uses an OpenAPI spec for control-plane operations like `create_index()`. The OpenAPI client is automatically generated using [openapi-generator](https://github.com/OpenAPITools/openapi-generator/blob/master/docs/generators/rust-server.md) during project build.
   This process uses Docker to `docker run` OpenAPI's generator image.  
   **If you don't have docker installed, or you don't want to use docker** -  you can download the generated code from the [latest release](https://github.com/pinecone-io/pinecone-client/releases). 
   Simply extract the `index_service.zip` file into the `index_service/` folder at the root of the project.

## Building from source
### Python package
#### Using the pyproject.toml file
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install as editable package
cd pinecone 
pip install -e .

# optionallly, install test dependencies and run tests:
pip install -e .[test]
pytest ../tests/unit
```
#### Using `maturin`
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install maturin
pip install maturin

# Install pinecone package for development
cd pinecone
maturin develop
```
#### Building a wheel for deployment
```bash
cd pinecone
maturin build --release
```
### Building rust library for linking with other languages
```bash
cargo build
```

# Contributing
TBD