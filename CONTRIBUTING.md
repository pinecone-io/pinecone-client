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
   
4. **Install [poetry](https://python-poetry.org/)**
5. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
5. **Generate OpenAPI client** (Optional, usually done automatically at build time)
   
   Pinecone uses an OpenAPI spec for control-plane operations like `create_index()`. The OpenAPI client is automatically generated using [openapi-generator](https://github.com/OpenAPITools/openapi-generator/blob/master/docs/generators/rust-server.md) during project build.
   This process uses Docker to `docker run` OpenAPI's generator image.  
   **If you don't have docker installed, or you don't want to use docker** -  you can download the generated code from the [latest release](https://github.com/pinecone-io/pinecone-client/releases). 
   Simply extract the `index_service.zip` file into the `index_service/` folder at the root of the project.

## Building from source

After you have installed prerequisites above, you can run 

```bash
make build
```

This command will do several things:
- Generate updated openapi client code (via docker in `make generate-index-service`)
- Build the rust code into a python module (via `make build-python` which invokes `maturin develop`)
- Install the built module in the venv managed by poetry (maturin handles this as well)
- Install python dependencies used in testing via `poetry install`

Depending on the situation, you may sometimes want to perform these actions individually. But if you're just getting started you probably want to run all of these steps.

## Run the tests

Rust tests:
```
PINECONE_API_KEY='foo' PINECONE_REGION='bar' cargo test -p pinecone -p client_sdk
```

Python tests:

```bash
PINECONE_API_KEY='foo' PINECONE_REGION='bar' make integration-test
```

### Try out the python module in an interactive repl session

```bash
poetry run python3
```

This drops you into an interactive session where the development module can be imported from and experimented with

```python
>>> from pinecone import Client
>>> Client()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: `Please provide a valid API key or set the 'PINECONE_API_KEY' environment variable`
>>>
```

#### Building a wheel for deployment
```bash
make release
```

### Building rust library for linking with other languages
```bash
cargo build
```

# Contributing
TBD