.PHONY: install build integration-test finish clean

install:
	poetry install

build-python:
	cd pinecone; poetry run maturin develop

release:
	cd pinecone; poetry run maturin build --release

build: generate-index-service install build-python

run:
	poetry run python3

integration-test: install
	poetry run pytest --self-contained-html --durations=10 --durations-min=1.0  --html=tests/unit/report.html tests/unit 

generate-index-service:
	docker run --rm -v "${CURDIR}:/local" openapitools/openapi-generator-cli:v6.3.0 generate --input-spec /local/openapi/index_service.json  --generator-name rust  --output /local/index_service --additional-properties packageName=index_service --additional-properties packageVersion=0.1.0 --additional-properties withSerde=true  --additional-properties supportMultipleResponses=true
