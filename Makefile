VENV_DIR = .temp_venv

.PHONY: setup venv develop integration-test-python finish clean

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	touch $(VENV_DIR)/bin/activate

develop: venv
	. $(VENV_DIR)/bin/activate; cd clients/python && pip3 install -e .[test]

integration-test-python: venv develop
	. $(VENV_DIR)/bin/activate; cd clients/python/tests/unit && pytest --self-contained-html --dist=loadscope --numprocesses 4 --durations=10 --durations-min=1.0  --html=report.html
	$(MAKE) finish

finish: venv
	rm -rf $(VENV_DIR)

clean:
	rm -rf $(VENV_DIR)

generate-index-service:
	docker run --rm -v "${CURDIR}:/local" openapitools/openapi-generator-cli:latest-release generate --input-spec /local/openapi/index_service.json  --generator-name rust  --output /local/index_service --additional-properties packageName=index_service --additional-properties packageVersion=0.1.0 --additional-properties withSerde=true  --additional-properties supportMultipleResponses=true
