import os
import time

from loguru import logger
import pinecone
from pinecone import Client
import pytest
import random
from .remote_index import RemoteIndex
from _pytest.python_api import approx
import numpy as np


def index_fixture_factory(remote_indices: [(RemoteIndex, str)]):
    """
    Creates and returns a pytest fixture for creating/tearing down indexes to test against.
    - adds the xdist testrun_uid to the index name unless include_random_suffix is set False
    - fixture yields a pinecone.Index object
    """

    @pytest.fixture(scope="module", params=[remote_index[0] for remote_index in remote_indices],
                    ids=[remote_index[1] for remote_index in remote_indices])
    def index_fixture(testrun_uid, request):
        request.param.index_name = request.param.index_name + '-' + testrun_uid[:8]

        def remove_index():
            env = os.getenv('PINECONE_ENVIRONMENT')
            api_key = os.getenv('PINECONE_API_KEY')
            client = Client(api_key, env)
            if request.param.index_name in client.list_indexes():
                client.delete_index(request.param.index_name)

        # attempt to remove index even if creation raises exception
        request.addfinalizer(remove_index)

        logger.info('Proceeding with index_name {}', request.param.index_name)
        with request.param as index:
            yield index, request.param.index_name

    return index_fixture


def retry_assert(fun, max_tries=5):
    wait_time = 0.5
    for i in range(max_tries):
        try:
            assert fun()
            return
        except Exception as e:
            if i == max_tries - 1:
                raise
            time.sleep(wait_time)
            wait_time = wait_time * 2

def sparse_values(dimension=32000, nnz=120):
    indices = []
    values = []
    threshold = nnz / dimension
    for i in range(dimension):
        key = random.uniform(0, 1)
        if key < threshold:
            indices.append(i)
            values.append(random.uniform(0, 1))
    return indices, values


def get_vector_count(index, namespace):
    stats = index.describe_index_stats().namespaces
    if namespace not in stats:
        return 0
    return stats[namespace].vector_count


def approx_sparse_equals(sv1, sv2):
    if sv1 is None and sv2 is None:
        return True
    return sv1.indices == sv2.indices and sv1.values == approx(sv2.values)