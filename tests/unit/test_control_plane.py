"""Tests for control plane api calls"""
import os
from pinecone import Client
import pytest
from time import time
import numpy as np 
from ..utils.utils import retry_assert
from ..utils.remote_index import PodType

env = os.getenv('PINECONE_REGION')
key = os.getenv('PINECONE_API_KEY')
client = Client(key, env)
INDEX_NAME_PREFIX = 'control-plane-' + str(np.random.randint(10000))
d = 512

INDEX_NAME_KEY = 'index_name'
POD_TYPE_KEY = 'pod_type'


@pytest.fixture(scope="module",
                params=[
                    {INDEX_NAME_KEY: f'{INDEX_NAME_PREFIX}-{PodType.P1}',
                     POD_TYPE_KEY: PodType.P1}
                ],
                ids=lambda param: str(param[POD_TYPE_KEY]))
def index_fixture(testrun_uid, request):
    index_name = request.param[INDEX_NAME_KEY] + '-' + testrun_uid[:8]
    pod_type = request.param[POD_TYPE_KEY]
    # Note: relies on grouping strategy –-dist=loadfile to keep different xdist workers
    # from repeating this
    index_creation_args = {'name': index_name,
                           'dimension': d,
                           'pod_type': str(pod_type),
                           'pods': 2}
    client.create_index(**index_creation_args)

    def remove_index():
        if index_name in client.list_indexes():
            client.delete_index(index_name)

    # attempt to remove index even if creation raises exception
    request.addfinalizer(remove_index)

    yield index_name, f"{pod_type}.x1" if pod_type.is_implicitly_x1() else str(pod_type)

# The client interface
def test_client_valid_params():
    # assert no error is raised
    pinecone = Client(api_key='api_key', region='env',project_id='project_id')

def test_env_vars():
    # assuming tess are run with env vars set
    pinecone = Client()

@pytest.fixture(scope="function")
def set_api_key_env_var(testrun_uid):
    old_key = os.environ.get('PINECONE_API_KEY')
    os.environ['PINECONE_API_KEY'] = "non-existent-key"
    yield old_key
    if old_key is not None:
        os.environ['PINECONE_API_KEY'] = old_key

@pytest.fixture(scope="function")
def set_region_env_var(testrun_uid):
    old_region = os.environ.get('PINECONE_REGION')
    os.environ['PINECONE_REGION'] = "non-existent-region"
    yield old_region
    if old_region is not None:
        os.environ['PINECONE_REGION'] = old_region

def test_env_vars_missing_api_key(set_api_key_env_var):
    with pytest.raises(ConnectionError):
        pinecone = Client()

def test_env_vars_missing_region(set_region_env_var):
    with pytest.raises(ConnectionError):
        pinecone = Client()

def test_env_var_override(set_api_key_env_var, set_region_env_var):
    assert os.environ['PINECONE_API_KEY'] == "non-existent-key"
    assert os.environ['PINECONE_REGION'] == "non-existent-region"
    pinecone = Client(api_key=set_api_key_env_var, region=set_region_env_var)
    pinecone.list_indexes()

# def test_env_var_override_region(set_region_env_var):
#     assert os.environ['PINECONE_REGION'] == "non-existent-region"
#     pinecone = Client(region=set_region_env_var)
#     pinecone.list_indexes()
def test_client_invalid_params():
    with pytest.raises(TypeError):
        pinecone = Client(random_param='random_param')

def test_client_invalid_region():
    with pytest.raises(ConnectionError):
        pinecone = Client(region='invalid_region')

def test_missing_dimension():
    # Missing Dimension
    name = 'test-missing-dim'
    with pytest.raises(TypeError):
        client.create_index(name)


def test_invalid_name():
    # Missing Dimension
    name = 'Test-Bad-Name'
    # TODO: raise proper exception
    with pytest.raises(Exception) as e:
        client.create_index(name, 32)
    # assert e.value.status == 400

def test_invalid_name_rfc_1123():
    name = "bad.name.with.periods"
    # TODO: raise proper exception
    with pytest.raises(Exception) as e:
        client.create_index(name, 32)
    # assert e.value.status == 400

@pytest.fixture(scope="module")
def timeout_index(testrun_uid):
    name = f'{INDEX_NAME_PREFIX}-create-timeout' + '-' + testrun_uid[:8]
    if name in client.list_indexes():
        client.delete_index(name)
    yield name
    if name in client.list_indexes():
        client.delete_index(name)

def test_create_timeout(timeout_index):
    timeout = 5 # seconds
    TOLERANCE = 0.5 # seconds
    before = time()
    with pytest.raises(RuntimeError) as e:
        client.create_index(timeout_index, 32, timeout=timeout)
        eplased = time() - before
        assert eplased - timeout < TOLERANCE
        assert "timed out" in str(e.value)

def test_create_timeout_invalid():
    timeout = -5 # seconds
    with pytest.raises(ValueError) as e:
        client.create_index('test-create-timeout-invalid', 32, timeout=timeout)
        assert "-1 or a positive integer" in str(e.value)
    assert "test-create-timeout-invalid" not in client.list_indexes()

def test_create_duplicate(index_fixture):
    index_name, _ = index_fixture
    # Duplicate index
    with pytest.raises(Exception):
        client.create_index(index_name, 32)

def test_get(index_fixture):
    index_name, pod_type = index_fixture
    # Successful Call
    result = client.describe_index(index_name)
    # TODO: dict vs obj
    return_val = {'name': index_name, 'dimension': 512, 'replicas': 1, 'shards': 2, 'pod_type': pod_type, 'metric': 'cosine', 'pods': 2, 'source_collection': None, 'metadata_config': None, 'status': 'Ready'}
    assert result.to_dict() == return_val
    # Calling non-existent index 
    with pytest.raises(Exception):
        client.describe_index('non-existent-index')

    # Missing Field
    with pytest.raises(TypeError):
        client.describe_index()


def test_update(index_fixture):
    index_name, _ = index_fixture
    # Scale Up
    num_replicas = 2
    client.scale_index(name=index_name, replicas=num_replicas)

    retry_assert(lambda:  client.describe_index(index_name).status == 'Ready')
    meta_obj = client.describe_index(index_name)
    assert meta_obj.replicas == 2
    assert meta_obj.pods == 4

    # Missing replicas field
    # TODO: raise proper exception
    with pytest.raises(ValueError):
        client.scale_index(index_name)

    # Calling on non-existent index
    with pytest.raises(Exception):
        client.scale_index('non-existent-index', 2)

    # Scale to zero
    num_replicas = 0
    client.scale_index(name=index_name, replicas=num_replicas)
    retry_assert(lambda:  client.describe_index(index_name).status == 'Ready')
    meta_obj = client.describe_index(index_name)
    assert meta_obj.replicas == num_replicas
    assert meta_obj.pods == num_replicas

    
    
    
def test_delete(index_fixture):
    index_name, _ = index_fixture   
    # Delete existing index
    client.delete_index(index_name)
    assert index_name not in client.list_indexes()

    # Delete non existent index
    with pytest.raises(Exception):
        client.delete_index('non-existent-index')

    # Missing Field
    with pytest.raises(TypeError):
        client.delete_index()


def test_create_collection_from_nonexistent_index():
    nonexistent_index_name = 'nonexistent_index_name'
    collection_name = 'collection'
    with pytest.raises(Exception) as e:
        client.create_collection(collection_name, nonexistent_index_name)
    assert f'source database {nonexistent_index_name} does not exist' in str(e.value)


def test_create_index_from_nonexistent_collection():
    index_name = 'test-index-collection'
    nonexistent_collection_name = 'nonexistent_collection_name'
    with pytest.raises(Exception) as e:
        client.create_index(index_name, d, pods=2, pod_type='s1',
                              source_collection=nonexistent_collection_name)
    assert f'failed to fetch source collection {nonexistent_collection_name}' in str(e.value)
