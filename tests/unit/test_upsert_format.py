from copy import deepcopy

import numpy as np
import pytest
from pinecone import Client, Vector, SparseValues
import os
from ..utils.remote_index import RemoteIndex, PodType
from ..utils.utils import index_fixture_factory, retry_assert

vector_dim = 512
env = os.getenv('PINECONE_REGION')
api_key = os.getenv('PINECONE_API_KEY')
client = Client(api_key,env)
INDEX_NAME_PREFIX = 'upsert-format'

test_data_plane_index = index_fixture_factory(
    [
        (RemoteIndex(pods=2, index_name=f'{INDEX_NAME_PREFIX}-{PodType.P1}',
                     dimension=vector_dim, pod_type=PodType.P1), str(PodType.P1))
    ]
)

def assert_sparse_vectors(sparse_vec_1:SparseValues,sparse_vec_2:dict):
    assert sparse_vec_1.indices == sparse_vec_2['indices']
    assert sparse_vec_1.values == sparse_vec_2['values']


def get_random_metadata():
    return {
        'some_string': np.random.choice(['action', 'documentary', 'drama']),
        'some_int': np.random.randint(2000, 2021),
        'some_bool' : np.random.choice([True, False]),
        'some_float' : np.random.rand(),
        'some_list' : [np.random.choice(['action', 'documentary', 'drama']) for _ in range(5)]
    }

def get_random_vector():
    return np.random.rand(vector_dim).astype(np.float32).tolist()

def get_random_sparse_vector():
    indices = np.random.choice(vector_dim, 10, replace=False).astype(np.int32).tolist()
    values = np.random.rand(10).astype(np.float32).tolist()
    return SparseValues(indices=indices, values=values)

def get_random_sparse_dict():
    indices = np.random.choice(vector_dim, 10, replace=False).astype(np.int32).tolist()
    values = np.random.rand(10).astype(np.float32).tolist()
    return {'indices': indices, 'values': values}

def sparese_dict_to_vec(sparse_vec):
    return SparseValues(indices=sparse_vec['indices'], values=sparse_vec['values'])
    
def test_upsert_tuplesOfIdVec_UpserWithoutMD(test_data_plane_index):
    index, _ = test_data_plane_index
    index.upsert([('vec1', get_random_vector()), ('vec2', get_random_vector())], namespace='ns')


def test_upsert_tuplesOfIdVecMD_UpsertVectorsWithMD(test_data_plane_index):
    index, _ = test_data_plane_index
    index.upsert([('vec1', get_random_vector(), get_random_metadata()), ('vec2', get_random_vector(), get_random_metadata())], namespace='ns')

def test_upsert_vectors_upsertInputVectorsSparse(test_data_plane_index):
    index, _ = test_data_plane_index
    index.upsert([Vector(id='vec1', values=get_random_vector(), metadata=get_random_metadata(),
                                sparse_values=get_random_sparse_vector()),
                        Vector(id='vec2', values=get_random_vector(), metadata=get_random_metadata(),
                                sparse_values=get_random_sparse_vector())],
                        namespace='ns')

def test_upsert_dict(test_data_plane_index):
    index, _ = test_data_plane_index
    dict1 = {'id': 'vec1', 'values': get_random_vector()}
    dict2 = {'id': 'vec2', 'values': get_random_vector()}
    index.upsert([dict1, dict2], namespace='ns')
    fetched_vectors = index.fetch(['vec1', 'vec2'], namespace='ns')
    assert len(fetched_vectors) == 2
    assert fetched_vectors.get('vec1').id == 'vec1'
    assert fetched_vectors.get('vec2').id == 'vec2'
    assert fetched_vectors.get('vec1').values == dict1['values']
    assert fetched_vectors.get('vec2').values == dict2['values']



def test_upsert_dict_md(test_data_plane_index):
    index, _ = test_data_plane_index
    dict1 = {'id': 'vec1', 'values': get_random_vector(), 'metadata': get_random_metadata()}
    dict2 = {'id': 'vec2', 'values': get_random_vector(), 'metadata': get_random_metadata()}
    index.upsert([dict1, dict2], namespace='ns')
    fetched_vectors = index.fetch(['vec1', 'vec2'], namespace='ns')
    assert len(fetched_vectors) == 2
    assert fetched_vectors.get('vec1').id == 'vec1'
    assert fetched_vectors.get('vec2').id == 'vec2'
    assert fetched_vectors.get('vec1').values == dict1['values']
    assert fetched_vectors.get('vec2').values == dict2['values']
    assert fetched_vectors.get('vec1').metadata == dict1['metadata']
    assert fetched_vectors.get('vec2').metadata == dict2['metadata']

def test_upsert_dict_sparse(test_data_plane_index):
    index, _ = test_data_plane_index
    dict1 = {'id': 'vec1', 'values': get_random_vector(),
                'sparse_values': get_random_sparse_dict()}
    dict2 = {'id': 'vec2', 'values': get_random_vector(),
                'sparse_values': get_random_sparse_dict()}
    index.upsert([dict1, dict2], namespace='ns')
    fetched_vectors = index.fetch(['vec1', 'vec2'], namespace='ns')
    assert len(fetched_vectors) == 2
    assert fetched_vectors.get('vec1').id == 'vec1'
    assert fetched_vectors.get('vec2').id == 'vec2'
    assert fetched_vectors.get('vec1').values == dict1['values']
    assert fetched_vectors.get('vec2').values == dict2['values']
    assert_sparse_vectors(fetched_vectors.get('vec1').sparse_values, dict1['sparse_values'])
    assert_sparse_vectors(fetched_vectors.get('vec2').sparse_values, dict2['sparse_values'])

def test_upsert_dict_sparse_md(test_data_plane_index):
    index, _ = test_data_plane_index
    dict1 = {'id': 'vec1', 'values': get_random_vector(),
                'sparse_values': get_random_sparse_dict(), 'metadata': get_random_metadata()}
    dict2 = {'id': 'vec2', 'values': get_random_vector(),
                'sparse_values': get_random_sparse_dict(), 'metadata': get_random_metadata()}
    index.upsert([dict1, dict2], namespace='ns')
    fetched_vectors = index.fetch(['vec1', 'vec2'], namespace='ns')
    assert len(fetched_vectors) == 2
    assert fetched_vectors.get('vec1').id == 'vec1'
    assert fetched_vectors.get('vec2').id == 'vec2'
    assert fetched_vectors.get('vec1').values == dict1['values']
    assert fetched_vectors.get('vec2').values == dict2['values']
    assert_sparse_vectors(fetched_vectors.get('vec1').sparse_values, dict1['sparse_values'])
    assert_sparse_vectors(fetched_vectors.get('vec2').sparse_values, dict2['sparse_values'])
    assert fetched_vectors.get('vec1').metadata == dict1['metadata']
    assert fetched_vectors.get('vec2').metadata == dict2['metadata']


def test_upsert_dict_negative(test_data_plane_index):
    index, _ = test_data_plane_index

    # Missing required keys
    dict1 = {'values': get_random_vector()}
    dict2 = {'id': 'vec2'}
    with pytest.raises(ValueError):
        index.upsert([dict1, dict2])
    with pytest.raises(ValueError):
        index.upsert([dict1])
    with pytest.raises(ValueError):
        index.upsert([dict2])

    # Excess keys
    dict1 = {'id': 'vec1', 'values': get_random_vector()}
    dict2 = {'id': 'vec2', 'values': get_random_vector(), 'animal': 'dog'}
    with pytest.raises(ValueError) as e:
        index.upsert([dict1, dict2])
        assert 'animal' in str(e.value)

    dict1 = {'id': 'vec1', 'values': get_random_vector(), 'metadatta': get_random_metadata()}
    dict2 = {'id': 'vec2', 'values': get_random_vector()}
    with pytest.raises(ValueError) as e:
        index.upsert([dict1, dict2])
        assert 'metadatta' in str(e.value)

@pytest.mark.parametrize("key,new_val", [
    ("id", 4.2),
    ("id", ['vec1']),
    ("values", ['the', 'lazy', 'fox']),
    ("values", 'the lazy fox'),
    ("values", 0.5),
    ("metadata", np.nan),
    ("metadata", ['key1', 'key2']),
    ("sparse_values", 'cat'),
    ("sparse_values", []),
])
def test_upsert_dict_negative_types(test_data_plane_index, key, new_val):
    index, _ = test_data_plane_index
    full_dict1 = {'id': 'vec1', 'values': get_random_vector(),
                    'sparse_values': get_random_sparse_dict(),
                    'metadata': get_random_metadata()}

    dict1 = deepcopy(full_dict1)
    dict1[key] = new_val
    with pytest.raises(ValueError) as e:
        index.upsert([dict1])
    assert key in str(e.value)

@pytest.mark.parametrize("key,new_val", [
    ("indices", 3),
    ("indices", [1.2, 0.5]),
    ("values", ['1', '4.4']),
    ("values", 0.5),
])
def test_upsert_dict_negative_types_sparse(test_data_plane_index, key, new_val):
    index, _ = test_data_plane_index

    full_dict1 = {'id': 'vec1', 'values': get_random_vector(),
                    'sparse_values': get_random_sparse_dict(),
                    'metadata': get_random_metadata()}

    dict1 = deepcopy(full_dict1)
    dict1['sparse_values'][key] = new_val
    # TODO: Lenght mismatch between indices and values should be done on client or server?
    with pytest.raises((Exception,ValueError)) as e:
        index.upsert([dict1])
    assert key in str(e.value)
