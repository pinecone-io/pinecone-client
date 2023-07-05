import os
import sys

from loguru import logger
from pinecone import Vector, Client, SparseValues

from ..utils.remote_index import PodType, RemoteIndex
from ..utils.utils import index_fixture_factory, retry_assert, sparse_values, get_vector_count, approx_sparse_equals

logger.remove()
logger.add(sys.stdout, level=(os.getenv("PINECONE_LOGGING") or "INFO"))

vector_dim = 512
env = os.getenv('PINECONE_REGION')
api_key = os.getenv('PINECONE_API_KEY')

INDEX_NAME = 'test-hybrid-search'

hybrid_index = index_fixture_factory(
    [
        (RemoteIndex(pods=2, index_name=f'{INDEX_NAME}-{PodType.S1}',
                     dimension=vector_dim, pod_type=PodType.S1, metric='dotproduct'), str(PodType.S1)),
    ]
)


def sparse_vector(dimension=32000, nnz=120):
    indices, values = sparse_values(dimension, nnz)
    return SparseValues(indices=indices, values=values)


def get_test_data(vector_count=10, no_meta_vector_count=5, dimension=vector_dim, sparse=True):
    """repeatably produces same results for a given vector_count"""
    meta_vector_count = vector_count - no_meta_vector_count
    metadata_choices = [
        {'genre': 'action', 'year': 2020},
        {'genre': 'documentary', 'year': 2021},
        {'genre': 'documentary', 'year': 2005},
        {'genre': 'drama', 'year': 2011},
    ]
    no_meta_vectors: list[Vector] = [
        Vector(id=f'vec{i}', values=[i / 1000] * dimension, sparse_values=sparse_vector() if sparse else None)
        for i in range(no_meta_vector_count)
    ]
    meta_vectors: list[Vector] = [
        Vector(id=f'mvec{i}', values=[i / 1000] * dimension, sparse_values=sparse_vector() if sparse else {},
               metadata=metadata_choices[i % len(metadata_choices)])
        for i in range(meta_vector_count)
    ]

    return list(meta_vectors) + list(no_meta_vectors)


def write_test_data(index, namespace, vector_count=10, no_meta_vector_count=5, dimension=vector_dim, batch_size=300):
    """writes vector_count vectors into index, half with metadata half without."""
    data = get_test_data(vector_count, no_meta_vector_count, dimension)
    index.upsert(vectors=data, namespace=namespace)
    return {vector.id: vector for vector in data}


def test_upsert_vectors(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_upsert_vectors'
    api_response = index.upsert(
        vectors=[
            Vector(id='mvec1', values=[0.1] * vector_dim, sparse_values=sparse_vector(), metadata={'genre': 'action', 'year': 2020}),
            Vector(id='mvec2', values=[0.2] * vector_dim, sparse_values=sparse_vector(), metadata={'genre': 'documentary', 'year': 2021}),
        ],
        namespace=namespace,
    )
    assert api_response.upserted_count == 2
    logger.debug('got openapi upsert with metadata response: {}', api_response)


def test_fetch_vectors_mixed_metadata(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_fetch_vectors_mixed_metadata'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count, no_meta_vector_count=5)
    api_response = index.fetch(ids=['vec1', 'mvec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)

    for vector_id in ['mvec2', 'vec1']:
        expected_vector = test_data.get(vector_id)
        fetched_vector = api_response[vector_id]
        assert fetched_vector
        assert fetched_vector.values == expected_vector.values
        assert fetched_vector.metadata == expected_vector.metadata
        assert approx_sparse_equals(fetched_vector.sparse_values, expected_vector.sparse_values)


def test_query_simple(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_query_simple'
    vector_count = 100
    write_test_data(index, namespace, vector_count)
    # simple query - no filter, no data, no metadata
    dense_query_response = index.query(
        values=[0.1] * vector_dim,
        namespace=namespace,
        top_k=10,
        include_values=False,
        include_metadata=False
    )
    logger.debug('got openapi query (no filter, no data, no metadata) response: {}', dense_query_response)

    dense_query_match = dense_query_response[0]
    assert not dense_query_match.values
    assert not dense_query_match.sparse_values
    assert not dense_query_match.metadata

    hybrid_query_response = index.query(
        values=[0.1] * vector_dim,
        sparse_values=sparse_vector(),
        namespace=namespace,
        top_k=10,
        include_values=False,
        include_metadata=False
    )

    assert dense_query_response != hybrid_query_response


def test_query_simple_with_include_values(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_query_simple_with_values_metadata'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    # simple query - no filter, with data, with metadata
    api_response = index.query(
        values=[0.1] * vector_dim,
        sparse_values=sparse_vector(),
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got openapi query (no filter, with data, with metadata) response: {}', api_response)

    first_match_vector = api_response[0]
    expected_vector = test_data.get(first_match_vector.id)
    assert first_match_vector.values == expected_vector.values
    assert approx_sparse_equals(first_match_vector.sparse_values, expected_vector.sparse_values)
    if first_match_vector.id.startswith('mvec'):
        assert first_match_vector.metadata == test_data.get(first_match_vector.id).metadata
    else:
        assert not first_match_vector.metadata


def test_delete(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_delete'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)

    expected_mvec1 = test_data.get('mvec1')
    api_response = index.fetch(ids=['mvec1', 'mvec2'], namespace=namespace)
    logger.debug('got openapi fetch response: {}', api_response)
    assert api_response and api_response.get('mvec1').values == expected_mvec1.values
    assert api_response and approx_sparse_equals(api_response.get('mvec1').sparse_values, expected_mvec1.sparse_values)

    vector_count = get_vector_count(index, namespace)
    api_response = index.delete(ids=['vec1', 'vec2'], namespace=namespace)
    logger.debug('got openapi delete response: {}', api_response)
    retry_assert(lambda: get_vector_count(index, namespace) == (vector_count - 2))


def test_update(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_update'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    assert get_vector_count(index, namespace) == vector_count

    api_response = index.update(id='mvec1', namespace=namespace, values=test_data.get('mvec2').values)
    logger.debug('got openapi update response: {}', api_response)
    retry_assert(
        lambda: index.fetch(ids=['mvec1'], namespace=namespace)['mvec1'].values == test_data.get('mvec2').values)
    fetched_response = index.fetch(ids=['mvec1'], namespace=namespace)['mvec1']
    assert fetched_response.metadata == test_data.get('mvec1').metadata
    assert approx_sparse_equals(fetched_response.sparse_values, test_data.get('mvec1').sparse_values)

    api_response = index.update(id='mvec1', namespace=namespace, sparse_values=test_data.get('mvec2').sparse_values)
    logger.debug('got openapi update response: {}', api_response)
    retry_assert(
        lambda: approx_sparse_equals(index.fetch(ids=['mvec1'], namespace=namespace)['mvec1'].sparse_values, test_data.get('mvec2').sparse_values))
    fetched_response = index.fetch(ids=['mvec1'], namespace=namespace)['mvec1']
    assert fetched_response.values == test_data.get('mvec2').values
    assert fetched_response.metadata == test_data.get('mvec1').metadata

    api_response = index.update(id='mvec2', namespace=namespace, set_metadata=test_data.get('mvec1').metadata)
    logger.debug('got openapi update response: {}', api_response)
    retry_assert(
        lambda: index.fetch(ids=['mvec2'], namespace=namespace)['mvec2'].metadata == test_data.get('mvec1').metadata)
    fetched_response = index.fetch(ids=['mvec1'], namespace=namespace)['mvec1']
    assert fetched_response.values == test_data.get('mvec2').values
    assert approx_sparse_equals(fetched_response.sparse_values, test_data.get('mvec2').sparse_values)

    api_response = index.update(id='mvec3', namespace=namespace, values=test_data.get('mvec1').values,
                                sparse_values=test_data.get('mvec1').sparse_values, set_metadata=test_data.get('mvec1').metadata)
    logger.debug('got openapi update response: {}', api_response)
    retry_assert(
        lambda: index.fetch(ids=['mvec3'], namespace=namespace)['mvec3'].values == test_data.get('mvec1').values)
    fetched_response = index.fetch(ids=['mvec1'], namespace=namespace)['mvec1']
    assert approx_sparse_equals(fetched_response.sparse_values, test_data.get('mvec2').sparse_values)
    assert fetched_response.metadata == test_data.get('mvec1').metadata


def test_upsert_with_dense_only(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_upsert_with_dense_only'
    vectors = [Vector('0', [0.1] * vector_dim, metadata={'colors': True, 'country': 'greece'}),
               Vector('1', [-0.1] * vector_dim, metadata={'colors': False}),
               Vector('2', [0.2] * vector_dim, sparse_values=sparse_vector(),metadata={})]
    index.upsert(vectors=vectors, namespace=namespace)

    api_response = index.fetch(ids=['0', '1', '2'], namespace=namespace)
    for expected_vector in vectors:
        vector = api_response.get(expected_vector.id)
        assert vector.values == expected_vector.values
        if(vector.metadata is not None and expected_vector.metadata!={}):
            assert vector.metadata == expected_vector.metadata
        assert approx_sparse_equals(vector.sparse_values,expected_vector.sparse_values)


def test_query_by_id(hybrid_index):
    index, _ = hybrid_index
    namespace = 'test_query_by_id'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    api_response = index.query_by_id(
        id='vec1',
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got openapi query response: {}', api_response)

    assert len(api_response) == 10
    for match_vector in api_response:
        expected_vector = test_data.get(match_vector.id)
        assert match_vector.values == expected_vector.values
        assert approx_sparse_equals(expected_vector.sparse_values, match_vector.sparse_values)
        if expected_vector.metadata:
            assert match_vector.metadata == expected_vector.metadata
        else:
            assert not match_vector.metadata