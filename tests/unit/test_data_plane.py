import itertools
import os
import sys
import numpy as np
import pinecone as pinecone
from pinecone import Vector,Client, PineconeOpError
import pytest
from loguru import logger
import asyncio

from ..utils.remote_index import RemoteIndex, PodType
from ..utils.utils import index_fixture_factory, retry_assert

logger.remove()
logger.add(sys.stdout, level=(os.getenv("PINECONE_LOGGING") or "INFO"))

vector_dim = 512
env = os.getenv('PINECONE_REGION')
api_key = os.getenv('PINECONE_API_KEY')
client = Client(api_key,env)
INDEX_NAME_PREFIX = 'data-plane'

test_data_plane_index = index_fixture_factory(
    [
        (RemoteIndex(pods=2, index_name=f'{INDEX_NAME_PREFIX}-{PodType.P1}',
                     dimension=vector_dim, pod_type=PodType.P1), str(PodType.P1))
    ]
)

def get_random_metadata():
    # return a random value for metadata key 
    # it can be either a string,float,int, bool or list of strings
    return np.random.choice(['action', 'documentary', 'drama'], np.random.randint(1, 5),np.random.rand(),[np.random.choice(['action', 'documentary', 'drama']) for _ in range(5)])

def construct_random_metadata():
    base_dict = {
        'some_string': np.random.choice(['action', 'documentary', 'drama']),
        'some_int': np.random.randint(2000, 2021),
        'some_bool' : np.random.choice([True, False]),
        'some_float' : np.random.rand(),
        'some_list' : [np.random.choice(['action', 'documentary', 'drama']) for _ in range(5)]
    }

    # Add some random additional keys
    for _ in range(3):
        base_dict[f'key_{np.random.randint(1, 100)}'] = np.random.choice(['action', 'documentary', 'drama',
                                                                          np.random.randint(2000, 2021),
                                                                          np.random.choice([True, False]),
                                                                          np.random.rand()])
    return base_dict

def get_test_data(vector_count=10, no_meta_vector_count=5, dimension=vector_dim):
    """repeatably produces same results for a given vector_count"""
    meta_vector_count = vector_count - no_meta_vector_count

    no_meta_vectors: list[Vector] = [
        Vector(f'vec{i}', np.random.rand(dimension).tolist())
        for i in range(no_meta_vector_count)
    ]
    meta_vectors: list[Vector] = [
        Vector(f'mvec{i}', np.random.rand(dimension).tolist(), None, construct_random_metadata())
        for i in range(meta_vector_count)
    ]
    assert len(meta_vectors)==meta_vector_count
    assert len(no_meta_vectors)==no_meta_vector_count
    return meta_vectors + no_meta_vectors


def get_test_data_dict(vector_count=10, no_meta_vector_count=5, dimension=vector_dim):
    return {vec.id: (vec.values, vec.metadata) for vec in
            get_test_data(vector_count, no_meta_vector_count, dimension)}


def get_vector_count(index, namespace):
    stats = index.describe_index_stats().namespaces
    if namespace not in stats:
        return 0
    return stats[namespace].vector_count


def write_test_data(index, namespace, vector_count=10, no_meta_vector_count=5, dimension=vector_dim, batch_size=300):
    """writes vector_count vectors into index, half with metadata half without."""
    data = get_test_data(vector_count, no_meta_vector_count, dimension)
    count_before = get_vector_count(index, namespace)

    upsert(index, namespace, data, batch_size)

    retry_assert(lambda: len(data) == (get_vector_count(index, namespace) - count_before))
    data_dict = {vec.id: vec for vec in
            data}
    return data_dict


def upsert(index, namespace, data, batch_size = -1):
    def chunks():
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(data)
        chunk = list(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(it, batch_size))
    total_vectors_upserted = 0
    for chunk in chunks():
        res = index.upsert(vectors=chunk, namespace=namespace)
        total_vectors_upserted += res.upserted_count
    assert total_vectors_upserted == len(data)

async def async_upload(index, vectors, batch_size):
    def chunker(seq, batch_size):
        return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

    async_results = [
        index.upsert(vectors=chunk, async_req=True)
        for chunk in chunker(vectors, batch_size=batch_size)
    ]

    total_vectors_upserted = 0
    for task in asyncio.as_completed(async_results):
        res = await task
        total_vectors_upserted += res.upserted_count
    return total_vectors_upserted


def test_client_invalid_api_key():
    with pytest.raises(ConnectionError):
        # all our api keys have a uuid format
        pinecone = Client(api_key='invalid_api_key')


def test_summarize_no_api_key():
    with pytest.raises(ValueError) as exc_info:
        client = pinecone.Client('', region=env)
        nonexistent_index = client.get_index('nonexistent-index')
        api_response = nonexistent_index.describe_index_stats()
        assert "api key" in str(exc_info.value.lower())
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


def test_summarize_nonexistent_index():
    logger.info("api key header: " + os.getenv('PINECONE_API_KEY'))
    with pytest.raises(ConnectionError) as exc_info:
        nonexistent_index = client.get_index('nonexistent-index')
        api_response = nonexistent_index.describe_index_stats()
        assert "nonexistent-index" in str(exc_info.value)
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


def test_invalid_upsert_no_params(test_data_plane_index):
    index, _ = test_data_plane_index
    with pytest.raises(TypeError) as exc_info:
        api_response = index.upsert()
        logger.debug('got api response {}', api_response)
        assert "missing" in str(exc_info.value)
    logger.debug('got expected exception: {}', exc_info.value)


def test_invalid_upsert_vector_no_values(test_data_plane_index):
    index, _ = test_data_plane_index
    with pytest.raises(TypeError) as exc_info:
        api_response = index.upsert([Vector(id='bad_vec1')])
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


def test_upsert_vectors_no_metadata(test_data_plane_index):
    # TODO : md and sparse should not be required postional args
    index, _ = test_data_plane_index
    namespace = 'test_upsert_vectors_no_metadata'
    test_data = get_test_data(vector_count=50, no_meta_vector_count=50)
    api_response = index.upsert(
        vectors=test_data,
        namespace=namespace,
    )
    assert api_response.upserted_count == len(test_data)
    logger.debug('got upsert without metadata response: {}', api_response)


def test_upsert_vectors(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_upsert_vectors'
    test_data = get_test_data(vector_count=50, no_meta_vector_count=0)
    api_response = index.upsert(
        vectors=test_data,
        namespace=namespace,
    )
    assert api_response.upserted_count == len(test_data)
    logger.debug('got upsert with metadata response: {}', api_response)


def test_upsert_vectors_async(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_upsert_vectors_async'
    test_data = get_test_data(vector_count=500, no_meta_vector_count=200)
    upserted_count = asyncio.run(async_upload(index, test_data, batch_size=100))
    assert upserted_count == len(test_data)


def test_invalid_upsert_vectors_wrong_dimension(test_data_plane_index):
    index, _ = test_data_plane_index
    with pytest.raises(PineconeOpError) as exc_info:
        api_response = index.upsert(
            vectors=[
                Vector(id='vec1', values=[0.1] * 50),
                Vector(id='vec2', values=[0.2] * 50),
            ],
            namespace='ns1',
        )
        assert "dimension" in str(exc_info.value)
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)
    # assert exc_info.value.status == 400
    assert "dimension" in str(exc_info.value)


def test_fetch_vectors_no_metadata(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_fetch_vectors_no_metadata'
    vector_count = 40
    test_data = write_test_data(index, namespace, vector_count, no_meta_vector_count=vector_count)

    api_response = index.fetch(ids=list(test_data.keys()), namespace=namespace)
    # Assert response is not none
    assert api_response
    logger.debug('got fetch without metadata response: {}', api_response)

    for test_vector in test_data.values():
        id = test_vector.id
        assert id in api_response
        assert api_response.get(id).values == test_vector.values
        assert not api_response.get(id).metadata


def test_fetch_vectors(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_fetch_vectors'
    vector_count = 40
    test_data = write_test_data(index, namespace, vector_count)
    api_response = index.fetch(ids=[vec.id for vec in list(test_data.values())], namespace=namespace)
    logger.debug('got fetch response: {}', api_response)

    for test_vector in test_data.values():
        id = test_vector.id
        assert id in api_response
        assert api_response.get(id).values == test_vector.values
        assert api_response.get(id).metadata == test_vector.metadata
        assert api_response.get(id).id == test_vector.id


def test_fetch_vectors_mixed_metadata(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_fetch_vectors_mixed_metadata'
    vector_count = 100
    test_data = write_test_data(index, namespace, vector_count, no_meta_vector_count=50)

    api_response = index.fetch(ids=list(test_data.keys()), namespace=namespace)
    logger.debug('got fetch response: {}', api_response)

    for vector_id in test_data.keys():
        assert api_response.get(vector_id)
        assert api_response.get(vector_id).values == test_data.get(vector_id).values
        if vector_id.startswith('m'):
            assert api_response.get(vector_id).metadata == test_data.get(vector_id).metadata


def test_invalid_fetch_nonexistent_vectors(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_invalid_fetch_nonexistent_vectors'
    write_test_data(index, namespace)

    api_response = index.fetch(ids=['no-such-vec1', 'no-such-vec2'], namespace=namespace)
    logger.debug('got fetch response: {}', api_response)


def test_invalid_fetch_nonexistent_namespace(test_data_plane_index):
    index, _ = test_data_plane_index
    api_response = index.fetch(ids=['no-such-vec1', 'no-such-vec2'], namespace='no-such-namespace')
    assert len(api_response.keys()) == 0
    logger.debug('got fetch response: {}', api_response)


def test_summarize(test_data_plane_index):
    index, _ = test_data_plane_index
    vector_count = 400
    namespace = 'test_describe_index_stats'
    stats_before = index.describe_index_stats()
    assert stats_before.index_fullness == 0
    write_test_data(index, namespace, vector_count=vector_count, dimension=vector_dim)
    response = index.describe_index_stats()
    assert response.namespaces[namespace].vector_count == vector_count
    assert response.total_vector_count == stats_before.total_vector_count + vector_count


def test_summarize_with_filter(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_describe_index_stats_with_filter'
    count_before = get_vector_count(index, namespace)
    before_total_count = index.describe_index_stats().total_vector_count
    vectors = [Vector('1', [0.1] * vector_dim,None, {'color': 'yellow'}),
               Vector('2', [-0.1] * vector_dim,None, {'color': 'red'}),
               Vector('3', [0.1] * vector_dim),
               Vector('4', [-0.1] * vector_dim,None,{'color': 'red'})]
    upsert_response = index.upsert(vectors=vectors, namespace=namespace)
    retry_assert(lambda: len(vectors) == get_vector_count(index, namespace) - count_before)
    logger.debug('got upsert response: {}', upsert_response)

    response = index.describe_index_stats(filter={'color': 'red'})
    assert response.namespaces[namespace].vector_count == 2
    assert response.total_vector_count == before_total_count + len(vectors)


def test_invalid_query_params(test_data_plane_index):
    index, _ = test_data_plane_index
    with pytest.raises(TypeError) as exc_info:
        api_response = index.query(top_k=4,
                                   values=[0.1] * vector_dim,
                                   queries=[[0.1] * vector_dim,
                                            [0.2] * vector_dim])
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


def test_negative_top_k(test_data_plane_index):
    index, _ = test_data_plane_index
    with pytest.raises(ValueError) as exc_info:
        api_response = index.query(top_k=-1,
                                   values=[0.1] * vector_dim)
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


def test_large_top_k(test_data_plane_index):
    index, _ = test_data_plane_index
    with pytest.raises(PineconeOpError) as exc_info:
        api_response = index.query(top_k=12000,
                                   values=[0.1] * vector_dim)
        logger.debug('got api response {}', api_response)
    logger.debug('got expected exception: {}', exc_info.value)


def test_query_simple(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_simple'
    vector_count = 10
    write_test_data(index, namespace, vector_count)
    # simple query - no filter, no data, no metadata
    api_response = index.query(
        values=np.random.rand(vector_dim).tolist(),
        namespace=namespace,
        top_k=10,
        include_values=False,
        include_metadata=False
    )
    logger.debug('got query (no filter, no data, no metadata) response: {}', api_response)

    first_match_vector = api_response[0]
    assert not first_match_vector.values
    assert not first_match_vector.metadata


def test_query_simple_with_values(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_simple_with_values'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    # simple query - no filter, with data, no metadata
    api_response = index.query(
        values=np.random.rand(vector_dim).tolist(),
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=False
    )
    logger.debug('got query (no filter, with data, no metadata) response: {}', api_response)

    first_match_vector = api_response[0]
    assert first_match_vector.values == test_data.get(first_match_vector.id).values
    assert not first_match_vector.metadata


def test_query_simple_with_values_metadata(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_simple_with_values_metadata'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    # simple query - no filter, with data, with metadata
    api_response = index.query(
        values=np.random.rand(vector_dim).tolist(),
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got query (no filter, with data, with metadata) response: {}', api_response)

    first_match_vector = api_response[0]
    assert first_match_vector.values == test_data.get(first_match_vector.id).values
    if first_match_vector.id.startswith('mvec'):
        assert first_match_vector.metadata == test_data.get(first_match_vector.id).metadata
    else:
        assert not first_match_vector.metadata


def test_query_simple_with_values_mixed_metadata(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_simple_with_values_mixed_metadata'
    top_k = 10
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count, no_meta_vector_count=5)
    # simple query - no filter, with data, with metadata
    api_response = index.query(
        values=
            np.random.rand(vector_dim).tolist(),
        namespace=namespace,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    logger.debug('got query (no filter, with data, with metadata) response: {}', api_response)


    assert len(api_response) == top_k
    for match_vector in api_response:
        assert match_vector.values == test_data.get(match_vector.id).values
        if test_data.get(match_vector.id).values:
            assert match_vector.metadata == test_data.get(match_vector.id).metadata
        else:
            assert not match_vector.metadata


def test_query_simple_with_filter_values_metadata(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_simple_with_filter_values_metadata'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    api_response = index.query(
        values=[0.1] * vector_dim,
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=True,
        filter={'genre': {'$in': ['action']}}
    )
    logger.debug('got query (with filter, with data, with metadata) response: {}', api_response)
    if not api_response:
        pytest.skip('no vectors match the filter')
    first_match_vector = api_response[0]
    assert first_match_vector.values == test_data.get(first_match_vector.id).values
    assert first_match_vector.metadata == test_data.get(first_match_vector.id).metadata
    assert first_match_vector.metadata.get('genre') == 'action'


def test_query_mixed_metadata_sanity(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_mixed_metadata'
    count_before = get_vector_count(index, namespace)
    vectors = [Vector('1', [0.1] * vector_dim, None, {'colors': 'yellow'}),
               Vector('2', [-0.1] * vector_dim, None, {'colors': 'red'})]
    upsert_response = index.upsert(vectors=vectors, namespace=namespace)
    retry_assert(lambda: len(vectors) == get_vector_count(index, namespace) - count_before)
    logger.debug('got upsert response: {}', upsert_response)

    query1_response = index.query(values=[0.1] * vector_dim,
                                  filter ={'colors': 'yellow'},
                                  top_k=10,
                                  include_metadata=True,
                                  namespace=namespace)
    logger.debug('got first query response: {}', query1_response)

    query2_response = index.query(values=[0.1] * vector_dim,
                                  filter = {'colors': 'yellow'},
                                  top_k=10,
                                  include_metadata=True,
                                  namespace=namespace)
    logger.debug('got second query response: {}', query2_response)

    vectors_dict = {v.id: v.metadata for v in vectors}
    for query_vector in query1_response:
        if vectors_dict.get(query_vector.id):
            assert query_vector.metadata == vectors_dict.get(query_vector.id)
        else:
            assert not query_vector.metadata

    for query_vector in query2_response:
        # for match_vector in query_vector_results:
        if vectors_dict.get(query_vector.id):
            assert query_vector.metadata == vectors_dict.get(query_vector.id)
        else:
            assert not query_vector.metadata


def test_invalid_query_nonexistent_namespace(test_data_plane_index):
    index, _ = test_data_plane_index
    api_response = index.query(
        values=[0.1] * vector_dim,
        namespace='no-such-ns',
        top_k=10,
        include_values=True,
        include_metadata=True,
        filter={'action': {'$in': ['action']}}
    )
    logger.debug('got query (with filter, with data, with metadata) response: {}', api_response)

def test_query_uses_distributed_knn(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_with_multi_shard'
    write_test_data(index, namespace, vector_count=1000, no_meta_vector_count=1000)
    query_response = index.query(
        values=[0.1] * vector_dim,
        namespace=namespace,
        top_k=500,
        include_values=False,
        include_metadata=False
    )
    # assert that we got the same number of results as the top_k
    # regardless of the number of shards
    assert len(query_response) == 500
    logger.debug('got query response: {}', query_response)


def test_query_by_id(test_data_plane_index):
    index, _ = test_data_plane_index
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
    logger.debug('got query response: {}', api_response)

    assert len(api_response) == 10
    for match_vector in api_response:
        assert match_vector.values == test_data.get(match_vector.id).values
        if test_data.get(match_vector.id).values:
            assert match_vector.metadata == test_data.get(match_vector.id).metadata
        else:
            assert not match_vector.metadata


def test_query_by_nonexistent_id(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_by_nonexistent_id'
    id_ = 'nonexistent'
    api_response = index.query_by_id(
        id=id_,
        namespace=namespace,
        top_k=10,
        include_values=True,
        include_metadata=False
    )
    logger.debug('got query response: {}', api_response)

    assert len(api_response) == 0


def test_query_with_exists_filter(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_query_exists'
    count_before = get_vector_count(index, namespace)
    vectors = [Vector('0', [0.1] * vector_dim, None,{'colors': True, 'country': 'greece'}),
               Vector('1', [-0.1] * vector_dim, None,{'colors': False}),
               Vector('2', [0.2] * vector_dim)]
    upsert_response = index.upsert(vectors=vectors, namespace=namespace)
    retry_assert(lambda: len(vectors) == get_vector_count(index, namespace) - count_before)
    logger.debug('got upsert response: {}', upsert_response)

    response_1 = index.query(values=[0.1] * vector_dim,
                             filter={'$or': [{'colors': {'$exists': False}}, {'country': {'$eq': 'greece'}}]},
                             top_k=10,
                             namespace=namespace)
    logger.debug('got query response: {}', response_1)

    matches_ids = {match_vector.id for match_vector in response_1}
    assert len(response_1) == 2
    assert matches_ids == {'0', '2'}

    response_2 = index.query(values=[0.1] * vector_dim,
                             filter={'colors': {'$exists': True}},
                             top_k=10,
                             namespace=namespace)
    logger.debug('got query response: {}', response_2)

    matches_ids = {match_vector.id for match_vector in response_2}
    assert len(response_2) == 2
    assert matches_ids == {'0', '1'}


def test_delete(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_delete'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)

    api_response = index.fetch(ids=['mvec1', 'mvec2'], namespace=namespace)
    logger.debug('got fetch response: {}', api_response)
    assert api_response and api_response.get('mvec1').values == test_data.get('mvec1').values

    vector_count = get_vector_count(index, namespace)
    api_response = index.delete(ids=['vec1', 'vec2'], namespace=namespace)
    logger.debug('got delete response: {}', api_response)
    retry_assert(lambda: get_vector_count(index, namespace) == (vector_count - 2))
    api_response = index.fetch(ids=['no-such-vec1', 'no-such-vec2'], namespace=namespace)
    logger.debug('got fetch response: {}', api_response)


def test_delete_all(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_delete_all'
    write_test_data(index, namespace)
    api_response = index.delete_all( namespace=namespace)
    logger.debug('got delete response: {}', api_response)
    retry_assert(lambda: namespace not in index.describe_index_stats().namespaces)


def test_invalid_delete_nonexistent_ids(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_nonexistent_ids'
    write_test_data(index, namespace)
    api_response = index.delete(ids=['no-such-vec-1', 'no-such-vec-2'], namespace=namespace)
    logger.debug('got delete response: {}', api_response)


def test_invalid_delete_from_nonexistent_namespace(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_delete_namespace_non_existent'
    api_response = index.delete(ids=['vec1', 'vec2'], namespace=namespace)
    logger.debug('got delete response: {}', api_response)


def test_delete_all_nonexistent_namespace(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_delete_all_non_existent'
    api_response = index.delete_all(namespace=namespace)
    logger.debug('got delete response: {}', api_response)


def test_update(test_data_plane_index):
    index, _ = test_data_plane_index
    namespace = 'test_update'
    vector_count = 10
    test_data = write_test_data(index, namespace, vector_count)
    assert get_vector_count(index, namespace) == vector_count

    api_response = index.update(id='mvec1', namespace=namespace, values=test_data.get('mvec2').values)
    logger.debug('got update response: {}', api_response)
    retry_assert(
        lambda: index.fetch(ids=['mvec1'], namespace=namespace).get('mvec1').values == test_data.get('mvec2').values)
    assert index.fetch(ids=['mvec1'], namespace=namespace).get('mvec1').metadata == test_data.get('mvec1').metadata

    api_response = index.update(id='mvec2', namespace=namespace, set_metadata=test_data.get('mvec1').metadata)
    logger.debug('got update response: {}', api_response)
    expected_metadata = test_data.get('mvec2').metadata.copy()
    expected_metadata.update(test_data.get('mvec1').metadata.copy())
    retry_assert(
        lambda: index.fetch(ids=['mvec2'], namespace=namespace).get('mvec2').metadata == expected_metadata)
    assert index.fetch(ids=['mvec2'], namespace=namespace).get('mvec2').values == test_data.get('mvec2').values

    api_response = index.update(id='mvec3', namespace=namespace, values=test_data.get('mvec1').values,
                                set_metadata=test_data.get('mvec2').metadata)
    logger.debug('got update response: {}', api_response)
    expected_metadata = test_data.get('mvec3').metadata.copy()
    expected_metadata.update(test_data.get('mvec2').metadata.copy())
    retry_assert(
        lambda: index.fetch(ids=['mvec3'], namespace=namespace).get('mvec3').values == test_data.get('mvec1').values)
    assert index.fetch(ids=['mvec3'], namespace=namespace).get('mvec3').metadata == expected_metadata
