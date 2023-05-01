import dataclasses
import os
import sys
import uuid
from itertools import cycle
from typing import Any

import numpy as np
import pytest
from loguru import logger

from pinecone import Vector, SparseValues
from ..utils.remote_index import RemoteIndex, PodType
from ..utils.utils import retry_assert, index_fixture_factory

logger.remove()
logger.add(sys.stdout, level=(os.getenv("PINECONE_LOGGING") or "INFO"))

d = 128
n = 100

INDEX_NAME_PREFIX = 'test-metadata'
MAPPING_INDEX_NAME_PREFIX = 'test-mapping'

test_metadata_index = index_fixture_factory(
    [
        (RemoteIndex(pods=1, index_name=f'{INDEX_NAME_PREFIX}-{PodType.P1}',
                     dimension=d, pod_type=PodType.P1), str(PodType.P1))
    ]
)

test_metadata_index_with_mapping = index_fixture_factory(
    [
        (RemoteIndex(pods=1, index_name=f'{MAPPING_INDEX_NAME_PREFIX}-{PodType.P1}',
                     dimension=d, pod_type=PodType.P1, metadata_config={"indexed": ["weather"]}),
         str(PodType.P1))
    ]
)


@dataclasses.dataclass
class RunData:
    ids: Any
    vectors: Any
    metadata: Any
    query_vector: Any


def insert_test_data(index, _n, _d, namespace=''):
    ids = [str(i) for i in range(_n)]
    vectors = [np.random.rand(_d).astype(np.float32).tolist() for _ in range(_n)]
    weather_vocab = ['sunny', 'rain', 'cloudy', 'snowy']
    loop = cycle(weather_vocab)
    metadata = [{"value": i, 'weather': next(loop), "bool_field": i % 2 == 0} for i in range(_n)]
    index.upsert(
        vectors=[
            Vector(id=ids[i], values=vectors[i], metadata=metadata[i])
            for i in range(_n)
        ],
        namespace=namespace
    )
    retry_assert(lambda: index.describe_index_stats().namespaces.get(namespace).vector_count == _n, max_tries=10)

    return ids, vectors, metadata


@pytest.fixture(scope="module")
def test_data(test_metadata_index):
    # Note: relies on grouping strategy –-dist=loadfile to keep different xdist workers
    # from running different tests below with different data/metadata
    index, _ = test_metadata_index
    query_vector = np.random.rand(d).astype(np.float32).tolist()
    ids, vectors, metadata = insert_test_data(index, n, d)
    yield RunData(ids=ids, vectors=vectors, metadata=metadata, query_vector=query_vector)


@pytest.fixture(scope="module")
def test_data_for_mapping(test_metadata_index_with_mapping):
    index = test_metadata_index_with_mapping[0]
    query_vector = np.random.rand(d).astype(np.float32).tolist()
    ids, vectors, metadata = insert_test_data(index, n, d)
    yield RunData(ids=ids, vectors=vectors, metadata=metadata, query_vector=query_vector)


def get_query_results(index, vector, filter):
    return index.query(
        values=vector,
        filter=filter,
        top_k=10,
        include_values=True,
        include_metadata=True
    )


def test_fetch(test_metadata_index, test_data):
    index, _ = test_metadata_index
    ids = test_data.ids
    metadata = test_data.metadata
    fetch_response = index.fetch(ids=[ids[0]])
    fetched_metadata = fetch_response[ids[0]].metadata
    assert fetched_metadata == metadata[0]


def test_delete_eq(test_metadata_index):
    index, _ = test_metadata_index
    namespace = 'test_delete_eq'
    _n = 10
    ids, values, metadata = insert_test_data(index, _n, d, namespace=namespace)
    first_md = metadata[0]['weather']
    matches = [i for i in range(_n) if metadata[i]['weather'] == first_md]

    fetch_response = index.fetch(ids=ids, namespace=namespace)
    fetched_vecs = list(fetch_response.values())
    assert len(fetched_vecs) == len(ids)
    assert all(fetch_response[ids[i]].metadata['weather'] == first_md for i in matches)

    index.delete_by_metadata(namespace=namespace, filter={'weather': first_md})
    retry_assert(
        lambda: index.describe_index_stats().namespaces[namespace].vector_count == len(ids) - len(matches))

    fetch_response = index.fetch(ids=ids, namespace=namespace)
    assert len(fetch_response.values()) == len(ids) - len(matches)
    assert all(vec.metadata.get('weather') != first_md for vec in fetch_response.values())


def test_gt(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    gt_filter = {"value": {"$gt": 10}}
    query_response = get_query_results(index, query_vector, gt_filter)
    result = query_response
    for match in result:
        assert 'value' in match.metadata
        assert match.metadata['value'] > 10


def test_lt(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    lt_filter = {"value": {"$lt": 10}}
    query_response = get_query_results(index, query_vector, lt_filter)
    result = query_response
    for match in result:
        assert match.metadata['value'] < 10


def test_eq(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    eq_filter = {"value": {"$eq": 25}}
    query_response = get_query_results(index, query_vector, eq_filter)
    result = query_response
    for match in result:
        assert match.metadata['value'] == 25


def test_boolean_eq(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    eq_filter = {"bool_field": True}
    query_response = get_query_results(index, query_vector, eq_filter)
    result = query_response
    for match in result:
        assert match.metadata['bool_field']


def test_boolean_ne(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    eq_filter = {"bool_field": False}
    query_response = get_query_results(index, query_vector, eq_filter)
    for match in query_response:
        assert not match.metadata['bool_field']


def test_in(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    in_filter = {"weather": {"$in": ['snowy', 'sunny']}}
    query_response = get_query_results(index, query_vector, in_filter)
    for match in query_response:
        metadata_value = match.metadata['weather']
        assert metadata_value in ['snowy', 'sunny']


def test_nin(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    nin_vals = ['snowy', 'rainy']
    nin_filter = {"weather": {"$nin": nin_vals}}
    query_response = get_query_results(index, query_vector, nin_filter)
    for match in query_response:
        metadata_value = match.metadata['value']
        assert metadata_value not in nin_vals


def test_compound_ne_and_lte(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    cmp_filter = {"$and": [{"weather": {"$ne": "sunny"}}, {"value": {"$lte": 2}}]}
    query_response = get_query_results(index, query_vector, cmp_filter)
    for match in query_response:
        assert 'sunny' != match.metadata["weather"]
        assert match.metadata["value"] <= 2


def test_compound_eq_or_gte(test_metadata_index, test_data):
    index, _ = test_metadata_index
    query_vector = test_data.query_vector
    cmp_filter = {"$or": [{"weather": {"$eq": "snowy"}}, {"value": {"$gte": 5}}]}
    # cmp_filter = {"$or": [{"weather": {"$eq": "snowy"}}, {"value": {"gte": 5}}]}
    query_response = get_query_results(index, query_vector, cmp_filter)
    for match in query_response:
        w = match.metadata["weather"]
        v = match.metadata["value"]
        assert w == "snowy" or v >= 5


def test_update_md(test_metadata_index):
    index, _ = test_metadata_index
    vector_id = 'vec-update'
    values = np.random.rand(d).astype(np.float32).tolist()
    # first write
    old_md = {'value': 11, 'weather': 'chilly'}
    index.upsert(vectors=[Vector(id=vector_id, values=values, metadata=old_md)])
    retry_assert(lambda: list(index.fetch(ids=[vector_id]).values())[0].metadata == old_md)

    # second write
    new_md = {'value': 12, 'weather': 'sunny'}
    index.upsert(vectors=[Vector(vector_id, values=values, metadata=new_md)])
    retry_assert(lambda: list(index.fetch(ids=[vector_id]).values())[0].metadata == new_md)


def test_multiple_values(test_metadata_index, test_data):
    index, _ = test_metadata_index
    ids = test_data.ids
    vectors = test_data.vectors
    query_vector = test_data.query_vector
    # multiple values
    in_vals = ['snowy', 'rainy', 'chilly']
    unique_value = 812312
    md = {'value': unique_value, 'weather': in_vals}
    index.upsert(vectors=[Vector(id=ids[0], values=vectors[0], metadata=md)])
    retry_assert(lambda: list(index.fetch(ids=[ids[0]]).values())[0].metadata == md)

    for val in in_vals:
        eq_filter = {'value': unique_value, 'weather': val}
        query_response = get_query_results(index, query_vector, eq_filter)
        matches = query_response
        assert len(matches) == 1
        assert matches[0].metadata['value'] == unique_value
        assert matches[0].metadata['weather'] == in_vals


# TODO: Fix this test after metadata config is finalized
# def test_metadata_mapping(test_metadata_index_with_mapping, test_data_for_mapping):
#     index = test_metadata_index_with_mapping[0]
#     query_vector = test_data_for_mapping.query_vector

#     # Check for a non-indexd field
#     eq_filter = {"value": {"$eq": 25}}
#     query_response = get_query_results(index, query_vector, eq_filter)
#     result = query_response[0]

#     assert len(result) == 0

#     # Check for an indexed field
#     eq_filter = {"weather": {"$eq": "sunny"}}
#     query_response = get_query_results(index, query_vector, eq_filter)
#     result = query_response[0]
#     print(result)
#     assert len(result) != 0