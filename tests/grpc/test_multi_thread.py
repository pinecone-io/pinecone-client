import sys
from typing import Optional

import pinecone
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing 
import pytest
from pinecone import Vector
import pinecone 
from ..utils.utils import index_fixture_factory
from ..utils.remote_index import RemoteIndex, PodType
import numpy as np
import random 
import time

from ..unit.test_data_plane import get_test_data

BATCH_SIZE = 100
DIMENSION = 128
TOP_K = 10

test_mutli_thread = index_fixture_factory(
    [
         (RemoteIndex(pods=2, index_name=f'test-multi-thread-{PodType.P1}',
                     dimension=DIMENSION, pod_type=PodType.P1), str(PodType.P1))
    ]
)

def get_random_vectors(num, dim):
    return [Vector(np.random.rand(dim).astype(np.float32).tolist()) for _ in range(num)]

def make_upsert(index):
    response = index.upsert(vectors=get_test_data(BATCH_SIZE, dimension=DIMENSION))
    return response

def make_query(index):
    return index.query(values=np.random.rand(DIMENSION).astype(np.float32).tolist(),top_k=TOP_K)

def check_upsert_results(results, num_expected):
    total_vectors = 0
    for res in results:
        total_vectors += res.result().upserted_count
    assert total_vectors == num_expected


def test_multithreaded_upsert(test_mutli_thread):
    index, _ = test_mutli_thread
    num_threads = 10
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = []
        for _ in range(num_threads):
            results.append(executor.submit(make_upsert, index))
        check_upsert_results(results, BATCH_SIZE * num_threads)


def test_multithreaded_query(test_mutli_thread):
    index, _ = test_mutli_thread
    num_threads = 10
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = []
        for _ in range(num_threads):
            results.append(executor.submit(make_query, index))
        for res in results:
            assert len(res.result()) == TOP_K


@pytest.mark.skip(reason="Multiprocessing using single Index object is currently not supported")
def test_multiprocess_query(test_mutli_thread):
    index, _ = test_mutli_thread
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = []
        for _ in range(num_processes):
            results.append(executor.submit(make_query, index))
        for res in results:
            assert len(res.result()) == TOP_K

@pytest.mark.skip(reason="Multiprocessing using single Index object is currently not supported")
def test_multiprocess_upsert(test_mutli_thread):
    index, _ = test_mutli_thread
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = []
        for _ in range(num_processes):
            results.append(executor.submit(make_upsert, index))
        check_upsert_results(results, BATCH_SIZE * num_processes)

def create_client_and_make_grpc_call(index_name, client: Optional[pinecone.Client] = None):
    if client is None:
        client = pinecone.Client()
    new_index = client.get_index(index_name)
    make_query(new_index)

@pytest.mark.parametrize("method", ["threads", "processes"])
def test_simultaneous_connections(test_mutli_thread, method):
    index, _ = test_mutli_thread
    test_name = index.name
    num_indexes = 10 if method == "processes" else multiprocessing.cpu_count()

    pool_class = ThreadPoolExecutor if method == "threads" else ProcessPoolExecutor
    with pool_class(max_workers=num_indexes) as executor:
        results = []
        for _ in range(num_indexes):
            executor.submit(create_client_and_make_grpc_call, test_name)
        for res in results:
            assert len(res.result()) == TOP_K

@pytest.mark.parametrize("method", ["threads", "processes"])
def test_simultaneous_connections_existing_client(test_mutli_thread, method):
    if sys.platform == 'darwin' and method == "processes":
        pytest.skip("Passing Client object to ProcessPool is not supported on MacOS")

    index, _ = test_mutli_thread
    test_name = index.name
    joint_client = pinecone.Client()
    num_indexes = 10 if method == "processes" else multiprocessing.cpu_count()

    pool_class = ThreadPoolExecutor if method == "threads" else ProcessPoolExecutor
    with pool_class(max_workers=num_indexes) as executor:
        results = []
        for _ in range(num_indexes):
            executor.submit(create_client_and_make_grpc_call, test_name, client=joint_client)
        for res in results:
            assert len(res.result()) == TOP_K

@pytest.mark.slow
def test_long_standing_connection(test_mutli_thread):
    index, _ = test_mutli_thread
    make_upsert(index)
    time.sleep(2 * 60 * 60)
    make_query(index)


@pytest.mark.slow
def test_random_delays_between_calls(test_mutli_thread):
    num_calls = 10
    min_delay = 0.1  # seconds
    # gloo will send back fin around this mark if the connection is idle
    max_delay = 300.0  # seconds
    index, _  = test_mutli_thread
    for _ in range(num_calls):
        make_query(index)
        delay = random.uniform(min_delay, max_delay)
        # Add some sporadic longer delays
        if random.random() < 0.2:
            delay += 15.0 * 60.0
        time.sleep(delay)
