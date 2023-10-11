
## Pinecone Client (Beta)

> **_⚠️ Warning_**
>
> The client is in **public preview** ("Beta") version.  The API may change prior to launch. For production applications, use the official [Python client](https://github.com/pinecone-io/pinecone-python-client).


Pinecone Client V3 is based on pre-compiled Rust code and gRPC, aimed at improving performance and stability.
Using native gRPC transport, client V3 is able to achieve a 2x-3x speedup for vector upsert over the previous RESTful client versions, as well as a 10-20% speedup for vector query latency.
As the client installation is fully self-contained, it does not require any additional dependencies (e.g. `grpcio`), making it easier to install and use in any Python environment.


## Installation

Install public preview version from pip:
```shell
pip3 install pinecone-client==3.0.0rc2
```

## Building from source and contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## Migrating from Pinecone client V2
If you are migrating from pinecone client V2, here is the most minimal code change required to upgrade to V3:
```python
# Pinecone client V2
import pinecone
pinecone.init() # One time init
...
index = pinecone.Index("example-index")
index.upsert(...)

# Pinecone client V3
from pinecone import Client
pinecone = Client() # This is now a `Client` instance
...
# Unchanged!
index = pinecone.Index("example-index")
index.upsert(...)
```
For more API changes see [CHANGELOG.md](CHANGELOG.md)

# Usage

## Index operations

### Creating a Client instance
The `Client` is the main entry point for index operations like creating, deleting and configuring Pinecone indexes.  
Initializing a `Client` requires your Pinecone API key and a region, which can be passed as either environment variables or as parameters to the `Client` constructor.

```python
import os
from pinecone import Client

# Initialize a client using environment variables
os.environ['PINECONE_API_KEY'] = 'YOUR_API_KEY'
os.environ['PINECONE_REGION'] = 'us-west1-gcp'
client = Client()

# Initialize a client using parameters
client = Client(api_key = 'YOUR_API_KEY', region = 'us-west1-gcp')
```

### Creating an index

The following example creates an index without a metadata configuration.  

By default, all metadata fields are indexed.

```python

from pinecone import Client

client = Client(api_key="YOUR_API_KEY", region="us-west1-gcp")

index = client.create_index("example-index", dimension=1024)
```

If some metadata fields contain data payload such as raw text, indexing these fields would make the Pinecone index less efficient.  In such cases, it is recommended to configure the index to only index specific metadata fields which are used for query filtering.  

The following example creates an index that only indexes the `"color"` metadata field. 

```python
metadata_config = {
    "indexed": ["color"]
}

client.create_index("example-index-2", dimension=1024,
                      metadata_config=metadata_config)
```

#### Listing all indexes

The following example returns all indexes in your project.

```python
active_indexes = client.list_indexes()
```

#### Getting index configuration

The following example returns information about the index `example-index`.

```python
index_description = client.describe_index("example-index")
```

#### Deleting an index

The following example deletes `example-index`.

```python
client.delete_index("example-index")
```

#### Scaling an existing index number of replicas

The following example changes the number of replicas for `example-index`.

```python
new_number_of_replicas = 4
client.scale_index("example-index", replicas=new_number_of_replicas)
```
## Vector operations
### Creating an Index instance
The index object is the entry point for vector operations like upserting, querying and deleting vectors to a given Pinecone index.
```python
from pinecone import Client
client = Client(api_key="YOUR_API_KEY", region="us-west1-gcp")
index = client.get_index("example-index")

# Backwards compatibility
index = client.Index("example-index")
```

#### Printing index statistics

The following example returns statistics about the index `example-index`.

```python
from pinecone import Client

client = Client(api_key="YOUR_API_KEY", region="us-west1-gcp")
index = client.Index("example-index")

print(index.describe_index_stats())
```


#### Upserting vectors

The following example upserts vectors to `example-index`.

```python
from pinecone import Client, Vector, SparseValues
client = Client(api_key="YOUR_API_KEY", region="us-west1-gcp")
index = client.get_index("example-index")

upsert_response = index.upsert(
    vectors=[
        ("vec1", [0.1, 0.2, 0.3, 0.4], {"genre": "drama"}),
        ("vec2", [0.2, 0.3, 0.4, 0.5], {"genre": "action"}),
    ],
    namespace="example-namespace"
)

# Mixing different vector representations is allowed
upsert_response = index.upsert(
    vectors=[
        # Tuples 
        ("vec1", [0.1, 0.2, 0.3, 0.4]),
        ("vec2", [0.2, 0.3, 0.4, 0.5], {"genre": "action"}),
        # Vector objects
        Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
        Vector(id='id3', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4])),
        # Dictionaries
        {'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
        {'id': 'id2', 'values': [1.0, 2.0, 3.0], 'sparse_values': {'indices': [1, 2], 'values': [0.2, 0.4]}},
    ],
    namespace="example-namespace"
)
```

#### Querying an index by a new unseen vector

The following example queries the index `example-index` with metadata
filtering.

```python
query_response = index.query(
    values=[1.0, 5.3, 8.9, 0.5], # values of a query vector
    sparse_values = None, # optional sparse values of the query vector
    top_k=10,
    namespace="example-namespace",
    include_values=True,
    include_metadata=True,
    filter={
        "genre": {"$in": ["comedy", "documentary", "drama"]}
    }
)
```

#### Querying an index by an existing vector ID

The following example queries the index `example-index` for the `top_k=10` nearest neighbors of the vector with ID `vec1`.

```python
query_response = index.query(
    id="vec1",
    top_k=10,
    namespace="example-namespace",
    include_values=True,
    include_metadata=True,
)
```

#### Deleting vectors

```python
# Delete vectors by IDs 
index.delete(ids=["vec1", "vec2"], namespace="example-namespace")

# Delete vectors by metadata filters
index.delete_by_metadata(filter={"genre": {"$in": ["comedy", "documentary", "drama"]}}, namespace="example-namespace")

# Delete all vectors in a given namespace (use namespace="" to delete all vectors in the DEFAULT namespace)
index.delete_all(namespace="example-namespace")
```

#### Fetching vectors by ids

The following example fetches vectors by ID without querying for nearest neighbors.

```python
fetch_response = index.fetch(ids=["vec1", "vec2"], namespace="example-namespace")
```


#### Update vectors

The following example updates vectors by ID.

```python
update_response = index.update(
    id="vec1",
    values=[0.1, 0.2, 0.3, 0.4],
    set_metadata={"genre": "drama"},
    namespace="example-namespace"
)
```
# Performance tuning for upserting large datasets
To upsert an entire dataset of vectors, we recommend using concurrent batched upsert requests. The following example shows how to do this using the `asyncio` library:
```python
import asyncio
from pinecone import Client, Vector

def chunker(seq, batch_size):
    return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

async def async_upload(index, vectors, batch_size, max_concurrent=50):
    sem = asyncio.Semaphore(max_concurrent)
    async def send_batch(batch):
        async with sem:
            return await index.upsert(vectors=batch, async_req=True)
    
    await asyncio.gather(*[send_batch(chunk) for chunk in chunker(vectors, batch_size=batch_size)]) 

# To use it:
client = Client()
index = client.get_index("example-index")
asyncio.run(async_upload(index, vectors, batch_size=100))

# In a jupyter notebook, asyncio.run() is not supported. Instead, use
await async_upload(index, vectors, batch_size=100)  
```

# Limitations

## Code completion and type hints
Due to limitations with the underlying `pyo3` library, code completion and type hints are not available in some IDEs, or might require additional configuration.
- **Jupyter notebooks**: Should work out of the box.
- **VSCode**: Change the [`languageServer`](https://code.visualstudio.com/docs/python/settings-reference#_intellisense-engine-settings) setting to `jedi`.
- **PyCharm**: For the moment, all function signatures would show `(*args, **kwargs)`. We are working on a solution ASAP. (Function docstrings would still show full arguments and type hints).
