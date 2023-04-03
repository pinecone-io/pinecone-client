# Changelog

## 3.0.0rc1 - breaking changes!!
**New client architecture based on pre-compiled Rust code and gRPC**
### Breaking changes and migration guide:
#### Client initialization
- The `Client` object is the main entry point for control operations like creating, deleting and configuring pinecone indexes. 
    It replaces the global `pinecone.init()` function from previous versions.    
    For most minimal code change, you can initialize a `Client` instance and use it as a drop-in replacement for the global `pinecone` object (see [README](https://github.com/pinecone-io/pinecone-client#migrating-from-pinecone-client-v2)).
- The `envrionement` parameter for `Client` initialization was renamed `region` instead. Similarly, the `PINECONE_ENVIRONMENT` environment variable was renamed to `PINECONE_REGION` as well.

#### Control plane operations
- The `create_index()` method is mostly unchanged, but a few deprecated or unused parameters were removed: `index_type`, `index_config`
- The `configure_index()` method was removed. Use `scale_index()` instead.

#### Upsert operation
- `index.upsert()` now supports mixing vector represenations in the same batch (see [Upserting vectors](https://github.com/pinecone-io/pinecone-client#upserting-vectors)
- When used with `async_req=True`, `index.upsert()` now returns an `asyncio` coroutine instead of a `concurrent.futures.Future` object. See [Performance tuning](https://github.com/pinecone-io/pinecone-client#performance-tuning-for-upsering-large-datasets) for more details.

#### Query operation
- Querying using an existing vector id was separated into a new method `index.query_by_id()`. The `index.query()` method now only accepts a vector values (and optioanl sparse values).
- The both `index.query()` and `index.query_by_id()` now return a list of `QueryResult` objects. The `QueryResult` object has the following attributes:
    - `id` - the vector id
    - `score` - the ANN score for the given query result
    - `values` - optional vector values if `include_values=True` was passed to the query method
    - `sparse_values` - optional vector sparse values if `include_values=True` was passed to the query method
    - `metadata` - optional vector metadata if `include_metadata=True` was passed to the query method
    - `to_dict()` - a method that returns a dictionary representation of the query result

