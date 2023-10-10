# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

from evadb.third_party.vector_stores.types import (
    FeaturePayload,
    VectorIndexQuery,
    VectorIndexQueryResult,
    VectorStore,
)
from evadb.utils.generic_utils import try_to_import_milvuslite_client

required_params = ["index_path"]

vector_field_name = "vector_field"

def get_milvuslite_client(index_path: str):
    global _pymilvus_client_instance
    if _pymilvus_client_instance is None:
        try_to_import_milvuslite_client()

        from milvus import default_server   # noqa: F401
        from pymilvus import connections, utility, Collection   # noqa: F401
           
        default_server.start()
        default_server.set_base_dir(index_path)

        if connections.has_connection("default"):
            connections.disconnect("default")
        
        connections.connect(host='localhost', port=default_server.listen_port)


class MilvusVectorStore(VectorStore):
    def __init__(self, index_name: str, index_path: str) -> None:
        get_milvuslite_client(index_path)
        self._collection_name = index_name

    def create(self, vector_dim: int):
        from pymilvus import (
            FieldSchema, CollectionSchema, DataType,
        )
        field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True)
        field2 = FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=vector_dim, is_primary=False)
        schema = CollectionSchema(fields=[field1, field2], description="collection description")
        collection = Collection(name=self._collection_name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})
        self._client = collection

    def add(self, payload: List[FeaturePayload]):
        ids = [str(row.id) for row in payload]
        embeddings = [row.embedding.reshape(-1).tolist() for row in payload]

        Collection(self._collection_name).insert([ids, embeddings])


    def delete(self) -> None:
        utility.drop_collection(
            collection_name=self._collection_name
        )

    def query(
        self,
        query: VectorIndexQuery,
    ) -> VectorIndexQueryResult:
        
        # create index 
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }

        Collection(self._collection_name).create_index("embeddings", index)

        # load data into memory
        Collection(self._collection_name).load()

        # search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        response = Collection(self._collection_name).search(
            data=query.embedding.reshape(-1).tolist(), 
            anns_field=vector_field_name, 
            param=search_params, 
            limit=query.top_k,
        )

        distances, ids = [], []
        for hit in response[0]:
            ids.append(int(hit.id))
            distances.append(hit.distance)

        # release the memory to reduce memory usage
        Collection(self._collection_name).release()

        return VectorIndexQueryResult(distances, ids)
