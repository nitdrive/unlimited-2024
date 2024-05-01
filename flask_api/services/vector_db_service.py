import time

from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import os

from llama_index.vector_stores.pinecone import PineconeVectorStore


class PineConeService:
    @classmethod
    def index_exists(cls, pinecone, index_name):
        for index_info in pinecone.list_indexes():
            if index_info["name"] == index_name:
                print(f"Index with {index_name} exists")
                return True

        print(f"Index with {index_name} does not exist")
        return False

    @classmethod
    def create_index(cls, pinecone, index_name, spec):
        try:
            pinecone.create_index(
                index_name,
                dimension=1536,  # dimensionality of minilm
                metric='cosine',
                spec=spec
            )

            # wait for index to be initialized
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
            return True
        except Exception as e:
            print("Error:")
            print(e)
            return False

    @classmethod
    def connect(cls, index_name):
        pc = PineconeGRPC(api_key=os.environ.get("PINECONE_API_KEY"))
        spec = ServerlessSpec(cloud='aws', region='us-east-1')

        # check if index already exists (it shouldn't if this is first time)
        if not cls.index_exists(index_name=index_name, pinecone=pc):
            # if does not exist, create index
            cls.create_index(pinecone=pc, index_name=index_name, spec=spec)

        # Initialize your index
        pinecone_index = pc.Index(index_name)
        print("Index stats")
        print(pinecone_index.describe_index_stats())

        # Initialize VectorStore
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        return vector_store

