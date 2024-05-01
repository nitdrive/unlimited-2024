from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


class QueryService:
    @classmethod
    def query(cls, user_query: str, vector_store):
        # Instantiate VectorStoreIndex object from your vector_store object
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # Grab 5 search results
        retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

        # Pass in your retriever from above, which is configured to return the top 5 results
        query_engine = RetrieverQueryEngine(retriever=retriever)

        user_query += " Include any Sources or Websites related in the response."

        llm_response = query_engine.query(user_query)

        print(llm_response)

        return {
                "user_query": user_query,
                "result": str(llm_response)
            }
