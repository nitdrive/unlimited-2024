from typing import List
from llama_index.core.schema import Document
import os

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline

embed_model = OpenAIEmbedding()


class CustomIngestionPipeline:
    @classmethod
    def ingest_documents(cls, documents: List[Document], vector_store):
        try:
            print("Starting document ingestion pipeline")
            pipeline = IngestionPipeline(
                transformations=[
                    SemanticSplitterNodeParser(
                        buffer_size=1,
                        breakpoint_percentile_threshold=95,
                        embed_model=embed_model,
                    ),
                    embed_model,
                ],
                vector_store=vector_store
            )

            # Now we run our pipeline!
            pipeline.run(documents=documents)

            print("Done ingesting documents")
        except Exception as e:
            print("Exception in ingest_documents")
            print(e)
