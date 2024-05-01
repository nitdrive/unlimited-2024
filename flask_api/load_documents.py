from custom_ingestion_pipeline import CustomIngestionPipeline
from services.loaders import LLamaDocumentLoader
from services.vector_db_service import PineConeService

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

docs = LLamaDocumentLoader.directory_loader(input_dir='./vanguard_etfs', extensions=[".pdf"])

# Get the vector store for this
vector_store = PineConeService.connect(index_name="vanguard-docs")

# Start the ingestion pipeline
CustomIngestionPipeline.ingest_documents(documents=docs, vector_store=vector_store)