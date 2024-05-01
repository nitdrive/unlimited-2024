import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader


class LLamaDocumentLoader:
    @classmethod
    def directory_loader(cls, input_dir, extensions: List):
        reader = SimpleDirectoryReader(input_dir=input_dir, required_exts=extensions, recursive=True)
        docs = reader.load_data()
        print(docs)
        return docs

    @classmethod
    def website_loader(cls, website_path):
        doc = SimpleWebPageReader(html_to_text=True).load_data([website_path])
        return doc

    @classmethod
    def load_document(cls, file_name, file_path):
        print(f"Loading {file_name} at {file_path}")
        name, extension = os.path.splitext(file_path)
        print(extension)

        if extension == '.html':
            unstructured = UnstructuredReader()
            doc = unstructured.load_data(file=file_path)
        else:
            reader = SimpleDirectoryReader(input_files=[file_path])
            doc = reader.load_data()

        print(doc)
        return doc


class DocumentLoader:
    @classmethod
    # loading PDF, DOCX and TXT files as LangChain Documents
    def load_document(cls, file):
        name, extension = os.path.splitext(file)
        print(f'Loading {file}')
        if extension == '.pdf':
            loader = PyPDFLoader(file)
        elif extension == '.docx':
            loader = Docx2txtLoader(file)
        elif extension == '.txt':
            loader = TextLoader(file)
        elif extension == '.html':
            loader = UnstructuredHTMLLoader(file)
        else:
            print('Document format is not supported!')
            return None

        data = loader.load()
        return data
