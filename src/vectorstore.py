# ../src/vectorstore.py

import os

from typing import Optional

from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter


class VectorstoreManager(object):

    
    def __init__(self):
        self.links = [
            "https://docs.cohere.com/docs/text-embeddings",
            "https://docs.cohere.com/docs/similarity-between-words-and\
                -sentences",
            "https://docs.cohere.com/docs/the-attention-mechanism",
            "https://docs.cohere.com/docs/transformer-models",
        ]
        
        self.chroma_dir_path = "./data"
        
    def setup(self, 
              embeddings: CohereEmbeddings,
              custom_links: Optional[list[str]]=None,
              force_reload: bool=False
              ):
        
        if custom_links:
            self.links = self.links + custom_links
            force_reload = True

        if len(self.chroma_dir_path) == 0 or force_reload:
            
            loader = WebBaseLoader(self.links)
            raw_docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            
            docs = splitter.split_documents(raw_docs)
            
            db = Chroma.from_documents(docs, 
                                    embeddings,
                                    persist_directory=self.chroma_dir_path)
        else:
            db = Chroma(persist_directory=self.chroma_dir_path, 
                        embedding_function=embeddings)

        return db
