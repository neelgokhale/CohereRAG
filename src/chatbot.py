# ../src/chatbot.py

import os
from pprint import pprint

from langchain_cohere import (
    ChatCohere, CohereRagRetriever, CohereEmbeddings, CohereRerank
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.base import Document
from dotenv import load_dotenv
load_dotenv()


class Chatbot(object):
    
    def __init__(self,
                 llm: ChatCohere,
                 rag: CohereRagRetriever,
                 embeddings: CohereEmbeddings,
                 rerank: CohereRerank
                 ):
        self.llm = llm
        self.rag = rag
        self.embeddings = embeddings
        self.rerank = rerank
        
    def setup(self, db: list[Document]):
        self.compression_ret = ContextualCompressionRetriever(
            base_compressor=self.rerank,
            base_retriever=db.as_retriever()
        )
        
    def query(self, user_query: str):
        
        compressed_docs = self.compression_ret.get_relevant_documents(
            user_query
        )
        
        docs = self.rag.get_relevant_documents(
            user_query,
            documents=compressed_docs
        )
        
        res = docs.pop()
        
        pprint("Relevant Documents:")
        pprint(docs)
        
        pprint("Answer:")
        pprint(res.page_content)
        pprint(res.metadata['citations'])
