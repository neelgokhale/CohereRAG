# ../src/chatbot.py

import os
from pprint import pprint

from langchain_cohere import (
    ChatCohere, CohereRagRetriever, CohereEmbeddings, CohereRerank
)
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.base import Document
from dotenv import load_dotenv
load_dotenv()


SYSTEM_MESSAGE = \
    "You are a helpful chatbot that can help answer questions \
    about large language models and related concepts. You will \
    only apply the provided context to answer the questions"
    
prompt_template = PromptTemplate.from_template("""
Answer the following user query using only the context provided as reference.

User query: {query}

Context: {context}
""")


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
        
        self.system_msg = SystemMessage(
            content=SYSTEM_MESSAGE
        )
        
        self.message_log = []
        self.message_memory_window = 10
        
    def setup(self, db: list[Document]):
        self.compression_ret = ContextualCompressionRetriever(
            base_compressor=self.rerank,
            base_retriever=db.as_retriever(search_kwargs={"k": 1})
        )
        
    async def _get_docs(self, query: str, compressed_docs: list[Document]):
        docs = await self.rag.ainvoke(
            query,
            documents=compressed_docs,
        )
        
        return docs
    
    async def _get_response(self):
        ai_msg = await self.llm.ainvoke(input=self.message_log)
        
        return ai_msg
        
    async def query(self, user_query: str, verbose: bool=True):
        
        compressed_docs = self.compression_ret.get_relevant_documents(
            user_query
        )
        
        docs = await self._get_docs(user_query, compressed_docs)
        
        rag_res = docs.pop()
        
        user_msg = HumanMessage(
            content=prompt_template.format(
                query=user_query,
                context=rag_res.page_content
            )
        )
        
        if len(self.message_log) > self.message_memory_window - 1:
            self.message_log.pop(0)
            self.message_log.append(user_msg)
        else:
            self.message_log.append(user_msg)
        
        ai_msg = await self._get_response()
        
        if len(self.message_log) > self.message_memory_window - 1:
            self.message_log.pop(0)
            self.message_log.append(ai_msg)
        else:
            self.message_log.append(ai_msg)
        
        print("USER MESSAGE:")
        print(user_query)
        print()
        print("COHERE CHATBOT:")
        print(ai_msg.content)
        print()
        if verbose:
            print("CITATIONS:")
            for c in rag_res.metadata['citations']:
                print(c)
        
        print("-" * 20)
