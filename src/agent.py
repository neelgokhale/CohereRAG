# ../src/agent.py

import os

from langchain.agents import AgentExecutor, load_tools
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_cohere import (
    ChatCohere, CohereRagRetriever, CohereEmbeddings, CohereRerank
)
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.base import Document

from dotenv import load_dotenv
load_dotenv()

from vectorstore import VectorstoreManager

    
prompt_template = ChatPromptTemplate.from_template("{input}")

SYSTEM_MESSAGE = \
"""
You are an expert who answers the user's question with the most relevant \
datasource. You are equipped with an arxiv search tool and a retriever \
that retrieves information from Cohere LLM University about llm-specific \
topics.
"""


class CohereAgent(object):
    
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
        self.message_memory_window = 4
        
        self.tools = []
        
    def setup(self, db: list[Document], verbose: bool=True):
        # setup retriever tool
        self.db = db
        self._setup_retriever_tool()
        
        # setup arxiv tool
        self._setup_arxiv_tool()
        
        # setup agent
        agent = create_cohere_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt_template
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=verbose
        )
        
    def _setup_retriever_tool(self):
        compression_ret = ContextualCompressionRetriever(
            base_compressor=self.rerank,
            base_retriever=self.db.as_retriever(search_kwargs={"k": 1})
        )
        ret_tool = create_retriever_tool(
            compression_ret,
            name="retriever_tool",
            description="Searches and reranks specific documentation from \
                Cohere LLM University."
        )
        self.tools.append(ret_tool)
                 
    def _setup_arxiv_tool(self):
        arxiv_tool = load_tools(['arxiv'])
        self.tools.extend(arxiv_tool)
        
    async def _get_response(self):
        ai_msg = await self.agent_executor.ainvoke(
            {
                "input": self.message_log,
                "system_message": self.system_msg
            }
        )
        
        return ai_msg
        
    async def query(self, user_query: str):
        
        user_msg = HumanMessage(user_query)
        
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
        
        print("-" * 20)
        print(self.message_log[-1])
        return self.message_log[-1]
