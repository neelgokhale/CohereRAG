# ../src/halluc_checker.py

import os

from langchain.agents import AgentExecutor, load_tools
from langchain.tools import BaseTool
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


class HallucinationChecker(object):
    pass


class ArxivChecker(BaseTool):
    name = "arxiv_checker"
    description = """
    This tool helps check if any responses made by the llm are accurate by \
    cross-checking any references to Arxiv-related research papers etc. and \
    using that as factual evidence to validate the response.
    """

    def _run(self, response: str):
        pass
