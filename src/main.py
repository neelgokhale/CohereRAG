# ../src/main.py

import os
import asyncio

from langchain_cohere import (
    ChatCohere, CohereRagRetriever, CohereEmbeddings, CohereRerank
)
from dotenv import load_dotenv
load_dotenv()

from agent import CohereAgent
from vectorstore import VectorstoreManager

def main():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    llm = ChatCohere(cohere_api_key=cohere_api_key)
    rag = CohereRagRetriever(llm=llm)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    rerank = CohereRerank(cohere_api_key=cohere_api_key)
    
    vs = VectorstoreManager()
    db = vs.setup(embeddings=embeddings, force_reload=False)
    
    agent = CohereAgent(
        llm,
        rag,
        embeddings,
        rerank
    )
    agent.setup(db)
    
    query = "Describe multi-headed attention using Cohere's LLM university\
    and find me some research papers from from arxiv about \
    research on multi-headed attention"
    
    asyncio.run(agent.query(query))

if __name__ == "__main__":
    main()
