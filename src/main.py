# ../src/main.py

import os

from langchain_cohere import (
    ChatCohere, CohereRagRetriever, CohereEmbeddings, CohereRerank
)
from dotenv import load_dotenv
load_dotenv()

from chatbot import Chatbot
from vectorstore import VectorstoreManager

def main():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    llm = ChatCohere(cohere_api_key=cohere_api_key)
    rag = CohereRagRetriever(llm=llm)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    rerank = CohereRerank(cohere_api_key=cohere_api_key)
    
    vs = VectorstoreManager()
    db = vs.setup(embeddings=embeddings, force_reload=False)
    
    chatbot = Chatbot(
        llm,
        rag,
        embeddings,
        rerank
    )
    chatbot.setup(db)
    
    query = "what is single headed attention"
    
    chatbot.query(query)

if __name__ == "__main__":
    main()
