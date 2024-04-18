# Chatbot with Langchain Cohere

> This project aims to build a simple chatbot leveraging the Langchain Cohere API. 

## Features

- All api functions are handled asynchronously
- Utilizes the Command LLM for response generation.
- Implements a retrieval-augmented generation (RAG) model for generating responses based on retrieved documents.
- Supports embedding-based reranking of responses to improve relevance.
- Integrates with a vector store manager (ChromaDB) for efficient retrieval and management of document embeddings.


### Prerequisites

- Python 3.9
- Libraries used: `langchain-cohere`, `langchain-core`, `cohere`, `python-dotenv`
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```
git clone https://github.com/your_username/CohereRag.git
cd CohereRag
```
2. Obtain the Cohere API key and set it in the environment variables.
3. Set environment variable `COHERE_API_KEY` to the API key from Cohere in a `.env` file in the project root dir

## Usage
Run `main.py` to see a sample usage of the query function and chabot setup
