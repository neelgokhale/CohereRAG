# Chatbot with Langchain Cohere

> This project aims to build a simple chatbot leveraging the Langchain Cohere API. 

## Features

- All api functions are handled asynchronously
- Utilizes the Command LLM for response generation.
- Implements a retrieval-augmented generation (RAG) model for generating responses based on retrieved documents.
- Supports embedding-based reranking of responses to improve relevance.
- Integrates with a vector store manager (ChromaDB) for efficient retrieval and management of document embeddings.
- [NEW] cohere agent class `CohereAgent` that leverages 2 tools:
    - `retriever_tool` - Cohere RAG retriever combined with the ReRanker
    - `arxiv` - searches arxiv database for research on LLM-related topics
- [NEW] a classifier class `SelfConsistentClassifier` that applies self-consistency to determine the overall sentiment of the query that is passed from `POSITIVE`, `NEGATIVE` or `NEUTRAL`.
    - samples a defined number of examples from `data\classify_data.json` for chain-of-thought generation
    - contains a `PydanticOutputParser` implementation to add structure to the output of the classifier
    - contains a function to manually apply more validations on an incorrectly intrepreted response structure

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
