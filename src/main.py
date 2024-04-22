# ../src/main.py

import os
import asyncio

from langchain_cohere import (
    ChatCohere
)
from dotenv import load_dotenv
load_dotenv()

from self_consistency import SelfConsistentClassifier

def main():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    llm = ChatCohere(cohere_api_key=cohere_api_key)

    classifier = SelfConsistentClassifier(
        llm
    )
    classifier.setup()
    
    query = "The old house stood silent and imposing against the backdrop of the setting sun. Its weathered facade hinted at a rich history, whispering tales of bygone days. As I approached, a sense of nostalgia washed over me, mingling with a tinge of apprehension. Yet, amidst the shadows of uncertainty, there lingered a glimmer of hope, a promise of new beginnings"
    
    output = asyncio.run(classifier.classify(query))
    print(output)

if __name__ == "__main__":
    main()
