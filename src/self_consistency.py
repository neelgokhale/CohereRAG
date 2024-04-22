# ../src/self_consistency.py

import os
import json
import random

from typing import Optional, Coroutine
from langchain_cohere import (
    ChatCohere
)
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

SYSTEM_MESSAGE = \
"""
You are a helpful assistant that serves as a sentiment classifier. For a \
given prompt, you will provide an answer and your reasnoning behind that \
answer. You are to format your output as described in the following \
instructions: \n"
"""


class ClassifyResponse(BaseModel):
    answer: str = Field(
        description="the classified sentiment of the text based on reasoning"
    )
    reasoning: str = Field(
        description="reasoning that the model uses to produce an answer."
    )
    
    @validator("answer")
    def answer_is_valid(cls, field):
        if field.lower() not in ["positive", "negative", "neutral", "na"]:
            raise ValueError("Got unexpected output")
        return field
        

class SelfConsistentClassifier(object):

    def __init__(self,
                 llm: ChatCohere,
                 ):
        self.llm = llm
        
        self.system_msg = SYSTEM_MESSAGE
        
        self.cot_preamble = self._generate_cot_examples()
        
    def _generate_cot_examples(self,
                               n_examples: int=3,
                               path: Optional[os.PathLike]=None) -> str:
        cot_preamble = "Here are some examples: \n"
        
        if path is None:
            path = "./data/classify_data.json"
        
        with open(path, "r") as file:
            data = json.load(file)
            
        samples = random.sample(data, k=n_examples)
        for sample in samples:
            cot_preamble += f"prompt: {sample['prompt']}\n"
            cot_preamble += f"answer: {sample['answer']}\n"
            cot_preamble += f"reasoning: {sample['reasoning']}\n"

        return cot_preamble
    
    def setup(self):
        # setup parser
        self.parser = PydanticOutputParser(pydantic_object=ClassifyResponse)

        # setup prompt template
        self.prompt_template = PromptTemplate(
            template=self.system_msg + "{format_instructions}\n" + \
                self.cot_preamble + "\nprompt: {prompt}",
            input_variables=["prompt"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )
        
        # setup chain
        self.chain = self.prompt_template | self.llm
        
    def _correct_output(self, output: AIMessage) -> ClassifyResponse:
        try:
            output = output.replace("\n", "")
            output = output.replace("```json", "")
            output = output.replace("```", "")

            output = json.loads(output)
            corr_output = \
                ClassifyResponse(
                    answer=output['properties']['answer']['value'],
                    reasoning=output['properties']['reasoning']['value']
                )
        except:
            corr_output = ClassifyResponse(
                answer="NA",
                reasoning=""
            )
        
        return corr_output
    
    async def _sample_classifications(self, prompt: str) -> Coroutine:
        output = await self.chain.ainvoke({
            "prompt": prompt
        })
        
        output = output.content
        
        try:
            output = self.parser.parse(output)
        except:
            output = self._correct_output(output)
            
        return output

    async def classify(self, prompt: str,
                       n_samples: int=10,
                       verbose: bool=True) -> str:
        votes = {
            "Positive": 0,
            "Negative": 0,
            "Neutral": 0
        }
        
        for _ in range(n_samples):
            sample = await self._sample_classifications(
                prompt=prompt
            )
            if verbose:
                print(f"Answer: {sample.answer}\nReasoning: {sample.reasoning}")
            if sample.answer != "NA":
                votes[sample.answer] += 1
        
        final_answer = max(votes, key=votes.get)
        if verbose:
            print(
                f"Final answer: \n{final_answer} with {votes[final_answer]} votes"
            )
            
        return final_answer
