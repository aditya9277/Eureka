import os
import json
import traceback
import pandas as pd 
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data

# importing package from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

#loading environment variable
from dotenv import load_dotenv
load_dotenv()
key=os.getenv("API_OPENAI")
llm=ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo",temperature=0.5)

TEMPLATE="""
Text:{text}
you are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice question for {subject} student in {tone} tone.
Make sure the question are not repeated and check all the question to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON and use it as a guide. \
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt=PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,verbose=True,output_key="quiz")

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject","quiz"],
    template=TEMPLATE2
)

review_chain=LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

# combining both chain
generate_evaluate_chain=SequentialChain(
    chains=[quiz_chain,review_chain],
    input_variables=["text","number","subject","tone","response_json"],
    output_variables=["quiz","review"],
    verbose=True,
)
