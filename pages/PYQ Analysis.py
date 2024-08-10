import os 
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file
import streamlit as st
from langchain.callbacks import get_openai_callback
# from src.mcqgenerator.QuestionGuess import generate_evaluate_chains
from src.mcqgenerator.logger import logging
#######################################################
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
you are an expert probaable question guessor. Given the above text, it is your job to\
create a  {number} probable  question for {subject} student in {tone} tone.
Make sure the question are not repeated and check all the question to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON and use it as a guide. \
{response_json}
"""

quiz_generation_prompt=PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,verbose=True,output_key="quiz")

TEMPLATE2="""
You are an expert QUESTION TOPIC . Given a QUESTION for {subject} students.\
You need to tell the topic of question and give a complete analysis of the question . Only use at max 150 words for complexity analysis. 

Question:
{quiz}

Check from an expert question topic helper of the above question:
"""

quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject","quiz"],
    template=TEMPLATE2
)

review_chain=LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

# combining both chain
generate_evaluate_chains=SequentialChain(
    chains=[quiz_chain,review_chain],
    input_variables=["text","number","subject","tone","response_json"],
    output_variables=["quiz","review"],
    verbose=True,
)
def get_table_data(quiz_str):
    try:
        quiz_dict = json.loads(quiz_str)
        quiz_table_data = []

        for key, value in quiz_dict.items():
            probable_question = value.get("probable Question", "")  # Assuming "probable Question" might not be present in all dictionaries
            correct = value.get("correct", "")
            quiz_table_data.append({"Probable Question": probable_question, "Correct": correct})

        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return None


##################################################################################
with open(r'pages\questionResponse.json',"r") as file:
    RESPONSE_JSON=json.load(file)

st.title("Probable Question Generator")

with st.form("user_inputs"):
    
    uploaded_file=st.file_uploader("Upload a PDF or Text File")

    mcq_counts=st.number_input("No of proable guess question",min_value=3,max_value=50)

    subject=st.text_input("Insert Question Subject",max_chars=20)

    tone=st.text_input("complexity Level of Question",max_chars=20,placeholder="simple")

    button=st.form_submit_button("Create question")

    if button and uploaded_file is not None and mcq_counts and subject and tone:
        with st.spinner("loading...."):
            try:
                text=read_file(uploaded_file)
                # count no of token
                with get_openai_callback() as cb:
                    logging.info("creating response")
                    response=generate_evaluate_chains(
                        {
                            "text":text,
                            "number":mcq_counts,
                            "subject":subject,
                            "tone":tone,
                            "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )
                    logging.info("created resonse")
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("ERROR 🤖")
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                logging.info("Total cost{cb.total_cost}")
                if isinstance(response,dict):
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)
                            st.text_area(label="reviews",value=response["review"])
                        else:
                            st.error("ERROR IN THE TABLE DATA")
                else:
                    st.write(response)



