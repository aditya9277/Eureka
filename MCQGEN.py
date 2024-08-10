import os 
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging
#StreamlitAPP.py
with open(r'Response.json',"r") as file:
    RESPONSE_JSON=json.load(file)

st.title("MCQ GENERATOR APPLICATION")

with st.form("user_inputs"):
    
    uploaded_file=st.file_uploader("Upload a PDF or Text File")

    mcq_counts=st.number_input("No of Mcq",min_value=3,max_value=50)

    subject=st.text_input("Insert Question Subject",max_chars=20)

    tone=st.text_input("complexity Level of Question",max_chars=20,placeholder="simple")

    button=st.form_submit_button("Create Mcq")

    if button and uploaded_file is not None and mcq_counts and subject and tone:
        with st.spinner("loading...."):
            try:
                text=read_file(uploaded_file)
                # count no of token
                with get_openai_callback() as cb:
                    logging.info("creating response")
                    response=generate_evaluate_chain(
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



