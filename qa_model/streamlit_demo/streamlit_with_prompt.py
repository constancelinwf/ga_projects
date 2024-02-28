# import the necessary libraries
import streamlit as st
import time
import os
import pinecone
from langchain_community.vectorstores import Pinecone as PineConeStore # so that it does not clash with pinecone library
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

#initialise pinecone and openAI
os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "a068f059-a7dc-4b90-859c-a974c7c622b8"
pinecone.Pinecone(api_key = os.environ["PINECONE_API_KEY"])

def rag_model(query):
    llm_model = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)
    embeddings_on_pdf = OpenAIEmbeddings(openai_api_key = os.environ["OPENAI_API_KEY"])
    pinecone_db = PineConeStore.from_existing_index("capstone-project", embeddings_on_pdf)
    retriever = pinecone_db.as_retriever()
    # memory = ConversationBufferMemory(memory_key = "chat_history", max_len = 150, 
    #                                   return_messages = True)
        
    # create prompt template
    prompt = """You are a helpful Assistant who answers users' 
    questions based on evidence given to you. 
    The evidence is from the PDF given to you only. 
    Keep your answer short and to the point. If you do not know the answer, 
    just say I don't know."
    {context}
    Question: {question} """

    prompt = PromptTemplate(template = prompt, input_variables =["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    
    model_chain = RetrievalQA.from_chain_type(llm_model, 
                                    chain_type = "stuff",
                                    chain_type_kwargs = chain_type_kwargs,
                                    retriever = retriever,
                                    return_source_documents = False)

    
    try:
        result = model_chain.run(query)

    except:
        result = "Error"

    return result


# streamlit application layout
col1,col2 = st.columns([8,4])
with col1:
    st.image("ga.png")
with col2: 
    st.image("lightsaber.png")
st.title("Capstone Project: ML, Brain Tumors & MRI :brain:")
st.header("MR Question-Answer Model")
st.write("This is a proof-of-concept question-answer model with retrieval capabilities. :i_love_you_hand_sign:")


# text box for user to type their question in
user_input = st.text_area("Enter Question Here:")

# user press the button after typing question
if st.button("Give me an answer!"):
    if len(user_input) > 0:
        with st.spinner("Please wait while it searches its memories...:nerd_face:"):
            time.sleep(2) # Add timed delay to feedback
            response = rag_model(user_input)
            st.write(response)
    else:
        st.write("Please ask a question or else I do not know what to answer :face_with_head_bandage:")