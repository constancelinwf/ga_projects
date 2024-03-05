import os
from langchain import hub
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
loader = UnstructuredWordDocumentLoader(r'C:\Users\Admin\OneDrive\Desktop\test.docx')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 6})
prompt = hub.pull('rlm/rag-prompt')
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = '''Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use two sentences maximum and keep the answer as concise as possible.
Always end the responses with a new line that says "Hope this helps! Is there anything else that I can help you with?" at the end of the answer.

{context}

Question: {question}

Helpful Answer:'''
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser() 
)

def get_answer(question):
    final_result = rag_chain.invoke(question)
    return final_result