import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.prompts import PromptTemplate




load_dotenv()
def run_llm(query:str) -> Any:
    llm = ChatOpenAI(temperature=0)
    embedding = OpenAIEmbeddings()
    retrieval_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    vector_store = PineconeVectorStore.from_existing_index(index_name=os.environ["INDEX_NAME"],embedding=embedding)
    combine_docs_chain = create_stuff_documents_chain(llm,retrieval_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(),combine_docs_chain=combine_docs_chain)
    result = retrieval_chain.invoke({"input": query})
    # print(result['answer'])
    return result

if __name__ == "__main__":
    run_llm("who is hitler?")