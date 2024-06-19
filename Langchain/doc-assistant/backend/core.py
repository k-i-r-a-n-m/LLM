import os

from langchain import hub
from dotenv import load_dotenv
from typing import Any,List,Tuple
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from  langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain



load_dotenv()
def run_llm(query:str,chat_history:List[Tuple]=[]) -> Any:
    # Initializing
    llm = ChatOpenAI(temperature=0)
    embedding = OpenAIEmbeddings()
    retrieval_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Initializing VectorDB
    vector_store = PineconeVectorStore.from_existing_index(index_name=os.environ["INDEX_NAME"],embedding=embedding)

    # Chains
    combine_docs_chain = create_stuff_documents_chain(llm,retrieval_prompt)


    chat_retriever_chain = create_history_aware_retriever(
        llm, vector_store.as_retriever(), rephrase_prompt
    )

    # retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(),combine_docs_chain=combine_docs_chain)
    retrieval_chain = create_retrieval_chain(chat_retriever_chain,
                                             combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})

    return result


if __name__ == "__main__":
    chat_history = [("human","who created langchain?"),("ai","harrison chase created it!")]
    run_llm("tell me more about him", chat_history)