import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub

#New imports related to RAG
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()
if __name__ == "__main__":
    print("Retrieving...")
    #Embedd the Input query
    embeddings = OpenAIEmbeddings()

    llm = ChatOpenAI()
    # query = 'who is the author of the medium blog post related to vector database?'
    # query = 'what did Chip Huen quoted'
    query = 'when was the blog published?'
    prompt = PromptTemplate.from_template(template=query)
    chain = prompt | llm
    result = chain.invoke({})
    print(result.content)

    vectorStore = PineconeVectorStore(embedding=embeddings,index_name=os.environ["INDEX_NAME"])


    #Download PROMPT for RETRIEVAL
    retrieval_prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    #Format list of document into a prompt and pass to llm
    combine_doc_chain = create_stuff_documents_chain(llm,retrieval_prompt)

    #Retrieve data from the VectorStore (Retriver)
    retrieval_chain = create_retrieval_chain(retriever=vectorStore.as_retriever(), combine_docs_chain=combine_doc_chain)

    #invoke the retrieval chain
    result = retrieval_chain.invoke(input={"input":query})

    print(result["answer"])


    # #  Using LCEL to build the RAG chain
    # template = """
    #     Use the following pieces of context to answer the question at the end.
    #     If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #     Use three sentences maximum and keep the answer as concise as possible. 
    #     Always say "thanks for asking!" at the end of the answer.
      
    #     {context}
        
    #     Question:{question}
        
    #     Helpful Answer:
    #     """

    # custom_rag_prompt = PromptTemplate.from_template(template=template)

    # rag_chain = {"context":vectorStore.as_retriever()}