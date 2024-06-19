import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI , OpenAIEmbeddings , OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub

#new imports for pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

#new imports for FAISS
from langchain_community.vectorstores import FAISS

load_dotenv()


if __name__ == "__main__":

    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    #extract the file's path
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "./data/ReAct-research-paper.pdf")
    index_path = os.path.join(script_dir,'./data/Faiss-index-react-paper')

    # load the pdf document and split into LIST of docs(each doc contains a page's content) [page-0,page-1...page-n]
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    #Control the chunk size using text_splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator='\n')

    docs = text_splitter.split_documents(documents=documents)



    # Create the vectors and keeps it in the RAM
    vectorstore = FAISS.from_documents(docs,embeddings)

    # To persist the Vector store --> store it in a file
    vectorstore.save_local(index_path)


    # Load the Locally stored VectorStore from the FAISS
    new_vectorstore = FAISS.load_local(index_path,embeddings,allow_dangerous_deserialization=True)

    # Create Document chain
    combine_doc_chain = create_stuff_documents_chain(llm,retrieval_prompt)

    # Crate retrieval chain
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=combine_doc_chain)

    result = retrieval_chain.invoke({"input":'explain react in layman terms!'})

    print(result['answer'])


