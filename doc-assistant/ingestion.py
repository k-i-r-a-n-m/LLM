import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# separators = ["\n\n", "\n", " ", ""])  --> recursively
def ingest_docs():
    file_path = os.path.abspath("./data/langchain-docs-old")
    embedding = OpenAIEmbeddings()
    index_name = os.environ.get("INDEX_NAME")

    #Document loader
    loader = ReadTheDocsLoader(path=file_path, encoding="UTF-8")
    raw_document = loader.load()

    #initializing text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""])

    documents = text_splitter.split_documents(documents=raw_document)
    print(f"Splitted into {len(documents)} chunks")

    print(f"Inserting {len(documents)} chunks to pinecone...")
    PineconeVectorStore.from_documents(documents, index_name=index_name, embedding=embedding)
    print(f"########---Embedded {len(documents)} chunks to pinecone---#######")




load_dotenv()

if __name__ == "__main__":
    ingest_docs()

    # llm = ChatOpenAI(temperature=0,max_tokens="50")
    # llm.invoke("what is langchain? explain in simple terms!")