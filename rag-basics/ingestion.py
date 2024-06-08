import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


load_dotenv()
if __name__ == "__main__":
    # 1. Load the Documents (Text file)
    loader = TextLoader('mediumBlog.txt', encoding='UTF-8')
    document = loader.load()

    #2.split it into chunks
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")

    #3.Create Embedding form the splitted text documnt

    #Embedding model Initialization
    embeddings = OpenAIEmbeddings()


    #4. Process and Store in Vector DB
    print("Ingestion....")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("##--Finish--##")