from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

"""
Chain that combines documents by stuffing into context.
Create a chain for passing a list of Documents to a model.
"""

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [("system", "What are everyone's favorite colors:\\n\\n{context}")]
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = create_stuff_documents_chain(llm, prompt)

docs = [
    Document(page_content="Jesse loves red but not yellow"),
    Document(page_content = "Jamal loves green but not as much as he loves orange")
]

print(chain.invoke({"context": docs}))