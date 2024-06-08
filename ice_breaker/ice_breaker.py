import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == '__main__':
    load_dotenv()
    llm = ChatOpenAI(temperature=0,max_tokens=10)

    aiMessage = llm.invoke('explain langchain in 10 words')
    print("Message:",aiMessage.content)


