from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_core.runnables import RunnablePassthrough , RunnableLambda

load_dotenv()

if __name__ == "__main__":
    llm = ChatOpenAI()
    # prompt = PromptTemplate.from_template("Answer based on the given question.\nquestion:{question} \ncontext:{context}")
    # print(prompt.format(question='how r you',context="dummy context"))

    prompt = PromptTemplate(template="Answer based on the given question.\nquestion:{question} \ncontext:{context}",input_variables=["question","context"])
    # print(prompt.invoke({"input":{"question":'Q!',"context":'c1'}}))  #error

    # chain = {"context":lambda x:x['context'],"question":lambda x:x['question']} | prompt

    # print(chain.invoke({"context":'context-dummy-1',"question":'question-1?'}))

    def context_retriever(_):
        return "--context dummy--"

    chainWithRunablePassthrough = {"context":RunnableLambda(context_retriever),"question":RunnablePassthrough()} | prompt

    print(chainWithRunablePassthrough.invoke('how ar you?'))
