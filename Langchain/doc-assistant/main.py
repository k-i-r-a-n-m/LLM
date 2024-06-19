import streamlit as st
from streamlit_chat import message

# own import
from backend.core import run_llm

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if __name__ == "__main__":
    st.header("Langchain(v1) -- Documentation Helper Bot")
    prompt = st.text_input("Prompt",placeholder="Enter your prompt here..")

    if prompt:
        with st.spinner("Generating responses..."):
            # import time
            # time.sleep(3)
            generated_response = run_llm(query=prompt, chat_history = st.session_state["chat_history"])
            formatted_answer = f"\n{generated_response["answer"]}"
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(formatted_answer)
            for x in [("human", prompt), ("ai", formatted_answer)]:
                st.session_state["chat_history"].append(x)
            print(generated_response)

    if st.session_state["chat_answer_history"]:
        for user_query, generated_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]):
            message(user_query, is_user=True)
            message(generated_response)





