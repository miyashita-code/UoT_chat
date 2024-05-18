import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


import openai
import streamlit as st
from dotenv import load_dotenv
from chat_modules import Agent


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        #print("OPENAI_API_KEY is set")
        pass
    

    # setup streamlit page
    st.set_page_config(
        page_title="UoT Chat"
    )

    if "prev_option" not in st.session_state:
        st.session_state.prev_option = "gpt4"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agents" not in st.session_state:
        st.session_state.agents = Agent()

        # update chat history
        st.session_state.messages = []

    
    
def main():
    with st.sidebar:
        st.title("UoT Chat")
        
        # model selection
        if option := st.selectbox(
                "Choose Agent",
                ["UoT"],
                index=0,
            ):

            if st.session_state.prev_option != option:
                st.session_state.prev_option = option

                # Update if necessary
                pass


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Agentにメッセージを送る"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            input = {"input": st.session_state.messages[-1]["content"]}  
            contents = {"input": input["input"]}

            with st.spinner(""):
                # Stream
                for s in st.session_state.agents.chain_dict["chain"].stream(contents):        
                    full_response += s
                    message_placeholder.markdown(full_response)

            #message_placeholder.markdown(full_response)
            st.session_state.agents.chain_dict["memory"].save_context(input, {"output": full_response})

    
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    init()


    try:
        main()
    except KeyboardInterrupt:
        st.session_state.socket_client.disconnect()
        sys.exit(1)