import streamlit as st

from frontend.api import chat

st.title("💬 LoanSense AI Chat")

if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

question = st.chat_input("Ask about loans...")

if question:

    st.session_state.messages.append({

        "role":"user",

        "content":question

    })

    with st.chat_message("user"):

        st.markdown(question)

    response = chat(question)

    answer = response["answer"]

    st.session_state.messages.append({

        "role":"assistant",

        "content":answer

    })

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.divider()

        st.write("Sources")

        for source in response["sources"]:

            st.write(

                f"Page {source['page']}"

            )

        st.metric(

            "Confidence",

            response["confidence"]

        )
