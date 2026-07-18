import streamlit as st

from frontend.api import knowledge

from frontend.api import documents

st.title("📚 Knowledge Base")

kb = knowledge()

st.metric(

    "Chunks",

    kb["chunks"]

)

st.write(

    "Embedding",

    kb["embedding_model"]

)

st.write(

    "Re-ranker",

    kb["reranker"]

)

st.divider()

docs = documents()

st.subheader(

    "Uploaded PDFs"

)

for file in docs["documents"]:

    st.write(file)
