import streamlit as st

from api import stats

stats_data = stats()

st.title("📊 Analytics")

c1,c2,c3 = st.columns(3)

c1.metric(

    "Documents",

    stats_data["documents"]

)

c2.metric(

    "Chunks",

    stats_data["chunks"]

)

c3.metric(

    "Bank",

    stats_data["bank"]

)

st.write(

    "Embedding Model:",

    stats_data["embedding_model"]

)

st.write(

    "LLM:",

    stats_data["llm"]

)
