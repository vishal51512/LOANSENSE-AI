import streamlit as st
from frontend.api import upload

st.title("📄 Upload PDF")

pdf = st.file_uploader(
    "Choose PDF",
    type=["pdf"]
)

if pdf:

    if st.button("Upload"):

        with st.spinner("Uploading..."):

            result = upload(pdf)

        st.write(result)

        if "message" in result:

            st.success(result["message"])

            st.write(
                f"Chunks Added: {result['chunks_added']}"
            )

        elif "detail" in result:

            st.error(result["detail"])

        else:

            st.error("Unexpected response from server.")
