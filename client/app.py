# frontend/app.py
import streamlit as st
import requests
from pathlib import Path

# Configure this to match your backend address/port
API_URL = st.secrets.get("api_url", "http://localhost:8000")

st.set_page_config(page_title="Content Moderation", layout="wide")
st.title("PDF Content Moderation (Policy-based)")

st.sidebar.header("Policy Management")
policy_files = st.sidebar.file_uploader("Upload policy PDFs (multiple)", type="pdf", accept_multiple_files=True)
if st.sidebar.button("Upload Policy PDFs") and policy_files:
    files = [("files", (f.name, f.read(), "application/pdf")) for f in policy_files]
    try:
        resp = requests.post(f"{API_URL}/upload_policy/", files=files, timeout=120)
        if resp.status_code == 200:
            st.sidebar.success("Policy PDFs uploaded and stored.")
        else:
            st.sidebar.error(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

if st.sidebar.button("Clear Policy Store"):
    try:
        resp = requests.post(f"{API_URL}/clear_policy/")
        if resp.status_code == 200:
            st.sidebar.success("Policy store cleared.")
        else:
            st.sidebar.error(f"Error clearing: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.sidebar.error(f"Clear failed: {e}")

st.header("Moderate a Document")
st.markdown("Upload the document you want to check against the stored policies. The file is not retained on the server after checking.")

current_file = st.file_uploader("Upload PDF to moderate (single file)", type="pdf")
if st.button("Run Moderation") and current_file:
    try:
        files = {"current_file": (current_file.name, current_file.read(), "application/pdf")}
        with st.spinner("Running moderation..."):
            resp = requests.post(f"{API_URL}/moderate/", files=files, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Moderation Result")
            st.write(f"Verdict: **{data.get('verdict')}**")
            st.write(f"Total chunks analyzed: {data.get('total_chunks')}")
            st.write(f"Allowed chunks: {data.get('allowed_chunks')}")
            violations = data.get("violations", [])
            if not violations:
                st.success("No violations found according to the policy store.")
            else:
                st.error(f"{len(violations)} violation(s) detected")
                for v in violations:
                    st.markdown("---")
                    st.markdown(f"**Chunk ID:** `{v.get('chunk_id')}`")
                    st.markdown(f"**Verdict:** {v.get('verdict')}")
                    st.markdown(f"**Explanation:** {v.get('explanation')}")
                    sources = v.get("sources", [])
                    if sources:
                        st.markdown("**Sources:**")
                        for s in sources:
                            st.markdown(f"- `{s}`")
                    st.markdown("**Chunk Preview:**")
                    st.code(v.get("chunk_text", "")[:1000])
        else:
            st.error(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        st.error(f"Moderation failed: {e}")
