import os
import io
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# =========================
# Access gate (passcode)
# =========================
if "authed" not in st.session_state:
    st.session_state.authed = False

ACCESS_CODE = st.secrets.get("ACCESS_CODE", "")
if ACCESS_CODE and not st.session_state.authed:
    st.sidebar.header("🔒 Accesso")
    code_in = st.sidebar.text_input("Access code", type="password")
    if st.sidebar.button("Entra"):
        if code_in == ACCESS_CODE:
            st.session_state.authed = True
            st.experimental_rerun()
        else:
            st.sidebar.error("Codice non valido.")
    st.stop()

# =========================
# Helpers & config
# =========================
def normalize_domain(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.
