import os
import requests
import pandas as pd
from urllib.parse import urlparse

import streamlit as st

# ---- Simple access gate (passcode) ----
if "authed" not in st.session_state:
    st.session_state.authed = False

ACCESS_CODE = st.secrets.get("ACCESS_CODE", "")
if ACCESS_CODE:  # attivo solo se c'è un codice nei Secrets
    if not st.session_state.authed:
        st.sidebar.header("🔒 Accesso")
        user_code = st.sidebar.text_input("Access code", type="password")
        if st.sidebar.button("Entra"):
            if user_code == ACCESS_CODE:
                st.session_state.authed = True
                st.experimental_rerun()
            else:
                st.sidebar.error("Codice non valido.")
        st.stop()  # blocca il resto dell’app finché non è autenticato


# -----------------------------
# Config & helpers
# -----------------------------
def get_serper_key() -> str:
    # leggiamo prima da st.secrets, poi da env vars
    k = st.secrets.get("SERPER_API_KEY", "") or os.getenv("SERPER_API_KEY", "")
    if not k:
        st.error("⚠️ Aggiungi la chiave SERPER_API_KEY in Secrets.")
    return k

def normalize_domain(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def belongs_to_site(url: str, site_domain: str) -> bool:
    d = normalize_domain(url)
    return d.endswith(site_domain)

# -----------------------------
# Serper: top10 organica IT
# -----------------------------
def serper_top10(query: str, gl: str = "it", hl: str = "it") -> pd.DataFrame:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": get_serper_key(), "Content-Type": "application/json"}
    payload = {"q": query, "gl": gl, "hl": hl}

    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()

    organic = data.get("organic", [])  # lista di risultati organici
    rows = []
    for i, item in enumerate(organic[:10], start=1):
        rows.append({
            "Pos": i,
            "Titolo": item.get("title", ""),
            "Snippet": item.get("snippet", ""),
            "URL": item.get("link", "")
        })
    return pd.DataFrame(rows)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Moca SERP Gap (IT) — Step 1", layout="wide")
st.title("Moca SERP Gap (IT) — Step 1: Top10 organica")

with st.sidebar:
    st.header("Parametri")
    query = st.text_input("Query di ricerca (IT)", value="frigoriferi da incasso")
    site = st.text_input("Tuo dominio (es. smeg.com)", value="smeg.com")
    run = st.button("🔎 Analizza SERP")

st.caption("Questo step mostra solo la **Top10 organica** su Google Italia tramite Serper. Niente Local Pack/News/FAQ.")

if run:
    if not query.strip():
        st.warning("Inserisci una query.")
        st.stop()

    try:
        df = serper_top10(query)
    except Exception as e:
        st.error(f"Errore chiamando Serper: {e}")
        st.stop()

    if df.empty:
        st.info("Nessun risultato.")
        st.stop()

    # Flag: risultati tuoi vs altri
    site_domain = normalize_domain("https://" + site) if site else ""
    df["Dominio"] = df["URL"].apply(normalize_domain)
    df["È il mio sito?"] = df["URL"].apply(lambda u: "✅" if belongs_to_site(u, site_domain) else "—")

    st.subheader("Top 10 organica (IT)")
    st.dataframe(df[["Pos", "È il mio sito?", "Titolo", "Snippet", "URL", "Dominio"]], use_container_width=True)

    # riepilogo rapido
    mine = (df["È il mio sito?"] == "✅").sum()
    st.markdown(f"**Copertura attuale:** {mine}/10 risultati appartengono a **{site_domain or site}**.")
