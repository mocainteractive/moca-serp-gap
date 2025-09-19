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
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def belongs_to_site(url: str, site_domain: str) -> bool:
    d = normalize_domain(url)
    return bool(site_domain) and d.endswith(site_domain)

def get_serper_key() -> str:
    k = st.secrets.get("SERPER_API_KEY", "") or os.getenv("SERPER_API_KEY", "")
    if not k:
        st.error("⚠️ Aggiungi la chiave SERPER_API_KEY nei Secrets.")
    return k

def get_semrush_key() -> str:
    k = st.secrets.get("SEMRUSH_API_KEY", "") or os.getenv("SEMRUSH_API_KEY", "")
    if not k:
        st.error("⚠️ Aggiungi la chiave SEMRUSH_API_KEY nei Secrets.")
    return k

def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not key:
        st.error("⚠️ Aggiungi la chiave OPENAI_API_KEY nei Secrets.")
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Errore OpenAI SDK: {e}")
        return None

# =========================
# Serper: Top10 organica IT
# =========================
def serper_top10(query: str, gl: str = "it", hl: str = "it") -> pd.DataFrame:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": get_serper_key(), "Content-Type": "application/json"}
    payload = {"q": query, "gl": gl, "hl": hl}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    organic = data.get("organic", [])
    rows = []
    for i, item in enumerate(organic[:10], start=1):
        rows.append({
            "Pos": i,
            "Titolo": item.get("title", ""),
            "Snippet": item.get("snippet", ""),
            "URL": item.get("link", ""),
        })
    return pd.DataFrame(rows)

# =========================
# SEMrush: keyword per URL
# =========================
def semrush_keywords_by_url(page_url: str, db: str = "it", limit: int = 500) -> pd.DataFrame:
    """
    Usa SEMrush per estrarre le keyword organiche per uno specifico URL.
    Filtra: pos <= 20, volume >= 20. Restituisce colonne base.
    """
    key = get_semrush_key()
    if not key or not page_url:
        return pd.DataFrame()

    endpoint = "https://api.semrush.com/"
    params = {
        "type": "url_organic",           # report per URL
        "key": key,
        "url": page_url,
        "database": db,
        "display_limit": limit,
        # Ph=Keyword, Po=Position, Nq=Volume, Ur=Url
        "export_columns": "Ph,Po,Nq,Ur",
    }

    r = requests.get(endpoint, params=params, timeout=60)
    r.raise_for_status()

    # SEMrush risponde in CSV (separatore ; o ,). Leggiamo in modo robusto.
    raw = r.text.strip()
    if not raw or raw.lower().startswith("error"):
        # Messaggi d'errore SEMrush iniziano spesso con "ERROR"
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(raw), sep=";|,", engine="python")
    # Normalizza nomi colonne possibili
    col_map = {
        "Ph": "Keyword",
        "Po": "Position",
        "Nq": "Volume",
        "Ur": "URL",
        "Keyword": "Keyword",
        "Position": "Position",
        "Volume": "Volume",
        "Url": "URL",
    }
    df = df.rename(columns=col_map)
    # Filtri richiesti
    if "Position" in df.columns:
        df = df[df["Position"] <= 20]
    if "Volume" in df.columns:
        df = df[df["Volume"] >= 20]
    # Ordina
    if "Volume" in df.columns:
        df = df.sort_values(["Position", "Volume"], ascending=[True, False])
    return df.reset_index(drop=True)

# =========================
# OpenAI: estrazione topic
# =========================
def fetch_visible_text(url: str, max_chars: int = 15000) -> str:
    try:
        resp = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # rimuovi blocchi inutili
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:max_chars]
    except Exception:
        return ""

def extract_topics_with_openai(url: str, model: str = "gpt-4o-mini") -> list[str]:
    client = get_openai_client()
    if not client:
        return []
    page_text = fetch_visible_text(url)
    if not page_text:
        return []

    system = "Sei un SEO analyst. Estrai temi e sotto-temi trattati in una pagina web in italiano."
    user = (
        "Testo pagina (estratto, può essere rumoroso):\n\n"
        f"{page_text}\n\n"
        "Restituisci una lista (max 12) di macro-temi concisi, senza spiegazioni."
    )
    try:
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=300,
        )
        content = out.choices[0].message.content.strip()
        # prova a splittare in elenco
        lines = [l.strip("-• ").strip() for l in content.splitlines() if l.strip()]
        # tieni solo righe brevi
        topics = [l for l in lines if len(l) <= 80][:12]
        return topics or [content]
    except Exception:
        return []

# =========================
# UI
# =========================
st.set_page_config(page_title="Moca SERP Gap (IT) — Step 1", layout="wide")
st.title("Moca SERP Gap (IT) — Step 1: Top10 organica")

with st.sidebar:
    st.header("Parametri")
    query = st.text_input("Query di ricerca (IT)", value="frigoriferi da incasso")
    my_domain = st.text_input("Tuo dominio (es. smeg.com)", value="smeg.com")
    my_url = st.text_input("Tua pagina (URL preciso, opzionale)", value="")
    st.divider()
    use_semrush = st.checkbox("Aggiungi keyword per URL (SEMrush)", value=True)
    use_topics = st.checkbox("Estrai temi con AI (OpenAI)", value=True)
    run = st.button("🔎 Analizza SERP")

st.caption(
    "Questo step mostra la **Top10 organica** su Google Italia (Serper). "
    "Opzionalmente arricchisce ogni risultato con **keyword SEMrush** (pos ≤20, vol ≥20) "
    "e con **temi principali** estratti via OpenAI. Analisi keyword-gap su base *pagina → pagina* se fornisci il tuo URL."
)

if run:
    if not query.strip():
        st.warning("Inserisci una query.")
        st.stop()

    try:
        df = serper_top10(query)
    except Exception as e:
        st.error(f"Errore Serper: {e}")
        st.stop()

    if df.empty:
        st.info("Nessun risultato.")
        st.stop()

    # Flag appartenenza dominio
    my_dom_norm = normalize_domain("https://" + my_domain) if my_domain else ""
    df["Dominio"] = df["URL"].apply(normalize_domain)
    df["È il mio sito?"] = df["URL"].apply(lambda u: "✅" if belongs_to_site(u, my_dom_norm) else "—")

    st.subheader("Top 10 organica (IT)")
    st.dataframe(df[["Pos", "È il mio sito?", "Titolo", "Snippet", "URL", "Dominio"]], use_container_width=True)

    # Riepilogo
    mine = (df["È il mio sito?"] == "✅").sum()
    st.markdown(f"**Copertura attuale:** {mine}/10 risultati appartengono a **{my_dom_norm or my_domain}**.")

    # ===== Enrichment per risultato =====
    competitor_kw_union = set()
    my_kw_set = set()

    if use_semrush or use_topics:
        st.subheader("Approfondimenti per risultato")
        for _, row in df.iterrows():
            url = row["URL"]
            with st.expander(f"{row['Pos']}. {row['Titolo']} — {url}"):
                # SEMrush
                if use_semrush:
                    try:
                        kws = semrush_keywords_by_url(url)
                        if not kws.empty:
                            st.markdown(f"**Keyword (pos ≤20, vol ≥20)** — {len(kws)} trovate")
                            st.dataframe(kws[["Keyword", "Position", "Volume"]], use_container_width=True, height=260)
                            # accumula solo se non è la tua pagina
                            if not (my_url and url == my_url):
                                competitor_kw_union.update(kws["Keyword"].astype(str).str.lower().tolist())
                        else:
                            st.info("Nessuna keyword (o limite SEMrush/URL non idoneo).")
                    except Exception as e:
                        st.warning(f"SEMrush errore: {e}")

                # OpenAI Topics
                if use_topics:
                    topics = extract_topics_with_openai(url)
                    if topics:
                        st.markdown("**Temi principali (AI)**")
                        st.write(", ".join(topics))
                    else:
                        st.info("Temi non disponibili (blocco/fetch fallito o limiti pagina).")

        # ===== Keyword-Gap pagina vs top10 =====
        if use_semrush and my_url:
            st.subheader("Keyword Gap (pagina vs Top10)")
            try:
                my_df = semrush_keywords_by_url(my_url)
                if not my_df.empty:
                    my_kw_set = set(my_df["Keyword"].astype(str).str.lower().tolist())
                    gap = sorted(list(competitor_kw_union - my_kw_set))
                    st.markdown(f"**La tua pagina:** {my_url}")
                    st.markdown(f"- Keyword tue (pos ≤20, vol ≥20): **{len(my_kw_set)}**")
                    st.markdown(f"- Keyword competitor aggregate: **{len(competitor_kw_union)}**")
                    st.markdown(f"- **Gap potenziale:** {len(gap)} keyword non coperte dalla tua pagina")
                    if gap:
                        st.dataframe(pd.DataFrame({"Keyword gap": gap}), use_container_width=True, height=300)
                else:
                    st.info("La tua pagina non ha keyword idonee (pos≤20 & vol≥20) o la chiamata è vuota.")
            except Exception as e:
                st.warning(f"SEMrush (tua pagina) errore: {e}")

    # Download CSV base SERP
    st.subheader("Esporta SERP")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Scarica CSV SERP", data=csv, file_name="serp_top10_it.csv", mime="text/csv")
