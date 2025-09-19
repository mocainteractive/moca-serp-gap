import os
import io
import time
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# =========================
# Branding
# =========================
LOGO_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca_logo-positivo-1.png"
FAVICON_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1.png"

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
    key = get_semrush_key()
    if not key or not page_url:
        return pd.DataFrame()

    endpoint = "https://api.semrush.com/"
    params = {
        "type": "url_organic",
        "key": key,
        "url": page_url,
        "database": db,
        "display_limit": limit,
        "export_columns": "Ph,Po,Nq,Ur",
    }
    r = requests.get(endpoint, params=params, timeout=60)
    r.raise_for_status()

    raw = r.text.strip()
    if not raw or raw.lower().startswith("error"):
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(raw), sep=";|,", engine="python")
    col_map = {"Ph": "Keyword", "Po": "Position", "Nq": "Volume", "Ur": "URL",
               "Keyword": "Keyword", "Position": "Position", "Volume": "Volume", "Url": "URL"}
    df = df.rename(columns=col_map)
    if "Position" in df.columns: df = df[df["Position"] <= 20]
    if "Volume" in df.columns:   df = df[df["Volume"] >= 20]
    if "Volume" in df.columns:   df = df.sort_values(["Position", "Volume"], ascending=[True, False])
    return df.reset_index(drop=True)

# =========================
# OpenAI: estrazione topic
# =========================
def fetch_visible_text(url: str, max_chars: int = 15000) -> str:
    try:
        resp = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
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
        lines = [l.strip("-• ").strip() for l in content.splitlines() if l.strip()]
        topics = [l for l in lines if len(l) <= 80][:12]
        return topics or [content]
    except Exception:
        return []

# =========================
# UI
# =========================
st.set_page_config(page_title="Moca SERP Gap (IT) — Step 1", page_icon=FAVICON_URL, layout="wide")
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:12px;margin-top:-8px;margin-bottom:12px;">
      <img src="{LOGO_URL}" alt="Moca Interactive" style="height:40px;">
      <h1 style="margin:0;font-weight:800;">Moca SERP Gap (IT) — Step 1</h1>
    </div>
    """, unsafe_allow_html=True
)

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
    "Mostra la **Top10 organica** su Google Italia (Serper). "
    "Opzionalmente arricchita con **keyword SEMrush** (pos ≤20, vol ≥20) e **temi** via OpenAI. "
    "Confronto **keyword & topic gap** se indichi la tua pagina."
)

serp_tab, deep_tab, gap_tab, export_tab = st.tabs(["🔍 SERP", "📚 Approfondimenti", "⚔️ Gap", "⬇️ Esporta"])

if run:
    if not query.strip():
        st.warning("Inserisci una query.")
        st.stop()

    progress = st.progress(0)

    # 1) SERP
    try:
        df = serper_top10(query)
        progress.progress(20)
    except Exception as e:
        st.error(f"Errore Serper: {e}")
        st.stop()

    if df.empty:
        st.info("Nessun risultato.")
        st.stop()

    my_dom_norm = normalize_domain("https://" + my_domain) if my_domain else ""
    df["Dominio"] = df["URL"].apply(normalize_domain)
    df["È il mio sito?"] = df["URL"].apply(lambda u: "✅" if belongs_to_site(u, my_dom_norm) else "—")

    with serp_tab:
        st.subheader("Top 10 organica (IT)")
        st.dataframe(df[["Pos", "È il mio sito?", "Titolo", "Snippet", "URL", "Dominio"]],
                     use_container_width=True)
        mine = (df["È il mio sito?"] == "✅").sum()
        st.markdown(f"**Copertura attuale:** {mine}/10 risultati appartengono a **{my_dom_norm or my_domain}**.")

    # 2) Arricchimento per URL
    competitor_kw_union = set()
    competitor_topic_union = set()
    my_kw_set = set()
    my_topics_set = set()

    with deep_tab:
        st.subheader("Approfondimenti per risultato")
        with st.spinner("Recupero keyword e temi dalle pagine..."):
            for idx, row in df.iterrows():
                url = row["URL"]
                with st.expander(f"{row['Pos']}. {row['Titolo']} — {url}"):
                    # SEMrush
                    if use_semrush:
                        try:
                            kws = semrush_keywords_by_url(url)
                            if not kws.empty:
                                st.markdown(f"**Keyword (pos ≤20, vol ≥20)** — {len(kws)} trovate")
                                st.dataframe(kws[["Keyword", "Position", "Volume"]],
                                             use_container_width=True, height=260)
                                if not (my_url and url == my_url):
                                    competitor_kw_union.update(kws["Keyword"].astype(str).str.lower())
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
                            if not (my_url and url == my_url):
                                competitor_topic_union.update(t.lower() for t in topics)
                        else:
                            st.info("Temi non disponibili (blocco/fetch fallito o limiti pagina).")

                # avanza progress bar in modo proporzionale
                progress.progress(20 + int((idx + 1) / len(df) * 60))

    # 3) Calcolo gap (keyword + topic)
    with gap_tab:
        st.subheader("Analisi Gap (pagina vs Top10)")

        # Keyword gap
        if use_semrush and my_url:
            try:
                my_df = semrush_keywords_by_url(my_url)
                if not my_df.empty:
                    my_kw_set = set(my_df["Keyword"].astype(str).str.lower())
                    kw_gap = sorted(list(competitor_kw_union - my_kw_set))
                    st.markdown("### Keyword Gap")
                    st.markdown(f"- Tua pagina: **{my_url}**")
                    st.markdown(f"- Keyword tue (pos ≤20, vol ≥20): **{len(my_kw_set)}**")
                    st.markdown(f"- Keyword competitor aggregate: **{len(competitor_kw_union)}**")
                    st.markdown(f"- **Gap potenziale:** {len(kw_gap)} keyword non coperte")
                    if kw_gap:
                        st.dataframe(pd.DataFrame({"Keyword gap": kw_gap}),
                                     use_container_width=True, height=280)
                else:
                    st.info("La tua pagina non ha keyword idonee (pos≤20 & vol≥20) o la chiamata è vuota.")
            except Exception as e:
                st.warning(f"SEMrush (tua pagina) errore: {e}")

        # Topic gap
        if use_topics and my_url:
            st.markdown("### Topic Gap")
            my_topics = extract_topics_with_openai(my_url)
            if my_topics:
                my_topics_set = set(t.lower() for t in my_topics)
                topic_gap = sorted([t for t in competitor_topic_union - my_topics_set])
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Temi rilevati nella tua pagina**")
                    st.write(", ".join(my_topics))
                with cols[1]:
                    st.markdown("**Temi ricorrenti nei competitor (unione)**")
                    st.write(", ".join(sorted(competitor_topic_union)) if competitor_topic_union else "—")
                st.markdown(f"**Topic non coperti:** {len(topic_gap)}")
                if topic_gap:
                    st.dataframe(pd.DataFrame({"Topic gap": topic_gap}),
                                 use_container_width=True, height=280)
            else:
                st.info("Non sono riuscito a estrarre temi dalla tua pagina.")

    # 4) Export
    with export_tab:
        st.subheader("Esporta SERP")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Scarica CSV SERP", data=csv, file_name="serp_top10_it.csv", mime="text/csv")

    progress.progress(100)
    st.success("Analisi completata ✅")
