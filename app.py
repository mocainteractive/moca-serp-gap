import os
import io
from collections import Counter
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# ---------- Branding ----------
LOGO_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca_logo-positivo-1.png"
FAVICON_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1.png"

# ---------- Access gate ----------
if "authed" not in st.session_state:
    st.session_state.authed = False

ACCESS_CODE = st.secrets.get("ACCESS_CODE", "")
if ACCESS_CODE and not st.session_state.authed:
    st.sidebar.header("🔒 Accesso")
    code_in = st.sidebar.text_input("Access code", type="password")
    if st.sidebar.button("Entra"):
        if code_in == ACCESS_CODE:
            st.session_state.authed = True
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        else:
            st.sidebar.error("Codice non valido.")
    st.stop()

# ---------- Helpers ----------
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
        st.error("⚠️ Aggiungi SERPER_API_KEY nei Secrets.")
    return k

def get_semrush_key() -> str:
    k = st.secrets.get("SEMRUSH_API_KEY", "") or os.getenv("SEMRUSH_API_KEY", "")
    if not k:
        st.error("⚠️ Aggiungi SEMRUSH_API_KEY nei Secrets.")
    return k

def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not key:
        st.error("⚠️ Aggiungi OPENAI_API_KEY nei Secrets.")
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Errore OpenAI SDK: {e}")
        return None

# ---------- Caching ----------
@st.cache_data(ttl=3600, show_spinner=False)
def serper_top10(query: str, gl: str = "it", hl: str = "it") -> pd.DataFrame:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": get_serper_key(), "Content-Type": "application/json"}
    payload = {"q": query, "gl": gl, "hl": hl}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for i, item in enumerate(data.get("organic", [])[:10], start=1):
        rows.append({
            "Pos": i,
            "Titolo": item.get("title", ""),
            "URL": item.get("link", ""),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def semrush_keywords_by_url(page_url: str, db: str = "it", limit: int = 500) -> pd.DataFrame:
    """Restituisce DF con colonne normalizzate: Keyword, Position, Volume, URL (quanto disponibile)."""
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
    # Rinominazioni robuste
    col_map = {
        "Ph": "Keyword", "Keyword": "Keyword",
        "Po": "Position", "Position": "Position",
        "Nq": "Volume", "Volume": "Volume",
        "Ur": "URL", "Url": "URL"
    }
    df = df.rename(columns=col_map)
    # Filtri se presenti
    if "Position" in df.columns:
        df = df[pd.to_numeric(df["Position"], errors="coerce") <= 20]
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df = df[df["Volume"] >= 20]
    # Ordine se possibile
    sort_cols = [c for c in ["Position", "Volume"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols))
    # Solo colonne note in output
    keep = [c for c in ["Keyword", "Position", "Volume", "URL"] if c in df.columns]
    return df[keep].reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
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
        "Restituisci solo una lista (max 12) di macro-temi, separati da a capo, senza numeri né spiegazioni."
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
        topics = [l.strip("-• ").strip() for l in content.splitlines() if l.strip()]
        # compattiamo/sanifichiamo
        topics = [t[:80] for t in topics][:12]
        return topics
    except Exception:
        return []

# ---------- UI base ----------
st.set_page_config(page_title="Moca SERP gap", page_icon=FAVICON_URL, layout="wide")
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:12px;margin-top:-8px;margin-bottom:12px;">
      <img src="{LOGO_URL}" alt="Moca Interactive" style="height:40px;">
      <h1 style="margin:0;font-weight:800;">SERP gap</h1>
    </div>
    """,
    unsafe_allow_html=True
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

# Tabs
serp_tab, deep_tab, gap_tab, export_tab = st.tabs(["🔍 SERP", "📚 Approfondimenti", "⚔️ Gap", "⬇️ Esporta"])

# ---------- run or restore from session ----------
if run:
    st.session_state.pop("serp_df", None)
    st.session_state.pop("kw_by_url", None)
    st.session_state.pop("topics_by_url", None)

    if not query.strip():
        st.warning("Inserisci una query.")
        st.stop()

    progress = st.progress(0)
    try:
        serp_df = serper_top10(query)
        progress.progress(20)
    except Exception as e:
        st.error(f"Errore Serper: {e}")
        st.stop()

    if serp_df.empty:
        st.info("Nessun risultato.")
        st.stop()

    my_dom_norm = normalize_domain("https://" + my_domain) if my_domain else ""
    serp_df["È il mio sito?"] = serp_df["URL"].apply(lambda u: "✅" if belongs_to_site(u, my_dom_norm) else "—")

    # Salva in sessione
    st.session_state.serp_df = serp_df
    st.session_state.my_dom_norm = my_dom_norm
    st.session_state.query = query
    st.session_state.my_url = my_url
    st.session_state.use_semrush = use_semrush
    st.session_state.use_topics = use_topics

    # Arricchimenti
    kw_by_url = {}
    topics_by_url = {}
    if use_semrush or use_topics:
        with st.spinner("Recupero keyword e temi..."):
            for i, row in serp_df.iterrows():
                url = row["URL"]
                if use_semrush:
                    try:
                        kw_by_url[url] = semrush_keywords_by_url(url)
                    except Exception:
                        kw_by_url[url] = pd.DataFrame()
                if use_topics:
                    try:
                        topics_by_url[url] = extract_topics_with_openai(url)
                    except Exception:
                        topics_by_url[url] = []
                progress.progress(20 + int((i + 1) / len(serp_df) * 60))

    st.session_state.kw_by_url = kw_by_url
    st.session_state.topics_by_url = topics_by_url
    progress.progress(100)
    st.success("Analisi completata ✅")

# Se abbiamo dati in sessione, mostriamo UI senza rigenerare
if "serp_df" in st.session_state:
    serp_df = st.session_state.serp_df.copy()
    my_dom_norm = st.session_state.get("my_dom_norm", "")
    my_url = st.session_state.get("my_url", "")
    use_semrush = st.session_state.get("use_semrush", True)
    use_topics = st.session_state.get("use_topics", True)
    kw_by_url = st.session_state.get("kw_by_url", {})
    topics_by_url = st.session_state.get("topics_by_url", {})

    # ---- SERP tab (pulita) ----
    with serp_tab:
        st.subheader("Top 10 organica (IT)")
        st.dataframe(serp_df[["Pos", "È il mio sito?", "Titolo", "URL"]],
                     use_container_width=True)
        mine = (serp_df["È il mio sito?"] == "✅").sum()
        st.markdown(f"**Copertura attuale:** {mine}/10 risultati appartengono a **{my_dom_norm or my_domain}**.")

    # ---- Approfondimenti per risultato ----
    competitor_kw_union = set()
    competitor_topics_counts = Counter()

    with deep_tab:
        st.subheader("Approfondimenti per risultato")
        for _, row in serp_df.iterrows():
            url = row["URL"]
            title = row["Titolo"] or url
            with st.expander(f"{row['Pos']}. {title} — {url}"):
                # Keywords (robusto: mostra solo colonne presenti)
                if use_semrush:
                    dfk = kw_by_url.get(url, pd.DataFrame())
                    if dfk is not None and not dfk.empty:
                        cols_to_show = [c for c in ["Keyword", "Position", "Volume"] if c in dfk.columns]
                        st.markdown(f"**Keyword (pos ≤20, vol ≥20)** — {len(dfk)} trovate")
                        st.dataframe(dfk[cols_to_show], use_container_width=True, height=260)
                        # aggiorna union competitor
                        if not (my_url and url == my_url):
                            competitor_kw_union.update(dfk["Keyword"].astype(str).str.lower())
                    else:
                        st.info("Nessuna keyword utile per questo URL (o limite SEMrush).")

                # Topics
                if use_topics:
                    topics = topics_by_url.get(url, [])
                    if topics:
                        st.markdown("**Temi principali (AI)**")
                        st.write(", ".join(topics))
                        if not (my_url and url == my_url):
                            competitor_topics_counts.update(t.lower() for t in topics)
                    else:
                        st.info("Temi non disponibili per questo URL.")

    # ---- Gap tab (compatto & per frequenza) ----
    with gap_tab:
        st.subheader("Analisi Gap (pagina vs Top10)")

        # Keyword gap
        if use_semrush and my_url:
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
                                 use_container_width=True, height=260)
            else:
                st.info("La tua pagina non ha keyword idonee (pos≤20 & vol≥20) oppure la chiamata è vuota.")

        # Topic gap
        if use_topics:
            st.markdown("### Topic Gap")
            # Top topic competitor per frequenza (più compatti)
            top_competitor_topics = [t for t, _ in competitor_topics_counts.most_common(15)]
            col1, col2 = st.columns(2)
            with col2:
                st.markdown("**Temi ricorrenti nei competitor (Top 15)**")
                st.write(", ".join(top_competitor_topics) if top_competitor_topics else "—")

            if my_url:
                my_topics = extract_topics_with_openai(my_url)
                with col1:
                    st.markdown("**Temi rilevati nella tua pagina**")
                    st.write(", ".join(my_topics) if my_topics else "—")

                my_topics_set = set(t.lower() for t in my_topics)
                topic_gap = [t for t in top_competitor_topics if t not in my_topics_set]
                st.markdown(f"**Topic non coperti (sui più ricorrenti): {len(topic_gap)}**")
                if topic_gap:
                    st.dataframe(pd.DataFrame({"Topic gap": topic_gap}),
                                 use_container_width=True, height=220)
            else:
                st.info("Per il topic gap, indica la tua pagina (URL preciso).")

    # ---- Export XLSX (tutto in un file) ----
    with export_tab:
        st.subheader("Esporta analisi (XLSX)")

        # Costruiamo fogli multipli
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # SERP
            serp_df[["Pos", "È il mio sito?", "Titolo", "URL"]].to_excel(writer, index=False, sheet_name="SERP")

            # Keywords per URL
            if use_semrush and kw_by_url:
                rows = []
                for url, dfk in kw_by_url.items():
                    if dfk is not None and not dfk.empty:
                        temp = dfk.copy()
                        temp.insert(0, "URL", url)
                        rows.append(temp)
                if rows:
                    pd.concat(rows, ignore_index=True).to_excel(writer, index=False, sheet_name="Keywords_URL")

            # Topics per URL
            if use_topics and topics_by_url:
                rows = []
                for url, topics in topics_by_url.items():
                    if topics:
                        rows.append(pd.DataFrame({"URL": [url]*len(topics), "Topic": topics}))
                if rows:
                    pd.concat(rows, ignore_index=True).to_excel(writer, index=False, sheet_name="Topics_URL")

            # GAP (se calcolabile)
            # Keyword gap
            if use_semrush and my_url:
                my_df = semrush_keywords_by_url(my_url)
                if not my_df.empty:
                    my_kw_set = set(my_df["Keyword"].astype(str).str.lower())
                    kw_gap = sorted(list(competitor_kw_union - my_kw_set))
                    pd.DataFrame({"Keyword gap": kw_gap}).to_excel(writer, index=False, sheet_name="Keyword_Gap")

            # Topic ricorrenti & gap
            if use_topics:
                top_competitor_topics = [t for t, _ in competitor_topics_counts.most_common(30)]
                pd.DataFrame({"Competitor topic (freq)": top_competitor_topics}).to_excel(
                    writer, index=False, sheet_name="Competitor_Topics")
                if my_url:
                    my_topics = extract_topics_with_openai(my_url)
                    my_topics_set = set(t.lower() for t in my_topics)
                    topic_gap = [t for t in top_competitor_topics if t not in my_topics_set]
                    pd.DataFrame({"Topic gap": topic_gap}).to_excel(writer, index=False, sheet_name="Topic_Gap")

        st.download_button(
            "⬇️ Scarica XLSX completo",
            data=output.getvalue(),
            file_name="moca_serp_gap_it.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    with serp_tab:
        st.info("Imposta i parametri a sinistra e premi **Analizza SERP**.")
