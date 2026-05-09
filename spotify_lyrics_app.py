"""
Spotify + Letras — Web App con Streamlit
=========================================
Permite explorar los álbumes de estudio de Radiohead (o cualquier artista
en el dataset de Kaggle), obtener letras via lyrics.ovh y analizar:
  - WordCloud del álbum
  - Análisis de sentimiento (polaridad / subjetividad)
  - Palabras más frecuentes
  - Correlación letra ↔ audio features

Ejecutar:
    streamlit run spotify_lyrics_app.py

Dependencias:
    pip install streamlit wordcloud textblob plotly pandas matplotlib requests scikit-learn
"""

import os
import re
import ast
import time
import requests
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

import streamlit as st

# ─── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify + Letras · Radiohead",
    page_icon="🎵",
    layout="wide",
)

# ─── Constantes ────────────────────────────────────────────────────────────────
STUDIO_ALBUMS_RADIOHEAD = {
    'Pablo Honey': 1993,
    'The Bends': 1995,
    'OK Computer': 1997,
    'Kid A': 2000,
    'Amnesiac': 2001,
    'Hail to the Thief': 2003,
    'In Rainbows': 2007,
    'The King of Limbs': 2011,
    'A Moon Shaped Pool': 2016,
}
ALBUM_ORDER = sorted(STUDIO_ALBUMS_RADIOHEAD, key=STUDIO_ALBUMS_RADIOHEAD.get)

AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy',
    'instrumentalness', 'liveness', 'valence',
]

STOP = set(STOPWORDS)
STOP.update(['like', 'know', 'just', 'got', "don't", "i'm", "it's",
             "you're", "we're", "let", 'oh', 'yeah', 'gonna', 'get',
             'go', 'come', 'one', 'two', 'see', 'say', 'said', 'us',
             'could', 'would', 'will', "can't", 'tell', 'well', 'want',
             'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
             'this', 'that', 'with', 'have', 'from', 'they', 'been'])


# ─── Helpers ───────────────────────────────────────────────────────────────────

def artist_id_in_list(artist_ids_str: str, target_id: str) -> bool:
    try:
        return target_id in ast.literal_eval(str(artist_ids_str))
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_and_filter_data(data_path: str, artist_id: str) -> pd.DataFrame:
    """Carga el CSV y filtra por artista + álbumes de estudio."""
    df = pd.read_csv(data_path)
    mask = df['artist_ids'].apply(lambda x: artist_id_in_list(x, artist_id))
    adf = df[mask].copy()
    adf['short_album_name'] = adf['album'].str.split('(').str[0].str.strip()
    adf = adf[adf['short_album_name'].isin(STUDIO_ALBUMS_RADIOHEAD)]
    adf = (
        adf.sort_values('year')
        .drop_duplicates(subset=['short_album_name', 'name'], keep='first')
        .reset_index(drop=True)
    )
    return adf


@st.cache_data(show_spinner=False)
def get_lyrics_cached(artist: str, title: str) -> str | None:
    """Llama a lyrics.ovh y devuelve la letra o None."""
    url = f"https://api.lyrics.ovh/v1/{artist.lower()}/{title.lower()}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json().get('lyrics')
    except Exception:
        pass
    return None


def tokenize(text: str) -> list:
    if not isinstance(text, str):
        return []
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOP and len(w) > 2]


def analyze_sentiment(text: str):
    if not isinstance(text, str) or not text.strip():
        return None, None
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎵 Configuración")
st.sidebar.markdown("---")

data_file = st.sidebar.text_input(
    "Ruta al CSV de Spotify",
    value="tracks_features.csv",
    help="tracks_features.csv del dataset de Kaggle",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros de álbum**")
selected_albums = st.sidebar.multiselect(
    "Álbumes a mostrar",
    options=ALBUM_ORDER,
    default=ALBUM_ORDER,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Análisis de letras**")
fetch_lyrics = st.sidebar.checkbox("Obtener letras via lyrics.ovh", value=False,
                                    help="Requiere conexión a internet. Puede tardar varios minutos.")
max_songs_lyrics = st.sidebar.slider(
    "Máx. canciones con letra a descargar", 5, 100, 30,
    disabled=not fetch_lyrics
)

# ─── Main ──────────────────────────────────────────────────────────────────────
st.title("🎵 Spotify + Análisis de Letras — Radiohead")
st.markdown(
    "Exploración de los **9 álbumes de estudio** de Radiohead: "
    "audio features de Spotify + análisis de letras via [lyrics.ovh](https://lyrics.ovh)."
)
st.markdown("---")

# ── Carga de datos ──────────────────────────────────────────────────────────────
if not os.path.exists(data_file):
    import gdown
    with st.spinner("Descargando dataset de Spotify (~500MB)..."):
        gdown.download(
            "https://drive.google.com/uc?id=1jsXTNtGhOrsCApQctYx-hRxAQASAcPlI",
            data_file, quiet=False
        )

with st.spinner("Cargando y filtrando datos..."):
    TARGET_ARTIST_ID = '4Z8W4fKeB5YxbusRsdQVPb'
    artist_df = load_and_filter_data(data_file, TARGET_ARTIST_ID)
    artist_df = artist_df[artist_df['short_album_name'].isin(selected_albums)]
    local_order = [a for a in ALBUM_ORDER if a in selected_albums]

st.success(f"✅ {len(artist_df)} canciones cargadas de {len(selected_albums)} álbum(es) de estudio.")

# ── Métricas rápidas ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Canciones", len(artist_df))
col2.metric("Álbumes", len(selected_albums))
col3.metric("Valence media", f"{artist_df['valence'].mean():.3f}")
col4.metric("Energy media", f"{artist_df['energy'].mean():.3f}")

st.markdown("---")

# ── Tab layout ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Audio Features", "📝 Letras & Sentimiento", "☁️ WordCloud", "🔍 Buscar Canción"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Audio Features
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Distribución de Audio Features por Álbum")

    feature = st.selectbox("Feature a explorar", AUDIO_FEATURES, index=4)  # valence default

    fig_box = px.box(
        artist_df.assign(
            short_album_name=pd.Categorical(artist_df['short_album_name'],
                                            categories=local_order, ordered=True)
        ).sort_values('short_album_name'),
        x='short_album_name', y=feature,
        color='short_album_name',
        category_orders={'short_album_name': local_order},
        points='all',
        hover_data=['name'],
        title=f'Distribución de {feature} por Álbum',
        labels={'short_album_name': 'Álbum', feature: feature},
        template='plotly_white',
        height=500,
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Scatter: Acousticness vs Valence")
    fig_sc = px.scatter(
        artist_df.assign(
            short_album_name=pd.Categorical(artist_df['short_album_name'],
                                            categories=local_order, ordered=True)
        ).sort_values('short_album_name'),
        x='valence', y='acousticness',
        color='short_album_name',
        size='duration_ms', size_max=25,
        hover_data=['name'],
        category_orders={'short_album_name': local_order},
        title='Acousticness vs Valence (tamaño = duración)',
        labels={'short_album_name': 'Álbum'},
        template='plotly_white',
        height=520,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Radar — Perfil Sonoro por Álbum")
    RADAR_FEATS = ['acousticness', 'danceability', 'energy',
                   'instrumentalness', 'liveness', 'valence']
    album_means = artist_df.groupby('short_album_name')[RADAR_FEATS].mean()
    album_norm = (album_means - album_means.min()) / (album_means.max() - album_means.min() + 1e-9)

    N = len(RADAR_FEATS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    fig_r, ax_r = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(local_order)))

    for album, color in zip(local_order, colors):
        if album not in album_norm.index:
            continue
        row = album_norm.loc[album]
        vals = row.tolist() + row.tolist()[:1]
        ax_r.plot(angles, vals, color=color, linewidth=1.8, label=album)
        ax_r.fill(angles, vals, color=color, alpha=0.07)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(RADAR_FEATS, size=10)
    ax_r.set_title('Audio Features por Álbum (normalizados)', size=12, pad=22)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15), fontsize=8)
    st.pyplot(fig_r, use_container_width=False)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Letras & Sentimiento
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Análisis de Sentimiento de las Letras")

    if 'lyrics' not in artist_df.columns:
        artist_df['lyrics'] = None

    if fetch_lyrics:
        songs_to_fetch = artist_df[artist_df['lyrics'].isna()].head(max_songs_lyrics)
        if len(songs_to_fetch) > 0:
            progress = st.progress(0, text="Obteniendo letras...")
            for i, (idx, row) in enumerate(songs_to_fetch.iterrows()):
                lyrics = get_lyrics_cached('radiohead', row['name'])
                artist_df.at[idx, 'lyrics'] = lyrics
                progress.progress((i + 1) / len(songs_to_fetch),
                                   text=f"[{i+1}/{len(songs_to_fetch)}] {row['name']}")
                time.sleep(0.3)
            progress.empty()
            st.success(f"✅ {artist_df['lyrics'].notna().sum()} canciones con letra disponible.")

    if artist_df['lyrics'].notna().sum() == 0:
        st.info("ℹ️ No hay letras cargadas. Activa **'Obtener letras via lyrics.ovh'** en el sidebar.")
    else:
        # Calcular sentimiento
        artist_df[['polarity', 'subjectivity']] = artist_df['lyrics'].apply(
            lambda x: pd.Series(analyze_sentiment(x))
        )

        sentiment_by_album = (
            artist_df.groupby('short_album_name')[['polarity', 'subjectivity']]
            .mean().round(3).loc[[a for a in local_order if a in artist_df['short_album_name'].values]]
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig_pol = px.bar(
                sentiment_by_album.reset_index(),
                x='polarity', y='short_album_name',
                orientation='h',
                color='polarity',
                color_continuous_scale='RdYlGn',
                title='Polaridad Media de Letras por Álbum',
                labels={'polarity': 'Polaridad', 'short_album_name': 'Álbum'},
                template='plotly_white',
                height=400,
            )
            fig_pol.add_vline(x=0, line_dash='dash', line_color='gray')
            st.plotly_chart(fig_pol, use_container_width=True)

        with col_b:
            fig_sub = px.bar(
                sentiment_by_album.reset_index(),
                x='subjectivity', y='short_album_name',
                orientation='h',
                color='subjectivity',
                color_continuous_scale='Blues',
                title='Subjetividad Media de Letras por Álbum',
                labels={'subjectivity': 'Subjetividad', 'short_album_name': 'Álbum'},
                template='plotly_white',
                height=400,
            )
            st.plotly_chart(fig_sub, use_container_width=True)

        st.subheader("Correlación: Polaridad de Letras vs Audio Features")
        df_corr = artist_df.dropna(subset=['polarity', 'valence'])
        fig_corr = px.scatter(
            df_corr,
            x='valence', y='polarity',
            color='short_album_name',
            size='energy', size_max=20,
            hover_data=['name'],
            trendline='ols',
            title='Valence (audio) vs Polaridad de Letras',
            labels={'valence': 'Valence', 'polarity': 'Polaridad letras',
                    'short_album_name': 'Álbum'},
            template='plotly_white',
            height=500,
        )
        fig_corr.add_hline(y=0, line_dash='dash', line_color='lightgray')
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Evolución temporal del sentimiento")
        album_ts = artist_df.groupby('short_album_name').agg(
            año=('year', 'min'),
            polaridad=('polarity', 'mean'),
            valence=('valence', 'mean'),
        ).loc[[a for a in local_order if a in artist_df['short_album_name'].values]].reset_index()

        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('Polaridad de Letras', 'Valence de Audio'))
        fig_ts.add_trace(go.Scatter(
            x=album_ts['short_album_name'], y=album_ts['polaridad'],
            mode='lines+markers', name='Polaridad letras',
            line=dict(color='#d62728', width=2), marker=dict(size=9)
        ), row=1, col=1)
        fig_ts.add_trace(go.Scatter(
            x=album_ts['short_album_name'], y=album_ts['valence'],
            mode='lines+markers', name='Valence audio',
            line=dict(color='#1f77b4', width=2), marker=dict(size=9)
        ), row=2, col=1)
        fig_ts.update_layout(height=500, template='plotly_white',
                              title='Evolución emocional a lo largo de la discografía')
        fig_ts.add_hline(y=0, line_dash='dash', line_color='lightgray', row=1)
        st.plotly_chart(fig_ts, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: WordCloud
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("WordCloud por Álbum")

    if 'lyrics' not in artist_df.columns or artist_df['lyrics'].notna().sum() == 0:
        st.info("ℹ️ Activa la descarga de letras en el sidebar para ver los WordClouds.")
    else:
        album_sel = st.selectbox("Selecciona un álbum", local_order, key='wc_album')
        sub_lyrics = artist_df[artist_df['short_album_name'] == album_sel]['lyrics'].dropna()
        text_all = ' '.join(sub_lyrics.tolist())

        col_wc, col_freq = st.columns([2, 1])

        with col_wc:
            if text_all.strip():
                wc = WordCloud(
                    width=800, height=500,
                    background_color='white',
                    stopwords=STOP,
                    colormap='tab10',
                    max_words=100,
                    min_font_size=10,
                ).generate(text_all)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title(f'WordCloud — {album_sel}', fontsize=14)
                st.pyplot(fig_wc)
            else:
                st.warning("No hay letras para este álbum.")

        with col_freq:
            st.markdown("**Top 20 palabras**")
            all_words = []
            for text in sub_lyrics:
                all_words.extend(tokenize(text))
            counter = Counter(all_words).most_common(20)
            if counter:
                words, counts = zip(*counter)
                fig_freq = px.bar(
                    x=list(counts)[::-1], y=list(words)[::-1],
                    orientation='h',
                    title=f'Top palabras — {album_sel}',
                    labels={'x': 'Frecuencia', 'y': ''},
                    template='plotly_white',
                    height=500,
                    color=list(counts)[::-1],
                    color_continuous_scale='Blues',
                )
                fig_freq.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_freq, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Buscar Canción
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Explorar una Canción")
    st.markdown("Busca una canción de Radiohead para ver sus features de audio y su letra.")

    search_q = st.text_input("Nombre de la canción", placeholder="Creep, Paranoid Android, Exit Music…")

    if search_q:
        results = artist_df[artist_df['name'].str.contains(search_q, case=False, na=False)]
        if results.empty:
            st.warning("No se encontró ninguna canción con ese nombre.")
        else:
            song_sel = st.selectbox(
                "Canción encontrada:",
                results['name'].tolist()
            )
            row = results[results['name'] == song_sel].iloc[0]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Álbum:** {row['short_album_name']}  \n**Año:** {int(row['year'])}")
                st.markdown("---")
                st.markdown("**Audio Features:**")
                feat_data = {f: round(row[f], 3) for f in AUDIO_FEATURES if f in row}
                st.dataframe(pd.DataFrame(feat_data, index=['valor']).T, use_container_width=True)

            with c2:
                # Radar individual
                vals = [row[f] for f in AUDIO_FEATURES]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals, theta=AUDIO_FEATURES,
                    fill='toself', fillcolor='rgba(31,119,180,0.15)',
                    line_color='#1f77b4',
                    name=song_sel,
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=song_sel,
                    template='plotly_white',
                    height=350,
                )
                st.plotly_chart(fig_r, use_container_width=True)

            st.markdown("---")
            st.markdown("**Letra:**")
            if 'lyrics' in row and isinstance(row['lyrics'], str) and row['lyrics'].strip():
                polarity, subjectivity = analyze_sentiment(row['lyrics'])
                mc1, mc2 = st.columns(2)
                mc1.metric("Polaridad", f"{polarity:.3f}", help="−1 muy negativa · +1 muy positiva")
                mc2.metric("Subjetividad", f"{subjectivity:.3f}", help="0 objetivo · 1 muy subjetivo")
                st.text_area("", value=row['lyrics'], height=300)
            else:
                if st.button("Obtener letra ahora"):
                    with st.spinner("Consultando lyrics.ovh..."):
                        lyrics = get_lyrics_cached('radiohead', song_sel)
                    if lyrics:
                        polarity, subjectivity = analyze_sentiment(lyrics)
                        mc1, mc2 = st.columns(2)
                        mc1.metric("Polaridad", f"{polarity:.3f}")
                        mc2.metric("Subjetividad", f"{subjectivity:.3f}")
                        st.text_area("", value=lyrics, height=300)
                    else:
                        st.warning("No se encontró la letra en lyrics.ovh.")

st.markdown("---")
st.caption(
    "Datos de audio: [Spotify 1.2M+ Songs (Kaggle)](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) · "
    "Letras: [lyrics.ovh](https://lyrics.ovh) · "
    "Discografía de estudio: [Wikipedia](https://en.wikipedia.org/wiki/Radiohead_discography)"
)
