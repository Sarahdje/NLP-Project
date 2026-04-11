"""
app.py — Interface Streamlit
Système de recommandation d'hôtels
Université 8 Mai 1945 Guelma — 4ING — Dr. Z. MEHENAOUI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import os

# ════════════════════════════════════════════════════════════
# CONFIG PAGE
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Recommandation Hôtels — NLP",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4361ee;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #4361ee;
    }
    .metric-card p {
        margin: 0;
        color: #555;
        font-size: 0.85rem;
    }
    .hotel-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.7rem;
        transition: box-shadow 0.2s;
    }
    .hotel-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .rank-badge {
        display: inline-block;
        background: #4361ee;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
    .rank-badge-gold   { background: #f4a261; }
    .rank-badge-silver { background: #adb5bd; }
    .rank-badge-bronze { background: #cd6e3a; }
    .positive-tag {
        background: #d4edda; color: #155724;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.8rem; font-weight: 600;
    }
    .negative-tag {
        background: #f8d7da; color: #721c24;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.8rem; font-weight: 600;
    }
    .neutral-tag {
        background: #fff3cd; color: #856404;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.8rem; font-weight: 600;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4361ee;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #4361ee;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #1a1a2e;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f4a261;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)

    # Chemins relatifs — adaptés à la structure du projet
    paths = {
        "ratings" : os.path.join(base, "results", "hotel_stats_ratings.csv"),
        "scores"  : os.path.join(base, "results", "hotel_scores.csv"),
        "compare" : os.path.join(base, "results", "comparaison_systemes.csv"),
        "metrics" : os.path.join(base, "results", "metrics_report.csv"),
    }

    data = {}

    # ── Ratings ──────────────────────────────────────────────
    try:
        df_r = pd.read_csv(paths["ratings"])
        df_r.columns = [c.strip() for c in df_r.columns]
        data["ratings"] = df_r
    except Exception as e:
        data["ratings"] = None
        data["ratings_err"] = str(e)

    # ── Hotel scores (Sarah) ──────────────────────────────────
    try:
        df_s = pd.read_csv(paths["scores"])
        df_s.columns = [c.strip() for c in df_s.columns]
        data["scores"] = df_s
    except Exception as e:
        data["scores"] = None
        data["scores_err"] = str(e)

    # ── Comparaison ───────────────────────────────────────────
    try:
        df_c = pd.read_csv(paths["compare"])
        df_c.columns = [c.strip() for c in df_c.columns]
        data["compare"] = df_c
    except Exception as e:
        data["compare"] = None

    # ── Métriques DistilBERT ──────────────────────────────────
    try:
        df_m = pd.read_csv(paths["metrics"])
        df_m.columns = [c.strip() for c in df_m.columns]
        data["metrics"] = df_m
    except Exception:
        data["metrics"] = None

    return data


def bayesian_score(n, avg, m, global_mean):
    return (n / (n + m)) * avg + (m / (n + m)) * global_mean


def get_rank_color(rank):
    if rank == 1:   return "rank-badge rank-badge-gold"
    if rank == 2:   return "rank-badge rank-badge-silver"
    if rank == 3:   return "rank-badge rank-badge-bronze"
    return "rank-badge"


def render_hotel_card_rating(rank, row):
    badge_cls = get_rank_color(rank)
    st.markdown(f"""
    <div class="hotel-card">
        <span class="{badge_cls}">{rank}</span>
        <strong style="font-size:1rem;">{row['Hotel_Name']}</strong>
        <span style="float:right; font-size:1.1rem; font-weight:700; color:#4361ee;">
            ⭐ {row['score_bayes']:.2f}
        </span><br>
        <small style="margin-left:42px; color:#777;">
            {int(row['nb_avis'])} avis · Score moyen brut : {row['score_moyen']:.2f}
        </small>
    </div>
    """, unsafe_allow_html=True)


def render_hotel_card_sentiment(rank, row, score_col, pct_pos_col, pct_neg_col):
    badge_cls = get_rank_color(rank)
    score = row[score_col]
    pct_pos = row[pct_pos_col]
    pct_neg = row[pct_neg_col]
    n = int(row.get('nb_avis_sent', row.get('n_reviews', 0)))
    st.markdown(f"""
    <div class="hotel-card">
        <span class="{badge_cls}">{rank}</span>
        <strong style="font-size:1rem;">{row['Hotel_Name']}</strong>
        <span style="float:right; font-size:1.1rem; font-weight:700; color:#2e7d32;">
            {score:.1f}
        </span><br>
        <small style="margin-left:42px; color:#777;">
            {n} avis ·
            <span class="positive-tag">+{pct_pos:.1f}% Positifs</span>
            &nbsp;
            <span class="negative-tag">-{pct_neg:.1f}% Négatifs</span>
        </small>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════
with st.spinner("Chargement des données…"):
    data = load_data()

df_ratings = data.get("ratings")
df_scores  = data.get("scores")
df_compare = data.get("compare")
df_metrics = data.get("metrics")

# ════════════════════════════════════════════════════════════
# EN-TÊTE
# ════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🏨 Système de Recommandation d\'Hôtels</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyse de Sentiments · DistilBERT · Université 8 Mai 1945 Guelma — 4ING — Dr. Z. MEHENAOUI</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MÉTRIQUES GLOBALES
# ════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total avis", "515 287")
with c2:
    st.metric("Hôtels", "1 492")
with c3:
    st.metric("Accuracy DistilBERT", "71.8%")
with c4:
    st.metric("F1 Weighted", "0.730")
with c5:
    st.metric("Accord Score↔BERT", "72.2%")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏅 Système 1 — Ratings",
    "🤖 Système 2 — Sentiments",
    "⚖️ Comparaison",
    "📊 Performance DistilBERT",
    "ℹ️ À propos"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — SYSTÈME RATINGS
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Système 1 — Recommandation par Ratings (Bayesian Weighted Score)</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Méthode :</strong> Bayesian Weighted Rating (formule IMDB) <br>
    <code>weighted_score = (n/(n+m)) × avg_score + (m/(n+m)) × global_mean</code><br>
    où <em>n</em> = nb d'avis de l'hôtel, <em>m</em> = seuil de confiance
    </div>
    """, unsafe_allow_html=True)

    if df_ratings is not None:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**Paramètres**")
            top_k = st.slider("Nombre de recommandations (Top K)", 3, 20, 10, key="top_k_r")
            min_avis = st.slider("Avis minimum requis", 10, 300, 50, key="min_avis_r")
            score_min = st.slider("Score bayésien minimum", 7.0, 10.0, 8.0, 0.1, key="score_min_r")

            # Recherche hôtel
            st.markdown("**Rechercher un hôtel**")
            search = st.text_input("Nom de l'hôtel", placeholder="ex: Hilton, Marriott…", key="search_r")

        with col_right:
            # Filtrage
            df_filt = df_ratings.copy()
            df_filt = df_filt[df_filt['nb_avis'] >= min_avis]
            df_filt = df_filt[df_filt['score_bayes'] >= score_min]

            if search:
                df_filt = df_filt[df_filt['Hotel_Name'].str.contains(search, case=False, na=False)]

            df_filt = df_filt.sort_values('score_bayes', ascending=False).reset_index(drop=True)

            st.markdown(f"**Top {top_k} hôtels recommandés** ({len(df_filt)} hôtels correspondent aux filtres)")

            if df_filt.empty:
                st.warning("Aucun hôtel ne correspond aux critères sélectionnés.")
            else:
                for i, row in df_filt.head(top_k).iterrows():
                    render_hotel_card_rating(i + 1, row)

        # Graphique distribution
        st.markdown("---")
        st.markdown("**Distribution des scores bayésiens**")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(df_ratings['score_bayes'], bins=40, color='#4361ee', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(df_ratings['score_bayes'].mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f"Moyenne ({df_ratings['score_bayes'].mean():.2f})")
        ax.set_xlabel("Score bayésien")
        ax.set_ylabel("Nombre d'hôtels")
        ax.legend()
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

        # Tableau téléchargeable
        with st.expander("📋 Voir le tableau complet"):
            st.dataframe(df_filt.head(50)[['Hotel_Name','nb_avis','score_moyen','score_bayes']].reset_index(drop=True), use_container_width=True)
    else:
        st.error(f"Fichier hotel_stats_ratings.csv introuvable.\n{data.get('ratings_err','')}")
        st.code("Chemin attendu : results/hotel_stats_ratings.csv")


# ════════════════════════════════════════════════════════════
# TAB 2 — SYSTÈME SENTIMENTS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Système 2 — Recommandation par Analyse de Sentiments (DistilBERT)</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Méthode :</strong> Sentiment Score (Sarah Djerrab)<br>
    <code>sentiment_score = % reviews Positive (BERT) − % reviews Negative (BERT)</code><br>
    Plage : −100 (tout négatif) à +100 (tout positif)
    </div>
    """, unsafe_allow_html=True)

    if df_scores is not None:
        # Détecter les colonnes disponibles
        cols = df_scores.columns.tolist()

        # Chercher les colonnes clés
        hotel_col = next((c for c in cols if 'hotel' in c.lower()), cols[0])
        sent_col  = next((c for c in cols if 'sentiment_score' in c.lower()), None)
        pos_col   = next((c for c in cols if 'pos' in c.lower() and 'pct' in c.lower()), None)
        neg_col   = next((c for c in cols if 'neg' in c.lower() and 'pct' in c.lower()), None)
        n_col     = next((c for c in cols if 'n_review' in c.lower() or 'nb_avis' in c.lower()), None)

        # Renommer pour uniformité
        rename_map = {hotel_col: 'Hotel_Name'}
        if sent_col: rename_map[sent_col] = 'sentiment_score'
        if pos_col:  rename_map[pos_col]  = 'pct_positif'
        if neg_col:  rename_map[neg_col]  = 'pct_negatif'
        if n_col:    rename_map[n_col]    = 'nb_avis_sent'
        df_s2 = df_scores.rename(columns=rename_map)

        col_left2, col_right2 = st.columns([1, 2])

        with col_left2:
            st.markdown("**Paramètres**")
            top_k2   = st.slider("Nombre de recommandations", 3, 20, 10, key="top_k_s")
            min_avis2 = st.slider("Avis minimum requis", 10, 200, 30, key="min_avis_s")
            score_min2 = st.slider("Sentiment score minimum", -100.0, 100.0, 50.0, 1.0, key="score_min_s")
            search2 = st.text_input("Rechercher un hôtel", placeholder="ex: Hilton…", key="search_s")

        with col_right2:
            df_s2_filt = df_s2.copy()
            if 'nb_avis_sent' in df_s2_filt.columns:
                df_s2_filt = df_s2_filt[df_s2_filt['nb_avis_sent'] >= min_avis2]
            if 'sentiment_score' in df_s2_filt.columns:
                df_s2_filt = df_s2_filt[df_s2_filt['sentiment_score'] >= score_min2]
            if search2:
                df_s2_filt = df_s2_filt[df_s2_filt['Hotel_Name'].str.contains(search2, case=False, na=False)]

            df_s2_filt = df_s2_filt.sort_values('sentiment_score', ascending=False).reset_index(drop=True)

            st.markdown(f"**Top {top_k2} hôtels recommandés**")

            if df_s2_filt.empty:
                st.warning("Aucun hôtel ne correspond aux critères.")
            else:
                for i, row in df_s2_filt.head(top_k2).iterrows():
                    n = int(row.get('nb_avis_sent', 0))
                    pct_pos = row.get('pct_positif', 0)
                    pct_neg = row.get('pct_negatif', 0)
                    score   = row.get('sentiment_score', 0)
                    badge_cls = get_rank_color(i + 1)
                    st.markdown(f"""
                    <div class="hotel-card">
                        <span class="{badge_cls}">{i+1}</span>
                        <strong style="font-size:1rem;">{row['Hotel_Name']}</strong>
                        <span style="float:right; font-size:1.1rem; font-weight:700; color:#2e7d32;">
                            {score:.1f}
                        </span><br>
                        <small style="margin-left:42px; color:#777;">
                            {n} avis ·
                            <span class="positive-tag">+{pct_pos:.1f}% Pos</span>
                            &nbsp;
                            <span class="negative-tag">-{pct_neg:.1f}% Nég</span>
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

        # Distribution sentiment scores
        if 'sentiment_score' in df_s2.columns:
            st.markdown("---")
            st.markdown("**Distribution des sentiment scores**")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.hist(df_s2['sentiment_score'].dropna(), bins=40, color='#2e7d32',
                     edgecolor='white', linewidth=0.5, alpha=0.85)
            ax2.axvline(df_s2['sentiment_score'].mean(), color='red', linestyle='--',
                        linewidth=1.5, label=f"Moyenne ({df_s2['sentiment_score'].mean():.1f})")
            ax2.set_xlabel("Sentiment Score")
            ax2.set_ylabel("Nombre d'hôtels")
            ax2.legend()
            ax2.spines[['top','right']].set_visible(False)
            st.pyplot(fig2)
            plt.close()

        with st.expander("📋 Voir le tableau complet"):
            st.dataframe(df_s2_filt.head(50).reset_index(drop=True), use_container_width=True)
    else:
        st.error(f"Fichier hotel_scores.csv introuvable.\n{data.get('scores_err','')}")
        st.code("Chemin attendu : results/hotel_scores.csv")


# ════════════════════════════════════════════════════════════
# TAB 3 — COMPARAISON
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Comparaison des deux systèmes de recommandation</p>', unsafe_allow_html=True)

    if df_compare is not None:
        cols_c = df_compare.columns.tolist()

        # Colonnes nécessaires
        needed = ['Hotel_Name', 'rang_ratings', 'rang_sentiments', 'score_bayes',
                  'score_sentiment_moy', 'diff_rang', 'pct_positif', 'pct_negatif']
        missing = [c for c in needed if c not in cols_c]

        if missing:
            st.warning(f"Colonnes manquantes dans comparaison_systemes.csv : {missing}")
            st.write("Colonnes disponibles :", cols_c)
        else:
            # ── Corrélation de Spearman ──────────────────────────
            corr, pval = stats.spearmanr(
                df_compare['rang_ratings'],
                df_compare['rang_sentiments']
            )

            c1c, c2c, c3c = st.columns(3)
            with c1c:
                st.metric("Corrélation de Spearman", f"{corr:.3f}")
            with c2c:
                st.metric("p-value", f"{pval:.4f}")
            with c3c:
                st.metric("Hôtels comparés", len(df_compare))

            if corr >= 0.8:
                msg = "✅ Les deux systèmes sont très similaires — le texte confirme les notes numériques."
                cls = "info-box"
            elif corr >= 0.6:
                msg = "⚠️ Corrélation modérée — des divergences existent. Le texte apporte une information supplémentaire aux notes."
                cls = "warning-box"
            else:
                msg = "🔴 Faible corrélation — les deux systèmes capturent des signaux très différents."
                cls = "warning-box"

            st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

            st.markdown("---")

            col_g1, col_g2 = st.columns(2)

            # ── Bump chart ───────────────────────────────────────
            with col_g1:
                st.markdown("**Classements croisés — Top 30 hôtels**")
                top30 = df_compare.nsmallest(30, 'rang_ratings').copy()
                fig3, ax3 = plt.subplots(figsize=(6, 7))
                for _, row in top30.iterrows():
                    div = abs(row['diff_rang'])
                    cl  = '#e74c3c' if div > 5 else '#bdc3c7'
                    lw  = 2.0       if div > 5 else 0.8
                    ax3.plot([1, 2], [row['rang_ratings'], row['rang_sentiments']],
                             color=cl, linewidth=lw, alpha=0.85)
                    if div > 5:
                        ax3.text(2.05, row['rang_sentiments'],
                                 row['Hotel_Name'][:22], fontsize=6.5,
                                 va='center', color='#c0392b')
                ax3.set_xticks([1, 2])
                ax3.set_xticklabels(['Ratings', 'Sentiments\n(DistilBERT)'], fontsize=10)
                ax3.invert_yaxis()
                ax3.set_ylabel("Rang (1 = meilleur)", fontsize=9)
                ax3.set_title("Rouge = divergence > 5 rangs", fontsize=9, color='#c0392b')
                ax3.spines[['top','right']].set_visible(False)
                st.pyplot(fig3)
                plt.close()

            # ── Distribution divergences ─────────────────────────
            with col_g2:
                st.markdown("**Distribution des divergences de rang**")
                fig4, ax4 = plt.subplots(figsize=(6, 7))
                ax4.hist(df_compare['diff_rang'], bins=40, color='#4361ee',
                         edgecolor='white', linewidth=0.5, alpha=0.85)
                ax4.axvline(df_compare['diff_rang'].mean(), color='red', linestyle='--',
                            linewidth=1.5, label=f"Moyenne ({df_compare['diff_rang'].mean():.1f})")
                ax4.axvline(df_compare['diff_rang'].median(), color='orange', linestyle='--',
                            linewidth=1.5, label=f"Médiane ({df_compare['diff_rang'].median():.1f})")
                ax4.set_xlabel("Écart de rang")
                ax4.set_ylabel("Nombre d'hôtels")
                ax4.legend(fontsize=8)
                ax4.spines[['top','right']].set_visible(False)
                st.pyplot(fig4)
                plt.close()

            # ── Top 10 divergents ────────────────────────────────
            st.markdown("---")
            st.markdown("**Top 10 hôtels avec les classements les plus divergents**")
            st.markdown("""
            <div class="info-box">
            Ces hôtels reçoivent des classements très différents selon la méthode utilisée.
            Ils illustrent le cas où la note numérique ne reflète pas l'opinion exprimée dans le texte.
            </div>
            """, unsafe_allow_html=True)

            divergents = df_compare.nlargest(10, 'diff_rang')[
                ['Hotel_Name','rang_ratings','rang_sentiments','diff_rang',
                 'score_bayes','pct_positif','pct_negatif']
            ].reset_index(drop=True)

            for i, row in divergents.iterrows():
                diff = int(row['diff_rang'])
                arrow = "⬆️" if row['rang_sentiments'] < row['rang_ratings'] else "⬇️"
                st.markdown(f"""
                <div class="hotel-card">
                    <strong>{row['Hotel_Name']}</strong>
                    <span style="float:right; color:#e74c3c; font-weight:700;">
                        Écart : {diff} rangs {arrow}
                    </span><br>
                    <small style="color:#777;">
                        Rang Ratings : <strong>{int(row['rang_ratings'])}</strong> →
                        Rang Sentiments : <strong>{int(row['rang_sentiments'])}</strong> &nbsp;|&nbsp;
                        Score Bayes : {row['score_bayes']:.2f} &nbsp;|&nbsp;
                        <span class="positive-tag">+{row['pct_positif']:.1f}%</span>
                        <span class="negative-tag">-{row['pct_negatif']:.1f}%</span>
                    </small>
                </div>
                """, unsafe_allow_html=True)

            # Statistiques globales
            st.markdown("---")
            st.markdown("**Statistiques globales de divergence**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Divergence moyenne",  f"{df_compare['diff_rang'].mean():.1f} rangs")
            sc2.metric("Divergence médiane",  f"{df_compare['diff_rang'].median():.1f} rangs")
            sc3.metric("Divergence max",      f"{df_compare['diff_rang'].max():.0f} rangs")
            sc4.metric("Hôtels > 50 rangs d'écart", f"{(df_compare['diff_rang'] > 50).sum()}")

            # Scatter plot
            st.markdown("---")
            st.markdown("**Score Bayes vs Score Sentiment**")
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            sc = ax5.scatter(df_compare['score_bayes'], df_compare['score_sentiment_moy'],
                             c=df_compare['diff_rang'], cmap='RdYlGn_r',
                             alpha=0.6, s=20)
            plt.colorbar(sc, ax=ax5, label="Divergence de rang")
            ax5.set_xlabel("Score Bayésien (Ratings)")
            ax5.set_ylabel("Score Sentiment moyen (DistilBERT)")
            ax5.set_title("Corrélation entre les deux scores par hôtel")
            ax5.spines[['top','right']].set_visible(False)
            st.pyplot(fig5)
            plt.close()

    elif df_ratings is not None and df_scores is not None:
        st.warning("Le fichier comparaison_systemes.csv n'existe pas encore.")
        st.markdown("Lance le notebook `07_comparison.ipynb` pour générer ce fichier.")
        st.code("résultat attendu : results/comparaison_systemes.csv")
    else:
        st.error("Fichiers de données manquants. Vérifie que les notebooks 06 et 07 ont bien été exécutés.")


# ════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE DISTILBERT
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Performance du modèle DistilBERT</p>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Résultats d'entraînement**")
        train_data = {
            "Epoch": [1, 2, 3],
            "Train Loss": [0.7304, 0.6182, 0.5736],
            "Val Loss":   [0.6667, 0.6611, 0.6611],
            "Val F1 (macro)": [0.6354, 0.6350, 0.6350]
        }
        df_train = pd.DataFrame(train_data)
        st.dataframe(df_train.style.highlight_min(['Val Loss'], color='#d4edda')
                                   .highlight_max(['Val F1 (macro)'], color='#d4edda'),
                     use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <strong>Meilleur modèle : Epoch 2</strong><br>
        Val Loss commence à stagner à l'epoch 3 → overfitting léger confirmé.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Évaluation sur le jeu de test**")
        eval_data = {
            "Classe":    ["Negative", "Neutral", "Positive", "Weighted avg"],
            "Precision": [0.42, 0.53, 0.88, "—"],
            "Recall":    [0.67, 0.63, 0.76, "—"],
            "F1":        [0.52, 0.58, 0.82, 0.730],
            "Support":   [4597, 22398, 50306, 77301]
        }
        st.dataframe(pd.DataFrame(eval_data), use_container_width=True)

        st.metric("Accuracy globale", "71.8%")
        st.metric("F1 Macro",   "0.637")
        st.metric("F1 Weighted", "0.730")

    with col_p2:
        st.markdown("**Cross-tabulation Score ↔ DistilBERT**")
        cross_data = {
            "Label Score →": ["Negative (≤5)", "Neutral (6-7)", "Positive (≥8)"],
            "BERT: Negative": ["71.8%", "12.5%", "1.2%"],
            "BERT: Neutral":  ["26.1%", "70.3%", "25.6%"],
            "BERT: Positive": ["2.1%",  "17.2%", "73.1%"]
        }
        st.dataframe(pd.DataFrame(cross_data), use_container_width=True)

        st.markdown("""
        <div class="warning-box">
        <strong>Observations clés :</strong><br>
        • <strong>17.2%</strong> des avis Neutral sont classés Positive par DistilBERT → le client a écrit positivement malgré une note moyenne<br>
        • <strong>25.6%</strong> des avis Positive sont classés Neutral → le client a mis une bonne note mais écrit de façon tiède<br>
        • Ces désaccords prouvent que les deux systèmes capturent des <strong>signaux différents</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Distribution des prédictions DistilBERT**")
        pred_data = {
            "Sentiment":  ["Positive", "Neutral", "Negative"],
            "Count":      [271593, 198831, 44863],
            "Pourcentage": ["52.7%", "38.6%", "8.7%"]
        }
        df_pred_dist = pd.DataFrame(pred_data)
        st.dataframe(df_pred_dist, use_container_width=True)

        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        colors = ['#2e7d32', '#f4a261', '#c0392b']
        ax_pie.pie([271593, 198831, 44863],
                   labels=['Positive\n52.7%', 'Neutral\n38.6%', 'Negative\n8.7%'],
                   colors=colors, autopct='', startangle=90,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax_pie.set_title("Répartition des prédictions DistilBERT", fontsize=10)
        st.pyplot(fig_pie)
        plt.close()

    # Visualisations sauvegardées
    st.markdown("---")
    st.markdown("**Visualisations générées**")
    viz_base = os.path.join(os.path.dirname(__file__), "results", "visualizations")
    viz_files = {
        "Matrice de confusion":      "confusion_matrix_distrilbert_finetuned.png",
        "Courbes d'entraînement":    "loss_val_distilbert_finetuned.png",
        "Résultats jeu de test":     "test_set_results_distilbert_finetuned.png",
        "Distribution sentiments":   "sentiment_distribution.png",
        "Comparaison classements":   "comparaison_classements.png",
    }
    cols_viz = st.columns(3)
    for idx, (label, fname) in enumerate(viz_files.items()):
        fpath = os.path.join(viz_base, fname)
        with cols_viz[idx % 3]:
            if os.path.exists(fpath):
                st.image(fpath, caption=label, use_column_width=True)
            else:
                st.markdown(f"<small style='color:#aaa;'>📷 {label}<br><em>Image non trouvée</em></small>",
                            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — À PROPOS
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">À propos du projet</p>', unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        **Mini-Projet 16 — NLP**

        | | |
        |---|---|
        | **Université** | 8 Mai 1945 Guelma |
        | **Niveau** | 4ème Ingénieur |
        | **Module** | TALN |
        | **Encadrant** | Dr. Z. MEHENAOUI |
        | **Année** | 2025/2026 |

        ---

        **Équipe**

        | Membre | Rôle |
        |---|---|
        | **Tasnim** | Prétraitement NLP |
        | **Sarah Djerrab** | DistilBERT + Système Sentiments |
        | **Khaoula** | Système Ratings + Comparaison |

        ---

        **Dataset**

        | | |
        |---|---|
        | Source | Kaggle — Hotel Reviews Europe |
        | Avis | 515 287 |
        | Hôtels | 1 492 |
        | Split | 360K train / 77K val / 77K test |
        """)

    with col_a2:
        st.markdown("""
        **Architecture du pipeline**

        ```
        Hotel_Reviews.csv (brut)
               ↓
        02_preprocessing.ipynb  (Tasnim)
        — Nettoyage, tokenisation, lemmatisation
        — Labels : Positive/Neutral/Negative
               ↓
        04_distilbert.ipynb  (Sarah — Kaggle GPU)
        — Fine-tuning DistilBERT 3 epochs
        — Class weights pour le déséquilibre
               ↓
        05_inference_kaggle.ipynb  (Sarah)
        — Prédiction sur 515k avis (~8 min GPU)
               ↓
        05_recommendation.ipynb  (Sarah)
        — Calcul sentiment_score par hôtel
               ↓
        06_rating_recomm.ipynb  (Khaoula)
        — Bayesian Weighted Rating par hôtel
               ↓
        07_comparison.ipynb  (Khaoula)
        — Corrélation de Spearman
        — Hôtels divergents
        — Visualisations
        ```

        **Limitations**

        1. Labels bruités (score → label)
        2. Déséquilibre Negative (6%)
        3. DistilBERT pré-entraîné sur texte général
        4. Seuils 8/5 arbitraires
        5. Biais de corrélation score↔texte
        """)

    st.markdown("---")
    st.markdown(
        "<center><small>Interface développée avec Streamlit · Projet NLP 2025/2026</small></center>",
        unsafe_allow_html=True
    )