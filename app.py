import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        font-size: 2.2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; margin-bottom: 1.5rem;
    }
    .hotel-card {
        background: white; border-radius: 12px;
        padding: 1rem 1.3rem; border: 1px solid #e0e0e0;
        margin-bottom: 0.7rem;
    }
    .rank-badge {
        display: inline-block; background: #4361ee; color: white;
        border-radius: 50%; width: 30px; height: 30px;
        text-align: center; line-height: 30px;
        font-weight: bold; margin-right: 10px;
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
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #1a1a2e;
        border-bottom: 2px solid #4361ee;
        padding-bottom: 0.3rem; margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background: #e8f4fd; border-left: 4px solid #4361ee;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 1rem 0; font-size: 0.9rem; color: #1a1a2e;
    }
    .warning-box {
        background: #fff8e1; border-left: 4px solid #f4a261;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 1rem 0; font-size: 0.9rem;
    }
    .success-box {
        background: #d4edda; border-left: 4px solid #2e7d32;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 1rem 0; font-size: 0.9rem; color: #155724;
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
    base = os.path.dirname(os.path.abspath(__file__))

    paths = {
        "ratings" : os.path.join(base, "results", "hotel_stats_ratings.csv"),
        "scores"  : os.path.join(base, "results", "hotel_scores.csv"),
        "compare" : os.path.join(base, "results", "comparaison_systemes.csv"),
        "metrics" : os.path.join(base, "results", "metrics_report.csv"),
    }

    data = {}

    for key, path in paths.items():
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            data[key] = df
        except Exception as e:
            data[key] = None
            data[f"{key}_err"] = str(e)

    # ── Pré-calcul pct_negatif dans hotel_scores ──────────────
    # hotel_scores.csv a pct_pred_positive mais pas pct_negatif
    if data.get("scores") is not None:
        df_s = data["scores"]
        # Normaliser nom colonne hotel → Hotel_Name
        if "hotel" in df_s.columns and "Hotel_Name" not in df_s.columns:
            df_s = df_s.rename(columns={"hotel": "Hotel_Name"})
        # Calculer pct_negatif
        if "pct_pred_positive" in df_s.columns and "pct_negatif" not in df_s.columns:
            df_s["pct_negatif"] = 100.0 - df_s["pct_pred_positive"]
        # Renommer pct_pred_positive → pct_positif pour cohérence affichage
        if "pct_pred_positive" in df_s.columns and "pct_positif" not in df_s.columns:
            df_s = df_s.rename(columns={"pct_pred_positive": "pct_positif"})
        # n_snippets → nb_avis_sent pour cohérence
        if "n_snippets" in df_s.columns and "nb_avis_sent" not in df_s.columns:
            df_s = df_s.rename(columns={"n_snippets": "nb_avis_sent"})
        data["scores"] = df_s

    # ── Pré-calcul pct_negatif dans comparaison ───────────────
    if data.get("compare") is not None:
        df_c = data["compare"]
        if "pct_pred_positive" in df_c.columns and "pct_positif" not in df_c.columns:
            df_c = df_c.rename(columns={"pct_pred_positive": "pct_positif"})
        if "pct_positif" in df_c.columns and "pct_negatif" not in df_c.columns:
            df_c["pct_negatif"] = 100.0 - df_c["pct_positif"]
        data["compare"] = df_c

    return data


def get_rank_color(rank):
    if rank == 1: return "rank-badge rank-badge-gold"
    if rank == 2: return "rank-badge rank-badge-silver"
    if rank == 3: return "rank-badge rank-badge-bronze"
    return "rank-badge"


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
st.markdown('<p class="main-title">🏨 Système de Recommandation d\'Hôtels</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Analyse de Sentiments · DistilBERT · '
    'Université 8 Mai 1945 Guelma — 4ING — Dr. Z. MEHENAOUI</p>',
    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MÉTRIQUES GLOBALES
# BUG CORRIGÉ #1 : valeurs de l'ancienne approche 3 classes
# remplacées par les vraies valeurs du modèle binaire (rapport Sarah)
# ════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total snippets", "709 313")
with c2:
    st.metric("Hôtels", "1 492")
with c3:
    # Correction : 97.37% (binaire) et non 71.8% (ancienne approche 3 classes)
    st.metric("Accuracy DistilBERT", "97.37%")
with c4:
    # Correction : 0.9733 et non 0.637
    st.metric("F1 Macro", "0.9733")
with c5:
    # Correction : 0.9737 et non 0.730
    st.metric("F1 Weighted", "0.9737")

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
    st.markdown(
        '<p class="section-header">Système 1 — Recommandation par Ratings (Bayesian Average)</p>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Méthode :</strong> Bayesian Average (formule IMDb)<br>
    <code>score_bayes = (v / (v+m)) × R + (m / (v+m)) × C</code><br>
    <em>v</em> = nb avis de l'hôtel · <em>R</em> = score moyen de l'hôtel ·
    <em>m</em> = médiane du nb d'avis (seuil) · <em>C</em> = moyenne globale (ancrage)
    </div>
    """, unsafe_allow_html=True)

    if df_ratings is not None:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**Paramètres**")
            top_k     = st.slider("Top K recommandations", 3, 20, 10, key="top_k_r")
            min_avis  = st.slider("Avis minimum requis", 10, 300, 50, key="min_avis_r")
            score_min = st.slider("Score bayésien minimum", 7.0, 10.0, 8.0, 0.1, key="score_min_r")
            search    = st.text_input("Rechercher un hôtel", placeholder="ex: Hilton…", key="search_r")

        with col_right:
            df_filt = df_ratings[df_ratings["nb_avis"] >= min_avis].copy()
            df_filt = df_filt[df_filt["score_bayes"] >= score_min]
            if search:
                df_filt = df_filt[df_filt["Hotel_Name"].str.contains(
                    search, case=False, na=False)]
            df_filt = df_filt.sort_values("score_bayes", ascending=False).reset_index(drop=True)

            st.markdown(f"**Top {top_k} hôtels** ({len(df_filt)} correspondent aux filtres)")

            if df_filt.empty:
                st.warning("Aucun hôtel ne correspond aux critères sélectionnés.")
            else:
                for i, row in df_filt.head(top_k).iterrows():
                    badge_cls = get_rank_color(i + 1)
                    st.markdown(f"""
                    <div class="hotel-card">
                        <span class="{badge_cls}">{i+1}</span>
                        <strong>{row['Hotel_Name']}</strong>
                        <span style="float:right;font-size:1.1rem;font-weight:700;color:#4361ee;">
                            ⭐ {row['score_bayes']:.3f}
                        </span><br>
                        <small style="margin-left:42px;color:#777;">
                            {int(row['nb_avis'])} avis · Score moyen brut : {row['score_moyen']:.2f}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Distribution des scores bayésiens**")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(df_ratings["score_bayes"], bins=40, color="#4361ee",
                edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(df_ratings["score_bayes"].mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"Moyenne ({df_ratings['score_bayes'].mean():.2f})")
        ax.set_xlabel("Score bayésien (/10)")
        ax.set_ylabel("Nombre d'hôtels")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

        with st.expander("📋 Tableau complet"):
            st.dataframe(
                df_filt.head(50)[["Hotel_Name", "nb_avis", "score_moyen", "score_bayes"]]
                .reset_index(drop=True), use_container_width=True)
    else:
        st.error(f"hotel_stats_ratings.csv introuvable.\n{data.get('ratings_err', '')}")
        st.code("Chemin attendu : results/hotel_stats_ratings.csv")


# ════════════════════════════════════════════════════════════
# TAB 2 — SYSTÈME SENTIMENTS
# BUG CORRIGÉ #2 : description formule, slider range, colonnes
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<p class="section-header">Système 2 — Recommandation par Analyse de Sentiments (DistilBERT)</p>',
        unsafe_allow_html=True)

    # Correction : la vraie formule est mean(P(Positive)) ∈ [0, 1]
    # L'ancienne description "% Pos − % Neg, plage [-100,+100]" était fausse
    st.markdown("""
    <div class="info-box">
    <strong>Méthode :</strong> Sentiment Score (Sarah Djerrab)<br>
    <code>sentiment_score = mean( P(Positive) )</code> sur tous les snippets de l'hôtel<br>
    Plage : 0 (tout négatif) à 1 (tout positif) · en pratique [0.34, 0.77] (structure bimodale)
    </div>
    """, unsafe_allow_html=True)

    if df_scores is not None:
        col_left2, col_right2 = st.columns([1, 2])

        with col_left2:
            st.markdown("**Paramètres**")
            top_k2    = st.slider("Top K recommandations", 3, 20, 10, key="top_k_s")
            min_avis2 = st.slider("Snippets minimum requis", 10, 200, 50, key="min_avis_s")

            # Correction : plage [0.0, 1.0] car sentiment_score = mean(P(Positive)) ∈ [0,1]
            score_min2 = st.slider("Sentiment score minimum", 0.0, 1.0, 0.5, 0.01,
                                   key="score_min_s")
            search2 = st.text_input("Rechercher un hôtel", placeholder="ex: Hilton…",
                                    key="search_s")

        with col_right2:
            df_s2 = df_scores.copy()

            # Filtres
            avis_col = "nb_avis_sent" if "nb_avis_sent" in df_s2.columns else None
            if avis_col:
                df_s2 = df_s2[df_s2[avis_col] >= min_avis2]
            if "sentiment_score" in df_s2.columns:
                df_s2 = df_s2[df_s2["sentiment_score"] >= score_min2]
            if search2:
                df_s2 = df_s2[df_s2["Hotel_Name"].str.contains(
                    search2, case=False, na=False)]

            df_s2 = df_s2.sort_values("sentiment_score", ascending=False).reset_index(drop=True)

            st.markdown(f"**Top {top_k2} hôtels**")

            if df_s2.empty:
                st.warning("Aucun hôtel ne correspond aux critères.")
            else:
                for i, row in df_s2.head(top_k2).iterrows():
                    badge_cls = get_rank_color(i + 1)
                    n       = int(row.get("nb_avis_sent", 0))
                    pct_pos = row.get("pct_positif", 0)
                    pct_neg = row.get("pct_negatif", 100 - pct_pos)
                    score   = row.get("sentiment_score", 0)
                    st.markdown(f"""
                    <div class="hotel-card">
                        <span class="{badge_cls}">{i+1}</span>
                        <strong>{row['Hotel_Name']}</strong>
                        <span style="float:right;font-size:1.1rem;font-weight:700;color:#2e7d32;">
                            {score:.4f}
                        </span><br>
                        <small style="margin-left:42px;color:#777;">
                            {n} snippets ·
                            <span class="positive-tag">+{pct_pos:.1f}% Pos</span>
                            &nbsp;
                            <span class="negative-tag">-{pct_neg:.1f}% Nég</span>
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

        if "sentiment_score" in df_scores.columns:
            st.markdown("---")
            st.markdown("**Distribution des sentiment scores** (mean P(Positive) par hôtel)")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.hist(df_scores["sentiment_score"].dropna(), bins=40, color="#2e7d32",
                     edgecolor="white", linewidth=0.5, alpha=0.85)
            ax2.axvline(df_scores["sentiment_score"].mean(), color="red", linestyle="--",
                        linewidth=1.5,
                        label=f"Moyenne ({df_scores['sentiment_score'].mean():.3f})")
            ax2.axvline(0.5, color="orange", linestyle=":", linewidth=1.5,
                        label="Seuil neutre (0.5)")
            ax2.set_xlabel("Sentiment Score  [0 = très négatif, 1 = très positif]")
            ax2.set_ylabel("Nombre d'hôtels")
            ax2.legend()
            ax2.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)
            plt.close()

        with st.expander("📋 Tableau complet"):
            st.dataframe(df_s2.head(50).reset_index(drop=True), use_container_width=True)
    else:
        st.error(f"hotel_scores.csv introuvable.\n{data.get('scores_err', '')}")
        st.code("Chemin attendu : results/hotel_scores.csv")


# ════════════════════════════════════════════════════════════
# TAB 3 — COMPARAISON
# BUG CORRIGÉ #5 : colonnes réelles de comparaison_systemes.csv
# score_sentiment_moy → sentiment_score
# diff_rang          → diff_rang_pct / diff_rang_abs
# pct_positif/negatif → calculées en amont dans load_data()
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="section-header">Comparaison des deux systèmes de recommandation</p>',
        unsafe_allow_html=True)

    if df_compare is not None:
        cols_c = df_compare.columns.tolist()

        # Colonnes réellement présentes dans comparaison_systemes.csv (produit par 07_comparison.ipynb)
        col_diff     = "diff_rang_pct"   if "diff_rang_pct"  in cols_c else "diff_rang_abs"
        col_sent     = "sentiment_score" if "sentiment_score" in cols_c else None
        col_pct_pos  = "pct_positif"     if "pct_positif"    in cols_c else None
        col_pct_neg  = "pct_negatif"     if "pct_negatif"    in cols_c else None

        required_cols = ["Hotel_Name", "rang_ratings", "rang_sentiments",
                         "score_bayes", col_diff]
        if col_sent: required_cols.append(col_sent)

        missing = [c for c in required_cols if c not in cols_c]

        if missing:
            st.warning(f"Colonnes manquantes dans comparaison_systemes.csv : {missing}")
            st.write("Colonnes disponibles :", cols_c)
            st.info("Lance le notebook `07_comparison.ipynb` pour régénérer ce fichier.")
        else:
            # ── Corrélation de Spearman ──────────────────────────
            corr, pval = stats.spearmanr(
                df_compare["rang_ratings"],
                df_compare["rang_sentiments"]
            )

            c1c, c2c, c3c = st.columns(3)
            c1c.metric("Corrélation de Spearman ρ", f"{corr:.3f}")
            c2c.metric("p-value", f"{pval:.4f}")
            c3c.metric("Hôtels comparés", len(df_compare))

            if corr >= 0.8:
                msg = "✅ Forte corrélation — le texte confirme les notes numériques."
                cls = "info-box"
            elif corr >= 0.5:
                msg = "⚠️ Corrélation modérée — le texte apporte une information supplémentaire. Système hybride justifié."
                cls = "warning-box"
            else:
                msg = "🔴 Faible corrélation — les deux systèmes capturent des signaux très différents."
                cls = "warning-box"
            st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

            st.markdown("---")
            col_g1, col_g2 = st.columns(2)

            # ── Bump chart (rangs percentiles) ───────────────────
            with col_g1:
                st.markdown("**Classements croisés — Top 30 hôtels (rang percentile)**")
                # Utilise rang_pct si disponible, sinon rang brut normalisé
                has_pct = ("rang_pct_ratings" in cols_c and "rang_pct_sentiments" in cols_c)
                top30   = df_compare.nsmallest(30, "rang_ratings").copy()

                fig3, ax3 = plt.subplots(figsize=(6, 7))
                for _, row in top30.iterrows():
                    if has_pct:
                        y1, y2 = row["rang_pct_ratings"], row["rang_pct_sentiments"]
                        div    = abs(y1 - y2)
                        thresh = 0.10
                    else:
                        y1, y2 = row["rang_ratings"], row["rang_sentiments"]
                        div    = abs(y1 - y2)
                        thresh = 50
                    cl = "#e74c3c" if div > thresh else "#bdc3c7"
                    lw = 2.0       if div > thresh else 0.8
                    ax3.plot([1, 2], [y1, y2], color=cl, linewidth=lw, alpha=0.85)
                    if div > thresh:
                        ax3.text(2.05, y2, row["Hotel_Name"][:22],
                                 fontsize=6.5, va="center", color="#c0392b")

                ax3.set_xticks([1, 2])
                ax3.set_xticklabels(["Ratings\n(Bayesian)", "Sentiments\n(DistilBERT)"],
                                    fontsize=10)
                ax3.invert_yaxis()
                ylabel = "Rang percentile (0 = meilleur)" if has_pct else "Rang (1 = meilleur)"
                ax3.set_ylabel(ylabel, fontsize=9)
                ax3.set_title("Rouge = divergence significative", fontsize=9, color="#c0392b")
                ax3.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig3)
                plt.close()

            # ── Distribution divergences ─────────────────────────
            with col_g2:
                st.markdown("**Distribution des divergences de classement**")
                div_vals = df_compare[col_diff]
                fig4, ax4 = plt.subplots(figsize=(6, 7))
                ax4.hist(div_vals, bins=40, color="#4361ee",
                         edgecolor="white", linewidth=0.5, alpha=0.85)
                ax4.axvline(div_vals.mean(), color="red", linestyle="--", linewidth=1.5,
                            label=f"Moyenne ({div_vals.mean():.3f})")
                ax4.axvline(div_vals.median(), color="orange", linestyle="--", linewidth=1.5,
                            label=f"Médiane ({div_vals.median():.3f})")
                xlabel = "|rang_pct_ratings − rang_pct_sentiments|" if "pct" in col_diff \
                         else "|rang_ratings − rang_sentiments|"
                ax4.set_xlabel(xlabel)
                ax4.set_ylabel("Nombre d'hôtels")
                ax4.legend(fontsize=8)
                ax4.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig4)
                plt.close()

            # ── Top 10 hôtels divergents ─────────────────────────
            st.markdown("---")
            st.markdown("**Top 10 hôtels avec les classements les plus divergents**")
            st.markdown("""
            <div class="info-box">
            Ces hôtels illustrent le cas où la note numérique ne reflète pas
            l'opinion exprimée dans le texte — c'est la valeur ajoutée du projet.
            </div>
            """, unsafe_allow_html=True)

            div_display_cols = ["Hotel_Name", "rang_ratings", "rang_sentiments",
                                col_diff, "score_bayes"]
            if col_sent:    div_display_cols.append(col_sent)
            if col_pct_pos: div_display_cols.append(col_pct_pos)
            if col_pct_neg: div_display_cols.append(col_pct_neg)

            divergents = df_compare.nlargest(10, col_diff)[
                [c for c in div_display_cols if c in cols_c]
            ].reset_index(drop=True)

            for i, row in divergents.iterrows():
                diff_val = row[col_diff]
                arrow = "⬆️" if row["rang_sentiments"] < row["rang_ratings"] else "⬇️"
                pct_pos = row.get(col_pct_pos, None) if col_pct_pos else None
                pct_neg = row.get(col_pct_neg, None) if col_pct_neg else None
                tags = ""
                if pct_pos is not None and pct_neg is not None:
                    tags = (f'<span class="positive-tag">+{pct_pos:.1f}% Pos</span>'
                            f'&nbsp;<span class="negative-tag">-{pct_neg:.1f}% Nég</span>')
                sent_str = f"Sentiment : {row[col_sent]:.4f} &nbsp;|&nbsp;" if col_sent else ""
                st.markdown(f"""
                <div class="hotel-card">
                    <strong>{row['Hotel_Name']}</strong>
                    <span style="float:right;color:#e74c3c;font-weight:700;">
                        Écart : {diff_val:.3f} {arrow}
                    </span><br>
                    <small style="color:#777;">
                        Rang Ratings : <strong>{int(row['rang_ratings'])}</strong> →
                        Rang Sentiments : <strong>{int(row['rang_sentiments'])}</strong>
                        &nbsp;|&nbsp; {sent_str}Score Bayes : {row['score_bayes']:.2f}
                        &nbsp;{tags}
                    </small>
                </div>
                """, unsafe_allow_html=True)

            # ── Stats globales ───────────────────────────────────
            st.markdown("---")
            st.markdown("**Statistiques globales de divergence**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Divergence moyenne",  f"{div_vals.mean():.3f}")
            sc2.metric("Divergence médiane",  f"{div_vals.median():.3f}")
            sc3.metric("Divergence max",      f"{div_vals.max():.3f}")
            thresh_25 = 0.25 if "pct" in col_diff else int(len(df_compare) * 0.25)
            sc4.metric(f"Hôtels écart > {'25%' if 'pct' in col_diff else '25% rang'}",
                       f"{(div_vals > thresh_25).sum()}")

            # ── Scatter score_bayes vs sentiment_score ───────────
            if col_sent:
                st.markdown("---")
                st.markdown("**Score Bayes vs Score Sentiment**")
                fig5, ax5 = plt.subplots(figsize=(8, 4))
                sc_plot = ax5.scatter(
                    df_compare["score_bayes"],
                    df_compare[col_sent],
                    c=df_compare[col_diff], cmap="RdYlGn_r",
                    alpha=0.6, s=20
                )
                plt.colorbar(sc_plot, ax=ax5, label="Divergence de classement")
                ax5.set_xlabel("Score Bayésien Ratings (/10)")
                ax5.set_ylabel("Score Sentiment DistilBERT [0-1]")
                ax5.set_title(f"Corrélation entre les deux scores  (ρ = {corr:.3f})")
                ax5.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig5)
                plt.close()

    elif df_ratings is not None and df_scores is not None:
        st.warning("comparaison_systemes.csv absent — lance le notebook 07_comparison.ipynb.")
    else:
        st.error("Fichiers manquants. Exécute les notebooks 05, 06 et 07 dans l'ordre.")


# ════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE DISTILBERT
# BUG CORRIGÉ #6 : données du MODÈLE BINAIRE (rapport Sarah)
#   - ancienne approche : 3 classes, accuracy 71.8%, F1=0.637
#   - approche retenue  : 2 classes, accuracy 97.37%, F1=0.9733
# BUG CORRIGÉ #7 : noms de fichiers visualisations corrigés
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<p class="section-header">Performance du modèle DistilBERT</p>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <strong>Approche retenue : classification binaire (Positive / Negative)</strong><br>
    Labels issus de la structure du formulaire Booking.com — indépendants des notes numériques.<br>
    Cette approche garantit que le Système 2 capte un signal réellement différent du Système 1.
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Courbes d'entraînement (3 epochs, Kaggle GPU T4)**")
        # Correction : valeurs réelles du modèle binaire (rapport Sarah, section 7.1)
        train_data = {
            "Epoch"          : [1, 2, 3],
            "Train Loss"     : [0.1188, 0.0748, 0.0558],
            "Val Loss"       : [0.0878, 0.0901, 0.1003],
            "Val F1 (macro)" : [0.9713, 0.9735, 0.9736],
        }
        df_train = pd.DataFrame(train_data)
        st.dataframe(
            df_train.style
                .highlight_min(["Val Loss"],      color="#d4edda")
                .highlight_max(["Val F1 (macro)"], color="#d4edda"),
            use_container_width=True
        )
        st.markdown("""
        <div class="info-box">
        <strong>Meilleur modèle : Epoch 2</strong> (val loss minimale = 0.0878 à epoch 1,
        mais F1 peak à epoch 2 = 0.9735).<br>
        Val loss remonte à epoch 3 → overfitting léger → epoch 2 retenu.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Évaluation sur le jeu de test (106 397 snippets)**")
        # Correction : résultats binaires (pas de classe Neutral)
        eval_data = {
            "Classe"   : ["Negative", "Positive", "Macro avg", "Weighted avg"],
            "Precision": [0.96, 0.98, 0.97, 0.97],
            "Recall"   : [0.98, 0.97, 0.97, 0.97],
            "F1"       : [0.97, 0.98, 0.9733, 0.9737],
            "Support"  : [46405, 59992, 106397, 106397],
        }
        st.dataframe(pd.DataFrame(eval_data), use_container_width=True)

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Accuracy", "97.37%")   # Correction : était 71.8%
        c_b.metric("F1 Macro", "0.9733")   # Correction : était 0.637
        c_c.metric("F1 Weighted", "0.9737") # Correction : était 0.730

    with col_p2:
        st.markdown("**Distribution des prédictions DistilBERT sur 709 313 snippets**")
        # Correction : modèle binaire — pas de classe Neutral
        pred_data = {
            "Sentiment"  : ["Positive", "Negative"],
            "Count"      : [399453, 309860],
            "Pourcentage": ["56.3%", "43.7%"],
        }
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True)

        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie(
            [399453, 309860],
            labels=["Positive\n56.3%", "Negative\n43.7%"],
            colors=["#2e7d32", "#c0392b"],
            autopct="", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax_pie.set_title("Répartition des labels dans le dataset", fontsize=10)
        st.pyplot(fig_pie)
        plt.close()

        st.markdown("**Distribution de confiance P(Positive) sur le jeu de test**")
        conf_data = {
            "Niveau de confiance": ["Haute (> 90%)", "Moyenne (70-90%)", "Faible (50-70%)"],
            "Snippets"           : [104572, 1237, 588],
            "Pourcentage"        : ["98.3%", "1.2%", "0.6%"],
        }
        st.dataframe(pd.DataFrame(conf_data), use_container_width=True)
        st.markdown("""
        <div class="info-box">
        98.3% des snippets classifiés avec confiance > 90% → distribution bimodale
        (très proche de 0 ou de 1) → le modèle est bien calibré sur ces données proprement annotées.
        </div>
        """, unsafe_allow_html=True)

    # ── Visualisations sauvegardées ───────────────────────────
    st.markdown("---")
    st.markdown("**Visualisations générées par les notebooks**")
    viz_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "visualizations")
    import glob

    viz_files = [
        ("Matrice de confusion DistilBERT", "confusion_matrix_distilbert"),
        ("Courbes d'entraînement",          "training_curve_distilbert"),
        ("Distribution sentiments",         "sentiment_score_distribution"),
        ("Top 15 sentiment",                "top15_sentiment"),
        ("Confidence par hôtel",            "confidence_by_hotel"),
        ("Distribution globale",            "confidence_distribution_full"),
    ]

    cols_viz = st.columns(3)
    for idx, (label, prefix) in enumerate(viz_files):
        with cols_viz[idx % 3]:
            matches = sorted(glob.glob(os.path.join(viz_base, prefix + "*.png")))
            if matches:
                img = plt.imread(matches[0])
                fig_v, ax_v = plt.subplots()
                ax_v.imshow(img)
                ax_v.axis("off")
                ax_v.set_title(label, fontsize=9)
                st.pyplot(fig_v)
                plt.close(fig_v)
            else:
                st.markdown(
                    f"<small style='color:#aaa;'>📷 {label}<br><em>Non trouvée</em></small>",
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — À PROPOS
# BUG CORRIGÉ #8 : architecture corrigée (labels binaires,
# bons chiffres de split, 709k snippets)
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">À propos du projet</p>',
                unsafe_allow_html=True)

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
| **Sarah Djerrab** | DistilBERT fine-tuning + Système Sentiments |
| **Khaoula** | Système Ratings + Comparaison + App |

---

**Dataset — Hotel Reviews**

| | |
|---|---|
| Source | Kaggle — Hotel Reviews Europe |
| Avis bruts | 515 287 (Hotel_Reviews.csv) |
| Snippets après split | 709 313 |
| Hôtels | 1 492 |
| Train / Val / Test | ~496k / ~106k / ~106k |
| Labels | **Positive / Negative** (structure formulaire) |
        """)

    with col_a2:
        st.markdown("""
**Architecture du pipeline**

```
Hotel_Reviews.csv (brut, 515k avis)
       ↓
02_preprocessing.ipynb  (Tasnim)
— Nettoyage, lemmatisation, stop words
— Séparation Positive_Review / Negative_Review
— Labels : Positive / Negative  ← structure formulaire
       ↓
04_distilbert.ipynb  (Sarah — Kaggle GPU T4)
— Fine-tuning DistilBERT binaire, 3 epochs
— 400 000 snippets d'entraînement
— Accuracy test : 97.37%, F1 macro : 0.9733
       ↓
05_inference_kaggle.ipynb  (Sarah — Kaggle GPU T4)
— Prédiction sur 709 313 snippets (~10 min)
— Sauvegarde prob_positive, prob_negative
       ↓
05_recommendation.ipynb  (Sarah)
— sentiment_score = mean(P(Positive)) par hôtel
— Filtre ≥ 50 snippets → 1 391 hôtels
       ↓
06_rating_recomm.ipynb  (Khaoula)
— Bayesian Average des Reviewer_Score → 1 492 hôtels
       ↓
07_comparison.ipynb  (Khaoula)
— Corrélation Spearman entre les deux classements
— Hôtels divergents (Rating ≠ Sentiment)
— Export comparaison_systemes.csv
```

**Choix technique clé**

Les labels DistilBERT viennent de la **structure du formulaire** Booking.com
(`Positive_Review` → Positive, `Negative_Review` → Negative), **indépendants** des
notes numériques. Cela garantit que le Système 2 mesure un signal
réellement différent du Système 1 — ce qui justifie leur comparaison.
        """)

    st.markdown("---")
    st.markdown(
        "<center><small>Interface développée avec Streamlit · "
        "Projet NLP 2025/2026 · Université 8 Mai 1945 Guelma</small></center>",
        unsafe_allow_html=True)