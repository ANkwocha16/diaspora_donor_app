# app.py — Diaspora Donor Recommender (Streamlit Cloud, artifacts-first)
# Uses precomputed CF estimates from artifacts/cf_estimates.csv.gz (if present)

import os, re, json, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ----------------------------- small charts / style -----------------------------
FIG_XS = (2.2, 1.6)   # tiny stat charts
FIG_S  = (2.8, 1.9)   # small bar charts

st.set_page_config(
    page_title="Diaspora Donor Recommender System",
    page_icon="🤝",
    layout="wide",
)

# ----------------------------- helpers (single definitions) -----------------------------
def take_colors(n: int):
    if n <= 0: return []
    return (["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#FFC000", "#5B9BD5",
             "#C00000", "#00B0F0", "#7F7F7F", "#92D050"] * ((n // 10) + 1))[:n]

def parse_multi(val):
    """Accept list or 'a;b;c' or None -> list[str]"""
    if val is None: return []
    if isinstance(val, list): return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).replace("|",";").replace(",", ";")
    return [p.strip() for p in s.split(";") if p.strip()]

def human_money(x):
    try:
        v = float(x)
        if v >= 1_000_000: return f"${v/1_000_000:.1f}M"
        if v >= 1_000:     return f"${v/1_000:.1f}k"
        return f"${int(v)}"
    except Exception:
        return str(x)

def normalize_col(df, col):
    """Add col+'_norm' scaled to [0,1] (handles constant columns)."""
    if col not in df.columns: return df
    vals = df[[col]].astype(float)
    if vals.max().item() == vals.min().item():
        df[col + "_norm"] = 0.5
    else:
        scaler = MinMaxScaler()
        df[col + "_norm"] = scaler.fit_transform(vals)
    return df

def has_rows(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def safe_df(df):
    return df if has_rows(df) else pd.DataFrame([{"Result":"N/A"}])

def cosine_score(vec_1D, proj_mat_2D):
    """vec_1D: (1, d), proj_mat_2D: (n, d) -> (n,1) cosine similarities"""
    cs = cosine_similarity(vec_1D, proj_mat_2D)
    return cs.ravel()

def build_proj_vectors_on_fly(projects: pd.DataFrame):
    """One-hot encode region + sector_focus for content scoring."""
    regions = sorted(projects.get("region", pd.Series(dtype=str)).dropna().unique().tolist())
    sectors = sorted(projects.get("sector_focus", pd.Series(dtype=str)).dropna().unique().tolist())
    region_cols = [f"region_{r}" for r in regions]
    sector_cols = [f"sector_{s}" for s in sectors]
    rows = []
    for _, r in projects.iterrows():
        v = {c: 0 for c in region_cols + sector_cols}
        if pd.notna(r.get("region")):
            k = f"region_{r['region']}"
            if k in v: v[k] = 1
        if pd.notna(r.get("sector_focus")):
            k = f"sector_{r['sector_focus']}"
            if k in v: v[k] = 1
        v["project_id"] = r["project_id"]
        rows.append(v)
    pv = pd.DataFrame(rows)
    feat_cols = [c for c in pv.columns if c != "project_id"]
    return pv, feat_cols

def build_donor_vector_from_prefs(pref_regions, pref_sectors, feature_cols):
    v = {c:0 for c in feature_cols}
    for r in pref_regions:
        k = f"region_{r}"; 
        if k in v: v[k] = 1
    for s in pref_sectors:
        k = f"sector_{s}"; 
        if k in v: v[k] = 1
    return np.array([[v[c] for c in feature_cols]], dtype=float)  # shape (1, d)

def rule_score(donor_row, proj_row):
    """Simple heuristic: region match + sector match + funding closeness + popularity."""
    s = 0.0
    r_prefs = set(parse_multi(donor_row.get("region_preference")))
    s_prefs = set(parse_multi(donor_row.get("sector_preference")))
    if proj_row.get("region") in r_prefs: s += 0.5
    if proj_row.get("sector_focus") in s_prefs: s += 0.5

    try:
        pref_target = float(donor_row.get("preferred_target", np.nan))
        if not np.isnan(pref_target):
            s += 0.1 * (1.0 / (1.0 + abs(float(proj_row.get("funding_target",0)) - pref_target)))
    except Exception:
        pass

    try:
        cap = float(donor_row.get("budget_cap", np.nan))
        if not np.isnan(cap) and pd.notna(proj_row.get("funding_target")):
            if float(proj_row.get("funding_target",0)) <= cap:
                s += 0.15
    except Exception:
        pass

    pop = float(proj_row.get("popularity", 0.0))
    s += min(pop, 1.0) * 0.2
    bt = str(donor_row.get("behaviour_type","")).lower()
    if "selective" in bt: s *= 1.05
    elif "active" in bt: s *= 1.02
    return s

# ----------------------------- data loading -----------------------------
BASE = "artifacts"

@st.cache_data(show_spinner=False)
def load_core(base):
    # find donors/proj CSVs (support multiple names)
    donors_path  = os.path.join(base, "donors_5000.csv")   if os.path.exists(os.path.join(base,"donors_5000.csv"))   else os.path.join(base,"donors.csv")
    projects_path= os.path.join(base, "projects_2000.csv") if os.path.exists(os.path.join(base,"projects_2000.csv")) else os.path.join(base,"projects.csv")

    donors   = pd.read_csv(donors_path)   if os.path.exists(donors_path)   else pd.DataFrame()
    projects = pd.read_csv(projects_path) if os.path.exists(projects_path) else pd.DataFrame()

    # normalize column names (lowercase)
    donors.columns   = [c.strip().lower() for c in donors.columns]
    projects.columns = [c.strip().lower() for c in projects.columns]

    # normalize ids -> strings "DRxxxx", "PRxxxx" if needed
    def norm_dnr(x):
        s = str(x).strip().upper()
        m = re.search(r"(\d+)$", s)
        return "DR" + (m.group(1).zfill(4) if m else re.sub(r"[^A-Z0-9]+","",s))
    if "donor_id" not in donors.columns:
        # try guess
        cand = [c for c in donors.columns if "donor" in c or c=="id"]
        if cand:
            donors.rename(columns={cand[0]: "donor_id"}, inplace=True)
    donors["donor_id"] = donors["donor_id"].apply(norm_dnr)

    def norm_prj(x):
        s = str(x).strip().upper()
        m = re.search(r"(\d+)$", s)
        return "PR" + (m.group(1).zfill(4) if m else re.sub(r"[^A-Z0-9]+","",s))
    if "project_id" not in projects.columns:
        cand = [c for c in projects.columns if "project" in c or c=="id"]
        if cand:
            projects.rename(columns={cand[0]: "project_id"}, inplace=True)
    projects["project_id"] = projects["project_id"].apply(norm_prj)

    # optional: interactions
    inter = None
    for name in ["interactions.csv", "ratings.csv", "ratings_5000x2000.csv", "synthetic_interactions_5000x2000.csv"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            inter = pd.read_csv(p)
            break
    if has_rows(inter):
        inter.columns = [c.strip().lower() for c in inter.columns]
        # normalize to donor_id, project_id, score
        rename_map = {}
        if "donor id" in inter.columns: rename_map["donor id"]="donor_id"
        if "project id" in inter.columns: rename_map["project id"]="project_id"
        if "rating" in inter.columns: rename_map["rating"]="score"
        inter.rename(columns=rename_map, inplace=True)
        if "donor_id" not in inter.columns or "project_id" not in inter.columns:
            inter = None
        else:
            inter["donor_id"]   = inter["donor_id"].apply(norm_dnr)
            inter["project_id"] = inter["project_id"].apply(norm_prj)
            if "score" not in inter.columns:
                inter["score"] = 1.0

    # precomputed CF estimates
    cf = None
    for name in ["cf_estimates.csv.gz", "cf_estimates.csv"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            cf = pd.read_csv(p)
            break
    if has_rows(cf):
        cf.columns = [c.strip().lower() for c in cf.columns]
        rename_cf = {}
        if "est" in cf.columns: rename_cf["est"] = "est"
        if "prediction" in cf.columns: rename_cf["prediction"] = "est"
        if "user_id" in cf.columns: rename_cf["user_id"]="donor_id"
        if "item_id" in cf.columns: rename_cf["item_id"]="project_id"
        if rename_cf: cf.rename(columns=rename_cf, inplace=True)
        # keep only needed
        need = [c for c in ["donor_id","project_id","est"] if c in cf.columns]
        cf = cf[need].copy()
        if "donor_id" in cf.columns:   cf["donor_id"]   = cf["donor_id"].apply(norm_dnr)
        if "project_id" in cf.columns: cf["project_id"] = cf["project_id"].apply(norm_prj)

    # project vectors (optional precomputed parquet)
    proj_vecs, feat_cols = None, None
    pv_path = os.path.join(base, "proj_vectors.parquet")
    if os.path.exists(pv_path):
        pv = pd.read_parquet(pv_path)
        proj_vecs = pv.drop(columns=["project_id"])
        feat_cols = proj_vecs.columns.tolist()
        proj_vecs = proj_vecs.values.astype(float)
        pv_lookup = pv[["project_id"]].copy()
    else:
        pv, feat_cols = build_proj_vectors_on_fly(projects)
        pv_lookup = pv[["project_id"]].copy()
        proj_vecs = pv.drop(columns=["project_id"]).values.astype(float)

    # tiny enrichments
    projects["popularity"] = projects.get("popularity", pd.Series(np.random.rand(len(projects))*0.3 + 0.5))
    normalize_col(projects, "funding_target")

    return donors, projects, inter, cf, proj_vecs, feat_cols, pv_lookup

donors, projects, interactions, cf_est, proj_vecs, FEATS, pv_lookup = load_core(BASE)

# ----------------------------- scoring / recommendations -----------------------------
def get_recs(
    donor_id: str,
    weights=(0.33, 0.34, 0.33),        # rule, content, cf
    topk=10,
    ethical=False,
    override_regions=None,
    override_sectors=None
):
    """Return ranked projects for donor_id with columns:
       project_id, title, region, sector_focus, rule_score, cosine_score, cf_score, hybrid_score, why
    """
    if donor_id not in set(donors["donor_id"]):
        return pd.DataFrame(), "Donor not found."

    drow = donors.loc[donors["donor_id"]==donor_id].iloc[0]

    # preferences (overrides ok)
    pref_regions = override_regions if override_regions else parse_multi(drow.get("region_preference"))
    pref_sectors = override_sectors if override_sectors else parse_multi(drow.get("sector_preference"))

    # rule score
    base = projects.copy()
    base["rule_score"] = [rule_score(drow, r) for _, r in base.iterrows()]
    normalize_col(base, "rule_score")

    # content cosine
    dv = build_donor_vector_from_prefs(pref_regions, pref_sectors, FEATS)
    base["cosine_score"] = cosine_score(dv, proj_vecs)
    normalize_col(base, "cosine_score")

    # CF score from precomputed estimates
    if has_rows(cf_est):
        cf_slice = cf_est[cf_est["donor_id"]==donor_id][["project_id","est"]].copy()
        normalize_col(cf_slice.rename(columns={"est":"cf_score"}), "cf_score")  # temp
        base = base.merge(cf_slice.rename(columns={"est":"cf_score"}), on="project_id", how="left")
    else:
        base["cf_score"] = np.nan
    base["cf_score"] = base["cf_score"].fillna(base["cosine_score"]*0.7)  # graceful fallback
    normalize_col(base, "cf_score")

    # ethical (down-weight popular ones a bit)
    if ethical:
        pop = base.get("popularity", pd.Series(0.5, index=base.index))
        base["ethical_adj"] = 1.0 - (pop - pop.min())/(pop.max()-pop.min()+1e-9) * 0.15
    else:
        base["ethical_adj"] = 1.0

    w_rule, w_cos, w_cf = weights
    base["hybrid_score"] = (
        w_rule*base["rule_score_norm"] + 
        w_cos*base["cosine_score_norm"] + 
        w_cf *base["cf_score_norm"]
    ) * base["ethical_adj"]

    # compact "why"
    reasons = []
    rset = set(pref_regions)
    sset = set(pref_sectors)
    for _, r in base.iterrows():
        parts = []
        if r.get("region") in rset: parts.append("Region match")
        if r.get("sector_focus") in sset: parts.append("Sector match")
        if r["cosine_score_norm"] > 0.8: parts.append("High content similarity")
        if r["cf_score_norm"] > 0.8: parts.append("Similar donors liked this")
        reasons.append("; ".join(parts) if parts else "Blend score")

    base["why"] = reasons

    # final rank
    cols_keep = ["project_id","title","region","sector_focus","funding_target",
                 "rule_score_norm","cosine_score_norm","cf_score_norm","hybrid_score","why"]
    # 'title' may not exist in some CSVs
    if "title" not in base.columns: base["title"] = base["project_id"]
    out = base[cols_keep].sort_values("hybrid_score", ascending=False).head(topk).reset_index(drop=True)

    return out, None

# ----------------------------- metrics (donor-level) -----------------------------
def eval_metrics_for_donor(donor_id, recs_df, k=5):
    """Return dict of precision@k, recall@k, map@k (simple), coverage@k, novelty, diversity, MAE/MSE/RMSE (where possible)."""
    K = max(1, int(k))
    universe = set(projects["project_id"].astype(str))
    topk = recs_df["project_id"].astype(str).head(K).tolist()

    # coverage@k
    covk = len(set(topk))/max(1, len(topk)) * 100.0

    # diversity@k (unique sector ratio)
    ddf = projects.set_index("project_id").reindex(topk)
    uniq_sec = ddf["sector_focus"].nunique(dropna=True) if has_rows(ddf) else 0
    divk = (uniq_sec/max(1,len(topk))) * 100.0

    # novelty (1 - popularity normalized) mean
    pop = ddf["popularity"].fillna(0.5) if has_rows(ddf) else pd.Series([0.5]*len(topk))
    nov = float((1.0 - (pop - pop.min())/(pop.max()-pop.min()+1e-9)).mean()) if len(pop) else 0.0

    # precision/recall/map need interactions as ground truth
    prec = rec = mapk = 0.0
    if has_rows(interactions):
        hist = interactions[interactions["donor_id"]==donor_id]
        if has_rows(hist):
            thr = hist["score"].median()
            relevant = set(hist.loc[hist["score"] >= thr, "project_id"].astype(str))
            if len(relevant):
                hits = [1 if pid in relevant else 0 for pid in topk]
                prec = 100.0 * (sum(hits)/len(topk) if topk else 0.0)
                rec  = 100.0 * (sum(hits)/len(relevant))
                # very light MAP@K
                running, cum = 0, 0.0
                for i, h in enumerate(hits, 1):
                    if h:
                        running += 1
                        cum += running / i
                mapk = 100.0 * (cum/max(1,sum(hits))) if sum(hits)>0 else 0.0

    # error metrics (compare CF est vs. actual)
    mae = mse = rmse = 0.0
    if has_rows(cf_est) and has_rows(interactions):
        left  = interactions[interactions["donor_id"]==donor_id][["project_id","score"]].copy()
        right = cf_est[cf_est["donor_id"]==donor_id][["project_id","est"]].copy()
        j = left.merge(right, on="project_id", how="inner")
        if has_rows(j):
            dif = (j["est"].astype(float) - j["score"].astype(float)).values
            mae = float(np.mean(np.abs(dif)))
            mse = float(np.mean(dif**2))
            rmse= float(np.sqrt(mse))

    return dict(
        precision@k=prec, recall@k=rec, map@k=mapk,
        coverage@k=covk, novelty=nov, diversity@k=divk,
        MAE=mae, MSE=mse, RMSE=rmse
    )

# ----------------------------- UI -----------------------------
st.markdown("## 🤝 Diaspora Donor Recommender System")
st.caption("Hybrid (Rule + Content Cosine + CF via **precomputed SVD**) with multi-preference controls, donor progress, metrics, diagnostics, and exports.")

# status bar
left, right = st.columns([0.4, 0.6])
with left:
    st.write(f"**Dataset & model status**")
    st.write(f"Donors: **{len(donors):,}**, Items: **{len(projects):,}**")
    st.write(f"CF source: **precomputed estimates** {'✅' if has_rows(cf_est) else '❌'}")
with right:
    st.empty()

tab_names = ["Home", "Insights", "Donor progress", "Metrics", "Why these picks", "Explore projects", "Compare algorithms", "Register donor", "Diagnostics"]
tab_home, tab_ins, tab_prog, tab_met, tab_why, tab_explore, tab_comp, tab_reg, tab_diag = st.tabs(tab_names)

# ----------------------------- HOME (find donor + recs) -----------------------------
with tab_home:
    st.subheader("Find donor and set preferences")

    # search
    q = st.text_input("Search donor (ID, name or email)")
    ddf = donors.copy()
    if q:
        ql = q.lower()
        ddf = ddf[
            ddf.apply(lambda r:
                      (ql in str(r.get("donor_id","")).lower()) or
                      (ql in str(r.get("name","")).lower()) or
                      (ql in str(r.get("email","")).lower()), axis=1)
        ]
    # label for selectbox
    if "name" not in ddf.columns: ddf["name"] = ddf["donor_id"]
    ddf["label"] = ddf.apply(lambda r: f'{r["donor_id"]} - {r["name"]} ✅', axis=1)

    donor_label = st.selectbox("Choose donor", options=ddf["label"].tolist())
    donor_id = donor_label.split(" - ")[0] if donor_label else ddf["donor_id"].iloc[0]

    # show donor card
    drow = donors.loc[donors["donor_id"]==donor_id].iloc[0]
    st.info(
        f"**{drow.get('name', donor_id)} [{donor_id}]**\n\n"
        f"{drow.get('email','')}\n\n"
        f"Behavior: **{drow.get('behaviour_type','Active')}**\n\n"
        f"Prefs — Regions: **{drow.get('region_preference','-')}**; "
        f"Sectors: **{drow.get('sector_preference','-')}**"
    )

    # preference UI
    # regions / sectors list from projects
    region_opts = sorted(projects.get("region", pd.Series([], dtype=str)).dropna().unique().tolist())
    sector_opts = sorted(projects.get("sector_focus", pd.Series([], dtype=str)).dropna().unique().tolist())
    pref_regions = st.multiselect("Preference: Regions (multi)", region_opts, default=parse_multi(drow.get("region_preference")))
    pref_sectors = st.multiselect("Preference: Sectors (multi)", sector_opts, default=parse_multi(drow.get("sector_preference")))
    budget = st.number_input("Preferred project funding target", min_value=0, value=int(drow.get("preferred_target", 0)) if pd.notna(drow.get("preferred_target", np.nan)) else 0, step=1000)
    cap = st.slider("Budget cap (filters funding target ≤)", 0, int(max(50_000, projects.get("funding_target", pd.Series([50_000])).max())), int(drow.get("budget_cap", 0)) if pd.notna(drow.get("budget_cap", np.nan)) else 0)

    # weights
    st.markdown("##### Blend weights")
    w_rule = st.slider("Rule-based", 0.0, 1.0, 0.30, 0.01)
    w_cos  = st.slider("Content (Cosine)", 0.0, 1.0, 0.40, 0.01)
    w_cf   = st.slider("Collaborative (SVD)", 0.0, 1.0, 0.30, 0.01)
    st.toggle("Ethical AI (reduce over-exposed items)", value=True, key="ethical")

    hybrid = st.toggle("Hybrid mode (blend all three)", value=True)
    weights = (w_rule, w_cos, w_cf) if hybrid else (1.0, 0.0, 0.0)

    # --- state init ---
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = pd.DataFrame()
if "recs" not in st.session_state:
    st.session_state["recs"] = pd.DataFrame()

# action buttons
col_go, col_clear = st.columns([1,1])
with col_go:
    go = st.button("Get recommendations", use_container_width=True)
with col_clear:
    clear_btn = st.button("Clear shortlist", use_container_width=True)

# clear: wipe shortlist + current recs and rerun to remove rendered cards
if clear_btn:
    st.session_state["shortlist"] = pd.DataFrame()
    st.session_state["recs"] = pd.DataFrame()
    st.info("Shortlist cleared.")
    st.rerun()   # Streamlit ≥1.27; if older, use st.experimental_rerun()

st.subheader("Top recommendations")

if go:
    recs, err = get_recs(
        donor_id,
        weights=weights,
        topk=10,
        ethical=st.session_state["ethical"],
        override_regions=pref_regions,
        override_sectors=pref_sectors
    )
    if err:
        st.warning(err)
        st.session_state["recs"] = pd.DataFrame()
    else:
        # apply budget cap filter if any
        if cap and "funding_target" in recs.columns:
            recs = recs[recs["funding_target"] <= cap]
        st.session_state["recs"] = recs.reset_index(drop=True)

# render whatever is in session
recs = st.session_state["recs"]
if recs.empty:
    st.info("Click **Get recommendations**.")
else:
    for i, row in recs.iterrows():
        st.markdown(
            f"""
            <div style="border:1px solid #e7e7e7;border-radius:10px;padding:8px 10px;margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div><b>{i+1}. {row.get('title', row['project_id'])}</b><br>
                        <span style="font-size:12px;color:#666">{row.get('region','')} · {row.get('sector_focus','')}</span>
                    </div>
                    <div style="font-weight:700">{row.get('hybrid_score',0):.2f}</div>
                </div>
                <div style="margin-top:6px;color:#666">Why: {row.get('why','')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        else:
            st.info("No recommendations yet. Adjust preferences and try again.")

# ----------------------------- INSIGHTS -----------------------------
with tab_ins:
    st.subheader("Insights")
    # tiny hist on funding_target
    df = projects.copy()
    if "funding_target" in df.columns and not df["funding_target"].isna().all():
        fig, ax = plt.subplots(figsize=FIG_XS)
        ax.hist(df["funding_target"].astype(float), bins=15)
        ax.set_title("Funding target distribution")
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig)
    else:
        st.info("No numeric funding target found.")

# ----------------------------- DONOR PROGRESS -----------------------------
with tab_prog:
    st.subheader("Donor progress")
    st.write("Light placeholder (no persistent progress store in Cloud).")

# ----------------------------- METRICS -----------------------------
with tab_met:
    st.subheader("Evaluation metrics (donor-level)")
    k = st.number_input("Top-K for evaluation", 1, 50, 5, step=1)
    donor_for_eval = st.selectbox("Donor", donors["donor_id"].tolist(), index=0, key="eval_dn")
    # generate recs silently
    recs_k, _ = get_recs(donor_for_eval, topk=int(k), weights=(0.33,0.34,0.33), ethical=True)
    m = eval_metrics_for_donor(donor_for_eval, recs_k, k=int(k))

    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Precision@K", f"{m['precision@k']:.1f}%")
        st.metric("Coverage@K", f"{m['coverage@k']:.1f}%")
        st.metric("MAE", f"{m['MAE']:.3f}")
    with c2:
        st.metric("Recall@K", f"{m['recall@k']:.1f}%")
        st.metric("Novelty ↑", f"{m['novelty']:.2f}")
        st.metric("MSE", f"{m['MSE']:.3f}")
    with c3:
        st.metric("MAP@K", f"{m['map@k']:.1f}%")
        st.metric("Diversity@K", f"{m['diversity@k']:.1f}%")
        st.metric("RMSE", f"{m['RMSE']:.3f}")

    # small bar: avg component weights from this donor’s recs
    if has_rows(recs_k):
        tmp = recs_k[["rule_score_norm","cosine_score_norm","cf_score_norm"]].mean()
        fig, ax = plt.subplots(figsize=FIG_S)
        ax.bar(["Rule","Content","CF"], tmp.values)
        for i,v in enumerate(tmp.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title("Avg score components")
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig)

# ----------------------------- WHY THESE PICKS -----------------------------
with tab_why:
    st.subheader("Why these picks")
    st.write("The hybrid score blends Rule (preferences fit), Content (cosine similarity on region/sector), and CF (precomputed estimates).")

# ----------------------------- EXPLORE PROJECTS -----------------------------
with tab_explore:
    st.subheader("Explore projects")
    show = projects[["project_id","title","region","sector_focus","funding_target"]].copy()
    show["funding_target"] = show["funding_target"].apply(human_money) if "funding_target" in show.columns else "-"
    st.dataframe(show, use_container_width=True, hide_index=True)

# ----------------------------- COMPARE ALGORITHMS -----------------------------
with tab_comp:
    st.subheader("Compare algorithms")
    donor_c = st.selectbox("Donor", donors["donor_id"].tolist(), key="cmp_dn")
    def run_algo(weights, label):
        recs, _ = get_recs(donor_c, weights=weights, topk=5, ethical=True)
        if not has_rows(recs): return pd.DataFrame()
        out = recs[["title","region","sector_focus","hybrid_score"]].rename(columns={"hybrid_score": f"{label} score"})
        return out

    cA, cB = st.columns(2)
    with cA:
        st.write("Rule-based Top-5")
        st.dataframe(safe_df(run_algo((1,0,0), "Rule")), use_container_width=True, hide_index=True)
        st.write("Content (Cosine) Top-5")
        st.dataframe(safe_df(run_algo((0,1,0), "Content")), use_container_width=True, hide_index=True)
    with cB:
        st.write("Collaborative (CF) Top-5")
        st.dataframe(safe_df(run_algo((0,0,1), "CF")), use_container_width=True, hide_index=True)
        st.write("Hybrid Top-5")
        st.dataframe(safe_df(run_algo((0.33,0.34,0.33), "Hybrid")), use_container_width=True, hide_index=True)

# ----------------------------- REGISTER DONOR -----------------------------
with tab_reg:
    st.subheader("Register donor")
    with st.form("reg_form"):
        nm = st.text_input("Name")
        em = st.text_input("Email")
        regs = st.multiselect("Region preference", sorted(projects["region"].dropna().unique().tolist()))
        secs = st.multiselect("Sector preference", sorted(projects["sector_focus"].dropna().unique().tolist()))
        pref_tgt = st.number_input("Preferred target ($)", min_value=0, value=0, step=1000)
        beh = st.selectbox("Behaviour type", ["Active","Passive","Selective"])
        ok = st.form_submit_button("Preview")
    if ok:
        st.success("Registered (preview only on Cloud).")
        st.json({
            "name": nm, "email": em,
            "region_preference": ";".join(regs), "sector_preference": ";".join(secs),
            "preferred_target": pref_tgt, "behaviour_type": beh
        })

# ----------------------------- DIAGNOSTICS -----------------------------
with tab_diag:
    st.subheader("Diagnostics")
    st.write(f"CF estimates present: {'✅' if has_rows(cf_est) else '❌'}")
    st.write(f"Interactions present: {'✅' if has_rows(interactions) else '❌'}")
    if has_rows(cf_est):
        st.write("Sample of CF estimates:")
        st.dataframe(cf_est.head(10), use_container_width=True, hide_index=True)
    st.write(f"Feature columns used: {len(FEATS)}")

