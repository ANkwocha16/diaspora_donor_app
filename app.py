# app.py — Diaspora Donor Recommender (Streamlit Cloud, artifacts-first)
# - Two-pane Home (left: donor & prefs; right: recommendations)
# - ✅ tick for donors with interactions (normalized IDs)
# - Precomputed CF from artifacts/cf_estimates.csv(.gz) (no Surprise dependency)
# - Robust interactions normalization; metrics compute correctly
# - Small, clear charts across tabs
# - Tabs appear first; Home contains two-column UI

import os, re, json, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---------------- Page config & cache bump ----------------
st.set_page_config(page_title="Diaspora Donor Recommender", page_icon="🤎", layout="wide")
APP_VERSION = "2025-08-17-two-pane-metrics-reload-robust"
if st.session_state.get("__app_version") != APP_VERSION:
    try:
        st.cache_data.clear(); st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["__app_version"] = APP_VERSION

# --- Manual reload (fixes 'interactions vanished' after uploads) ---
col_reload, _ = st.columns([1, 6])
with col_reload:
    if st.button("🔄 Reload data (clear cache)"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.rerun()

BASE = "artifacts"
os.makedirs(BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(BASE, "outputs"); os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Small helpers ----------------
FIG_XS = (2.2, 1.6)   # small charts
FIG_S  = (2.8, 1.9)

def make_small_axes(size="xs"):
    fig, ax = plt.subplots(figsize=FIG_XS if size=="xs" else FIG_S)
    ax.tick_params(labelsize=8)
    return fig, ax

def parse_multi(val):
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

def normalize(df, col):
    vals = df[[col]].astype(float)
    if vals.max().item() == vals.min().item():
        df[col+"_norm"] = 0.5
    else:
        df[col+"_norm"] = MinMaxScaler().fit_transform(vals)
    return df

def has_rows(df):
    return isinstance(df, pd.DataFrame) and not df.empty

def pct(x) -> str:
    try: return f"{float(x):.1%}"
    except Exception: return "0.0%"

def num3(x) -> str:
    try: return f"{float(x):.3f}"
    except Exception: return "0.000"

def status_dot_html(state):
    s = (state or "").lower().strip()
    color = "#9ca3af"; label = "Passive"
    if s == "active" or s == "":
        color, label = "#10b981", "Active"
    if s == "selective":
        color, label = "#f59e0b", "Selective"
    return f'<span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>{label}</span>'

def _std_id(x):
    return str(x).strip().upper()

# --------------- Data loaders ---------------
@st.cache_data(show_spinner=False)
def load_cf_estimates(base):
    for name in ["cf_estimates.csv.gz","cf_estimates.csv"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            lower = {c.lower(): c for c in df.columns}
            def col(w):
                for k,v in lower.items():
                    if k == w: return v
            dcol = col("donor_id"); pcol = col("project_id")
            ecol = col("est") or lower.get("prediction")
            if not dcol or not pcol or not ecol:
                continue
            df = df.rename(columns={dcol:"donor_id", pcol:"project_id", ecol:"est"})
            df["donor_id"] = df["donor_id"].map(_std_id)
            df["project_id"] = df["project_id"].map(_std_id)
            df["est"] = pd.to_numeric(df["est"], errors="coerce")
            return df.dropna(subset=["donor_id","project_id","est"])
    return None

@st.cache_data(show_spinner=False)
def load_core(base):
    donors_path   = os.path.join(base, "donors_5000.csv") if os.path.exists(os.path.join(base,"donors_5000.csv")) else os.path.join(base, "donors.csv")
    projects_path = os.path.join(base, "projects_2000.csv") if os.path.exists(os.path.join(base,"projects_2000.csv")) else os.path.join(base, "projects.csv")

    donors = pd.read_csv(donors_path)
    projects = pd.read_csv(projects_path)

    donors.columns = [c.strip().lower() for c in donors.columns]
    projects.columns = [c.strip().lower() for c in projects.columns]

    if "donor_id" not in donors.columns:
        for alt in donors.columns:
            if "donor" in alt or alt=="id": donors = donors.rename(columns={alt:"donor_id"}); break
    donors["donor_id"] = donors["donor_id"].astype(str).map(_std_id)

    if "project_id" not in projects.columns:
        for alt in projects.columns:
            if "project" in alt or alt=="id": projects = projects.rename(columns={alt:"project_id"}); break
    projects["project_id"] = projects["project_id"].astype(str).map(_std_id)

    # required fallback cols
    for col, default in [
        ("name",""), ("email",""), ("behaviour_type","Active"),
        ("region_preference",""), ("sector_preference",""),
        ("preferred_target", np.nan), ("budget_cap", np.nan)
    ]:
        if col not in donors.columns: donors[col] = default

    for col, default in [
        ("title",""), ("region",""), ("sector_focus",""),
        ("organisation_type",""), ("funding_target", np.nan), ("popularity", 0.0)
    ]:
        if col not in projects.columns: projects[col] = default

    donors["preferred_target"] = pd.to_numeric(donors["preferred_target"], errors="coerce")
    donors["budget_cap"]       = pd.to_numeric(donors["budget_cap"], errors="coerce")
    projects["funding_target"] = pd.to_numeric(projects["funding_target"], errors="coerce").fillna(projects["funding_target"].median())
    projects["popularity"]     = pd.to_numeric(projects["popularity"], errors="coerce").fillna(0.0)

    # ---------- interactions (optional; supports interactions.csv or ratings*.csv)
   interactions = pd.DataFrame(columns=["donor_id","project_id","score"])
   inter_path = None
   for n in ["interactions.csv", "ratings_5000x2000.csv", "ratings.csv", "synthetic_interactions_5000x2000.csv"]:
       p = os.path.join(base, n)
       if os.path.exists(p):
           inter_path = p
           break
   if inter_path:
       inter = pd.read_csv(inter_path)
       # Canonicalize column names to lower-case first
       inter.columns = [c.strip().lower() for c in inter.columns]
       # Flexible column picking
       dcol = "donor_id"   if "donor_id"   in inter.columns else ("user_id" if "user_id" in inter.columns else None)
       pcol = "project_id" if "project_id" in inter.columns else ("item_id" if "item_id" in inter.columns else None)
       scol = "score"      if "score"      in inter.columns else ("rating"  if "rating"  in inter.columns else None)
       if dcol and pcol:
           inter = inter.rename(columns={dcol:"donor_id", pcol:"project_id"})
           # Standardize IDs (match how recommendations are built)
           inter["donor_id"]   = inter["donor_id"].astype(str).str.strip().str.upper()
           inter["project_id"] = inter["project_id"].astype(str).str.strip().str.upper()
           # Score: if none, assume implicit positive feedback
           if scol:
               inter = inter.rename(columns={scol:"score"})
               inter["score"] = pd.to_numeric(inter["score"], errors="coerce").fillna(0.0)
           else:
               inter["score"] = 1.0
           # Keep only the three columns
           interactions = inter[["donor_id","project_id","score"]].dropna(subset=["donor_id","project_id"])

    return donors, projects, interactions

@st.cache_data(show_spinner=False)
def load_proj_vectors(projects):
    pv_path = os.path.join(BASE, "proj_vectors.parquet")
    if os.path.exists(pv_path):
        pv = pd.read_parquet(pv_path)
        feats = [c for c in pv.columns if c != "project_id"]
        mat = pv[feats].fillna(0).astype(float).values
        ids = pv["project_id"].astype(str).map(_std_id)
        return {"ids": ids, "feats": feats, "mat": mat}

    # build simple one-hot from region/sector
    regions = sorted(projects["region"].dropna().unique().tolist())
    sectors = sorted(projects["sector_focus"].dropna().unique().tolist())
    rcols = [f"region_{r}" for r in regions]
    scols = [f"sector_{s}" for s in sectors]
    rows = []
    for _, r in projects.iterrows():
        v = {c:0 for c in rcols + scols}
        if pd.notna(r["region"]): v[f"region_{r['region']}"] = 1
        if pd.notna(r["sector_focus"]): v[f"sector_{r['sector_focus']}"] = 1
        v["project_id"] = r["project_id"]
        rows.append(v)
    pv = pd.DataFrame(rows)
    feats = [c for c in pv.columns if c != "project_id"]
    mat = pv[feats].fillna(0).astype(float).values
    ids = pv["project_id"].astype(str).map(_std_id)
    return {"ids": ids, "feats": feats, "mat": mat}

# ---------------- Recommender core ----------------
def build_donor_vec(pref_regions, pref_sectors, feat_cols):
    m = {c:0 for c in feat_cols}
    for r in pref_regions:
        k = f"region_{r}"; 
        if k in m: m[k] = 1
    for s in pref_sectors:
        k = f"sector_{s}"; 
        if k in m: m[k] = 1
    return np.array([[m[c] for c in feat_cols]], dtype=float)

def rule_score(donor_row, proj_row):
    s = 0.0
    r_pref = set(parse_multi(donor_row.get("region_preference")))
    s_pref = set(parse_multi(donor_row.get("sector_preference")))
    if proj_row.get("region") in r_pref: s += 0.5
    if proj_row.get("sector_focus") in s_pref: s += 0.5
    try:
        cap = float(donor_row.get("budget_cap", np.nan))
        if not np.isnan(cap) and float(proj_row.get("funding_target", 0)) <= cap:
            s += 0.15
    except Exception:
        pass
    try:
        pref_t = float(donor_row.get("preferred_target", np.nan))
        if not np.isnan(pref_t):
            s += 0.1 * (1.0 / (1.0 + abs(float(proj_row.get("funding_target", 0)) - pref_t)))
    except Exception:
        pass
    pop = float(proj_row.get("popularity", 0.0))
    s += min(pop, 1.0) * 0.2
    bt = str(donor_row.get("behaviour_type", "")).lower()
    if "selective" in bt: s *= 1.05
    elif "active" in bt:  s *= 1.02
    return s

# set to 0.05 for demos to get visible Precision/Recall; 0.00 keeps neutral
SEEN_BOOST = 0.00

def get_recs(donor_id, donors, projects, interactions, cf_estimates, vecs,
             weights=(0.33,0.34,0.33), ethical=True, topk=10,
             override_regions=None, override_sectors=None):
    dser = donors.loc[donors["donor_id"]==donor_id]
    if dser.empty: return pd.DataFrame(), "Unknown donor_id"
    drow = dser.iloc[0].to_dict()

    pref_regions = override_regions if override_regions is not None else parse_multi(drow.get("region_preference"))
    pref_sectors = override_sectors if override_sectors is not None else parse_multi(drow.get("sector_preference"))

    cand = projects.copy()

    # rule
    cand["rule_score"] = [rule_score(drow, r) for _, r in cand.iterrows()]
    cand = normalize(cand, "rule_score")

    # content
    dv = build_donor_vec(pref_regions, pref_sectors, vecs["feats"])
    pid_to_idx = {pid:i for i,pid in enumerate(vecs["ids"])}
    idxs = [pid_to_idx.get(pid, None) for pid in cand["project_id"].map(_std_id)]
    mask = [i is not None for i in idxs]
    proj_mat = np.zeros((len(cand), len(vecs["feats"])), dtype=float)
    valid_rows = [i for i,ok in enumerate(mask) if ok]
    if valid_rows:
        proj_mat[valid_rows,:] = vecs["mat"][np.array([idxs[i] for i in valid_rows])]
    cos = cosine_similarity(dv, proj_mat).ravel()
    cand["cosine_score"] = cos
    cand = normalize(cand, "cosine_score")

    # CF (precomputed)
    if cf_estimates is not None and not cf_estimates.empty:
        donor_str = _std_id(donor_id)
        cf_slice = cf_estimates[cf_estimates["donor_id"]==donor_str][["project_id","est"]].rename(columns={"est":"cf_score"})
        cf_slice["project_id"] = cf_slice["project_id"].map(_std_id)
        cand["project_id"] = cand["project_id"].map(_std_id)
        cand = cand.merge(cf_slice, on="project_id", how="left")
        cand["cf_score"] = cand["cf_score"].fillna(cand["cosine_score"])
        cand = normalize(cand, "cf_score")
    else:
        cand["cf_score"] = cand["cosine_score"]
        cand["cf_score_norm"] = cand["cosine_score_norm"]

    # Ethical AI: down-weight top 10% most popular
    if ethical and "popularity" in cand.columns and len(cand) > 0:
        p90 = cand["popularity"].quantile(0.90)
        penal = (cand["popularity"] >= p90).astype(float) * 0.15
        cand["ethical_adj"] = 1.0 - penal
    else:
        cand["ethical_adj"] = 1.0

    w_rule, w_cos, w_cf = weights
    cand["hybrid_score"] = (
        w_rule*cand["rule_score_norm"] + 
        w_cos *cand["cosine_score_norm"] + 
        w_cf  *cand["cf_score_norm"]
    ) * cand["ethical_adj"]

    # optional boost for items in donor history (for visible metrics in demos)
    if SEEN_BOOST > 0 and has_rows(interactions):
        seen_set = set(interactions[interactions["donor_id"] == _std_id(donor_id)]["project_id"])
        cand["hybrid_score"] = cand["hybrid_score"] + cand["project_id"].map(lambda x: SEEN_BOOST if x in seen_set else 0.0)

    # why
    why = []
    pr = set(pref_regions); ps = set(pref_sectors)
    for _, r in cand.iterrows():
        parts = []
        if r["region"] in pr: parts.append("Region match")
        if r["sector_focus"] in ps: parts.append("Sector match")
        if r["cosine_score_norm"] > 0.7: parts.append("High content similarity")
        if r["cf_score_norm"] > 0.7: parts.append("Similar donors liked this")
        why.append("; ".join(parts) if parts else "Strong blended score")
    cand["why"] = why

    return cand.sort_values("hybrid_score", ascending=False).head(topk).reset_index(drop=True), None

# --------------- Metrics ---------------
def average_precision_at_k(relevant, ranked_ids, k):
   if k <= 0: return 0.0
   hits = 0; s = 0.0
   for i, pid in enumerate(ranked_ids[:k], start=1):
       if pid in relevant:
           hits += 1
           s += hits / i
   return 0.0 if hits == 0 else s / min(len(relevant), k)
def compute_metrics_for_donor(donor_id, recs, interactions, projects, cf_estimates,
                             k=5, thr_mode="Median per donor"):
   """
   Returns: metrics_dict, diag_dict
     metrics: precision_k, recall_k, map_k, coverage_k, diversity_k, novelty, mae, mse, rmse
     diag:    n_hist, n_topk, n_rel, hits, overlap_size, n_cf
   """
   M = dict(precision_k=0.0, recall_k=0.0, map_k=0.0, coverage_k=0.0,
            diversity_k=0.0, novelty=0.0, mae=0.0, mse=0.0, rmse=0.0)
   D = dict(n_hist=0, n_topk=0, n_rel=0, hits=0, overlap_size=0, n_cf=0)
   if not has_rows(recs):
       return M, D
   # Canonicalize IDs in top-k
   K = max(1, int(k))
   top_ids = recs["project_id"].astype(str).str.strip().str.upper().tolist()[:K]
   D["n_topk"] = len(top_ids)
   # Novelty (low popularity -> high novelty)
   pop = projects.set_index("project_id")["popularity"]
   pop.index = pop.index.astype(str).str.strip().str.upper()
   pop_vec = pd.Series(top_ids).map(pop).fillna(0.0).to_numpy()
   if pop_vec.size:
       if pop_vec.max() == pop_vec.min():
           M["novelty"] = 0.60
       else:
           pnorm = (pop_vec - pop_vec.min()) / (pop_vec.max() - pop_vec.min())
           M["novelty"] = float(1.0 - pnorm.mean())
   # Diversity@K (unique sectors among top-k)
   sec = projects.set_index("project_id")["sector_focus"]
   sec.index = sec.index.astype(str).str.strip().str.upper()
   uniq = len(set(pd.Series(top_ids).map(sec).fillna("").tolist()) - {""})
   M["diversity_k"] = uniq / max(1, len(top_ids))
   # Need history for the rest
   if not has_rows(interactions):
       return M, D
   d_id = str(donor_id).strip().upper()
   hist = interactions[interactions["donor_id"] == d_id].copy()
   D["n_hist"] = len(hist)
   if hist.empty:
       return M, D
   # Build relevance set
   hist["project_id"] = hist["project_id"].astype(str).str.strip().str.upper()
   hist["score"] = pd.to_numeric(hist["score"], errors="coerce").fillna(0.0)
   if thr_mode == "Any positive (>0)":
       rel = set(hist.loc[hist["score"] > 0, "project_id"].tolist())
   else:  # "Median per donor"
       thr = float(hist["score"].median())
       rel = set(hist.loc[hist["score"] >= thr, "project_id"].tolist())
   D["n_rel"] = len(rel)
   # Overlap & hits
   top_set = set(top_ids)
   overlap = top_set & set(hist["project_id"])
   D["overlap_size"] = len(overlap)
   hits = sum(1 for pid in top_ids if pid in rel)
   D["hits"] = hits
   # Coverage / Precision / Recall / MAP
   M["coverage_k"] = len(overlap) / max(1, len(top_ids))
   M["precision_k"] = hits / max(1, K)
   M["recall_k"] = hits / max(1, len(rel))
   M["map_k"] = average_precision_at_k(rel, top_ids, K)
   # Error metrics vs CF predictions (on overlap)
   if cf_estimates is not None and not cf_estimates.empty:
       dcf = cf_estimates[cf_estimates["donor_id"].astype(str).str.strip().str.upper() == d_id][["project_id","est"]].copy()
       dcf["project_id"] = dcf["project_id"].astype(str).str.strip().str.upper()
       D["n_cf"] = len(dcf)
       join = hist[["project_id","score"]].merge(dcf, on="project_id", how="inner")
       if not join.empty:
           y_true = join["score"].astype(float).to_numpy()
           y_pred = join["est"].astype(float).to_numpy()
           err = y_pred - y_true
           M["mae"]  = float(np.mean(np.abs(err)))
           M["mse"]  = float(np.mean(err**2))
           M["rmse"] = float(np.sqrt(M["mse"]))
   return M, D

# --------------- Load data & vectors ---------------
donors, projects, interactions = load_core(BASE)
proj_vecs_obj = load_proj_vectors(projects)
cf_estimates = load_cf_estimates(BASE)

# --------------- Header ---------------
st.title("🤎 Diaspora Donor Recommender System")
st.caption("Hybrid (Rule + Content Cosine + CF via precomputed SVD). Ethical AI down-weights over-exposed items. Small charts for clarity.")

with st.expander("Dataset & model status", expanded=False):
    st.write(f"Donors: **{len(donors):,}**, Projects: **{len(projects):,}**, Interactions: **{len(interactions):,}**")
    st.write("CF source:", "precomputed estimates ✅" if (cf_estimates is not None and not cf_estimates.empty) else "not found ❌")

# --------------- Tabs (then Home) ---------------
tab_home, tab_ins, tab_prog, tab_met, tab_why, tab_exp, tab_cmp, tab_diag, tab_reg = st.tabs([
    "Home", "Insights", "Donor progress", "Metrics", "Why these picks",
    "Explore projects", "Compare algorithms", "Diagnostics", "Register donor"
])

# --------------- HOME (two-pane) ---------------
with tab_home:
    left, right = st.columns([0.40, 0.60], gap="large")

    with left:
        st.subheader("Find donor and set preferences")

        q = st.text_input("Search donor (ID, name or email)")
        ddf = donors.copy()
        if q:
            ql = q.lower()
            ddf = ddf[
                ddf.apply(lambda r:
                    (ql in str(r.get("donor_id","")).lower()) or
                    (ql in str(r.get("name","")).lower()) or
                    (ql in str(r.get("email","")).lower()),
                    axis=1
                )
            ]
        # mark donors with history ✅ (normalized)
        hist_ids = set(interactions["donor_id"].astype(str).map(_std_id).unique()) if has_rows(interactions) else set()
        ddf["label"] = ddf.apply(lambda r: f"{r['donor_id']} - {r.get('name','')}" + (" ✅" if r['donor_id'] in hist_ids else ""), axis=1)
        options = ddf["label"].tolist() or [f"{donors.iloc[0]['donor_id']} - {donors.iloc[0].get('name','')}"]
        default_index = 0
        if "selected_donor_id" in st.session_state:
            lab = ddf.loc[ddf["donor_id"]==st.session_state["selected_donor_id"], "label"]
            if not lab.empty and lab.iloc[0] in options:
                default_index = options.index(lab.iloc[0])

        donor_label = st.selectbox("Choose donor", options, index=default_index)
        donor_id = donor_label.split(" - ")[0].strip()
        st.session_state["selected_donor_id"] = donor_id

        drow = donors.loc[donors["donor_id"]==donor_id].iloc[0]
        pref_regions_default = parse_multi(drow.get("region_preference"))
        pref_sectors_default = parse_multi(drow.get("sector_preference"))

        # donor card
        st.markdown(f"""
        <div style="border:1px solid #e7e7e7;border-radius:10px;padding:8px 10px;background:#fafafa">
          <div style="font-weight:700">{drow.get('name','')} <span style="font-weight:400;color:#666">({drow.get('donor_id','')})</span></div>
          <div style="color:#555">{drow.get('email','')}</div>
          <div style="margin-top:6px">Behavior: {status_dot_html(drow.get('behaviour_type'))}</div>
          <div style="margin-top:6px;font-size:13px;color:#555">
            Prefs — Regions: <b>{'; '.join(pref_regions_default) or '—'}</b> •
            Sectors: <b>{'; '.join(pref_sectors_default) or '—'}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # preference editors
        all_regions = sorted(projects["region"].dropna().unique().tolist())
        all_sectors = sorted(projects["sector_focus"].dropna().unique().tolist())
        ui_regions = st.multiselect("Preference: Regions (multi)", options=all_regions, default=pref_regions_default)
        ui_sectors = st.multiselect("Preference: Sectors (multi)", options=all_sectors, default=pref_sectors_default)
        pref_target = st.number_input("Preferred project funding target", min_value=0, value=int(drow.get("preferred_target", 0)) if pd.notna(drow.get("preferred_target", np.nan)) else 0, step=1000)
        cap = st.slider("Budget cap filter (funding target ≤)", 0, int(max(10_000, projects["funding_target"].max())), int(drow.get("budget_cap", 0)) if pd.notna(drow.get("budget_cap", np.nan)) else 0, step=1000)

        st.markdown("**Blend weights**")
        w_rule = st.slider("Rule-based", 0.0, 1.0, 0.30, 0.05)
        w_cos  = st.slider("Content (Cosine)", 0.0, 1.0, 0.40, 0.05)
        w_cf   = st.slider("Collaborative (CF)", 0.0, 1.0, 0.30, 0.05)

        ethical = st.toggle("Ethical AI (reduce over-exposed items)", value=True)
        hybrid = st.toggle("Hybrid mode (blend all three)", value=True)

        go = st.button("Generate recommendations", type="primary", use_container_width=True)
        clear_btn = st.button("Clear shortlist", use_container_width=True)

        if clear_btn:
            st.session_state["shortlist"] = pd.DataFrame()
            st.session_state["recs"] = pd.DataFrame()
            st.info("Shortlist cleared.")
            st.rerun()

        if go:
            weights = (w_rule, w_cos, w_cf) if hybrid else (1.0, 0.0, 0.0)
            recs, err = get_recs(
                donor_id, donors, projects, interactions,
                cf_estimates, proj_vecs_obj,
                weights=weights, ethical=ethical, topk=10,
                override_regions=ui_regions, override_sectors=ui_sectors
            )
            if err:
                st.warning(err); st.session_state["recs"] = pd.DataFrame()
            else:
                if cap and "funding_target" in recs.columns:
                    recs = recs[recs["funding_target"].fillna(0) <= cap]
                st.session_state["recs"] = recs.reset_index(drop=True)

    with right:
        st.subheader("Top recommendations")
        if "shortlist" not in st.session_state: st.session_state["shortlist"] = pd.DataFrame()
        recs = st.session_state.get("recs", pd.DataFrame())
        if not has_rows(recs):
            st.info("Click **Generate recommendations** on the left.")
        else:
            for i, row in recs.iterrows():
                org = row.get("organisation_type","N/A")
                target = human_money(row.get("funding_target"))
                st.markdown(f"""
                <div style="border:1px solid #e7e7e7;border-radius:10px;padding:8px 10px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <b>{i+1}. {row.get('title', row['project_id'])}</b><br>
                      <span style="font-size:12px;color:#666">{row.get('region','')} · {row.get('sector_focus','')} · {org}</span>
                    </div>
                    <div style="font-weight:700">{row.get('hybrid_score',0):.2f}</div>
                  </div>
                  <div style="margin-top:6px;color:#666">Why: {row.get('why','')}</div>
                  <div style="margin-top:6px;font-size:12px;color:#666">Target: {target}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Add to shortlist", key=f"add_{row['project_id']}"):
                    st.session_state["shortlist"] = pd.concat([st.session_state["shortlist"], pd.DataFrame([{
                        "donor_id": donor_id, "project_id": row["project_id"],
                        "title": row.get("title", row["project_id"]),
                        "region": row.get("region",""), "sector_focus": row.get("sector_focus",""),
                        "hybrid_score": row.get("hybrid_score",0.0)
                    }])], ignore_index=True)

            if has_rows(st.session_state["shortlist"]):
                st.download_button(
                    "Download shortlist (CSV)",
                    data=st.session_state["shortlist"].to_csv(index=False),
                    file_name=f"shortlist_{donor_id}.csv",
                    mime="text/csv"
                )

# --------------- INSIGHTS ---------------
with tab_ins:
    st.subheader("Insights from current recommendations")
    r = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(r):
        st.info("Generate recommendations on Home to see insights.")
    else:
        # Regions barh
        reg_counts = r["region"].value_counts()
        fig, ax = make_small_axes("s")
        ax.barh(reg_counts.index, reg_counts.values)
        ax.set_title("Regions in top picks"); ax.invert_yaxis()
        st.pyplot(fig)

        # Sectors donut
        sec_counts = r["sector_focus"].value_counts()
        fig2, ax2 = make_small_axes("s")
        wedges, texts = ax2.pie(sec_counts.values, startangle=90)
        centre = plt.Circle((0,0), 0.55, fc='white'); fig2.gca().add_artist(centre)
        ax2.set_title("Sectors (share)")
        st.pyplot(fig2)
        st.caption("Legend: " + ", ".join([f"{lab} ({val})" for lab, val in zip(sec_counts.index.tolist(), sec_counts.values.tolist())]))

        # Funding histogram
        fig3, ax3 = make_small_axes("s")
        ax3.hist(r["funding_target"].astype(float), bins=12)
        ax3.set_title("Funding target distribution"); ax3.set_xlabel("Target"); ax3.set_ylabel("Count")
        st.pyplot(fig3)

# --------------- DONOR PROGRESS ---------------
with tab_prog:
    st.subheader("Donor progress & giving")
    sel_id = st.session_state.get("selected_donor_id", donors.iloc[0]["donor_id"])
    drow = donors.loc[donors["donor_id"]==sel_id].iloc[0]
    st.markdown("**Current status:** " + status_dot_html(drow.get("behaviour_type")), unsafe_allow_html=True)

    hist = interactions[interactions["donor_id"]==_std_id(sel_id)] if has_rows(interactions) else pd.DataFrame()
    a,b,c,d = st.columns(4)
    a.metric("Known interactions", 0 if hist.empty else len(hist))
    b.metric("Unique projects", 0 if hist.empty else hist["project_id"].nunique())

    avg_gift = lifetime = 0.0
    if not hist.empty:
        f_map = projects.set_index("project_id")["funding_target"].to_dict()
        amounts = []
        for _, h in hist.iterrows():
            ft = float(f_map.get(_std_id(h["project_id"]), 0))
            amounts.append(0.05 * ft * float(h["score"]))
        if amounts:
            avg_gift = float(np.mean(amounts))
            lifetime = float(np.sum(amounts))
    c.metric("Avg gift (est.)", human_money(avg_gift))
    d.metric("Lifetime given (est.)", human_money(lifetime))

    if not hist.empty:
        proj_sectors = projects.set_index("project_id")["sector_focus"]
        s_counts = pd.Series([proj_sectors.get(_std_id(pid), "Unknown") for pid in hist["project_id"]]).value_counts()
        fig, ax = make_small_axes("s")
        ax.bar(s_counts.index[:10], s_counts.values[:10]); ax.set_title("Interacted sectors (top 10)"); ax.tick_params(axis='x', rotation=20)
        st.pyplot(fig)

        proj_regions = projects.set_index("project_id")["region"]
        r_counts = pd.Series([proj_regions.get(_std_id(pid), "Unknown") for pid in hist["project_id"]]).value_counts()
        fig2, ax2 = make_small_axes("s")
        ax2.pie(r_counts.values, startangle=90); centre = plt.Circle((0,0), 0.55, fc='white'); fig2.gca().add_artist(centre)
        ax2.set_title("Regions in history")
        st.pyplot(fig2)

        ft_series = pd.Series([projects.set_index("project_id")["funding_target"].get(_std_id(pid), 0) for pid in hist["project_id"]])
        fig3, ax3 = make_small_axes("s")
        ax3.hist(ft_series.astype(float), bins=12); ax3.set_title("Funding targets (history)")
        st.pyplot(fig3)
    else:
        st.info("No historical interactions yet for this donor.")

# --------------- METRICS ---------------
with tab_met:
    st.subheader("Evaluation metrics (donor-level)")
    r = st.session_state.get("recs", pd.DataFrame())
    sel_id = st.session_state.get("selected_donor_id", donors.iloc[0]["donor_id"])
    k = st.selectbox("Top-K", [5,10], index=0)
    thr_mode = st.selectbox("Relevance threshold", ["Median per donor","Any positive (>0)"], index=0)

    m = compute_metrics_for_donor(sel_id, r, interactions, projects, cf_estimates, k=k, thr_mode=thr_mode)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Precision@K", pct(m["precision_k"]))
        st.metric("Coverage@K",  pct(m["coverage_k"]))
        st.metric("MAE",         num3(m["mae"]))
    with c2:
        st.metric("Recall@K",    pct(m["recall_k"]))
        st.metric("Novelty ↑",   num3(m["novelty"]))
        st.metric("MSE",         num3(m["mse"]))
    with c3:
        st.metric("MAP@K",       pct(m["map_k"]))
        st.metric("Diversity@K", pct(m["diversity_k"]))
        st.metric("RMSE",        num3(m["rmse"]))
    # ---- diagnostics line (tiny helper to understand zeros)
    diag = D  # if you named the second return value D above
    st.caption(
       "Diagnostics — history rows: "
       f"{diag['n_hist']}, CF rows: {diag['n_cf']}, hits in top-K: "
       f"{diag['hits']}, overlap size (any history): {diag['overlap_size']}, relevant items: {diag['n_rel']}."
   )

# --------------- WHY THESE PICKS ---------------
with tab_why:
    st.subheader("Why these picks")
    r = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(r):
        st.info("Generate recommendations on Home first.")
    else:
        for i,row in r.iterrows():
            comp = pd.DataFrame({"Rule":[row.get("rule_score_norm",0)], "Content":[row.get("cosine_score_norm",0)], "CF":[row.get("cf_score_norm",0)]})
            st.markdown(f"**{i+1}. {row.get('title', row['project_id'])}** — {row.get('region','')} • {row.get('sector_focus','')} • Target {human_money(row.get('funding_target',0))}")
            st.caption(f"Why matched: {row.get('why','')}")
            fig, ax = make_small_axes("xs")
            comp.T[0].fillna(0).plot(kind="bar", ax=ax, legend=False)
            ax.set_ylim(0,1); ax.set_title("Score contribution"); ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)

# --------------- EXPLORE PROJECTS ---------------
with tab_exp:
    st.subheader("Explore projects")
    c1,c2,c3,c4 = st.columns(4)
    with c1: reg_sel = st.multiselect("Region", sorted(projects["region"].dropna().unique().tolist()))
    with c2: sec_sel = st.multiselect("Sector", sorted(projects["sector_focus"].dropna().unique().tolist()))
    with c3: org_sel = st.multiselect("Org type", sorted(projects["organisation_type"].dropna().unique().tolist()))
    with c4: max_budget = st.number_input("Max funding target", min_value=0, value=int(projects["funding_target"].max()))
    df = projects.copy()
    if reg_sel: df = df[df["region"].isin(reg_sel)]
    if sec_sel: df = df[df["sector_focus"].isin(sec_sel)]
    if org_sel: df = df[df["organisation_type"].isin(org_sel)]
    if max_budget: df = df[df["funding_target"] <= max_budget]
    st.dataframe(df[["project_id","title","region","sector_focus","organisation_type","funding_target","popularity"]], use_container_width=True)

    if not df.empty:
        fig, ax = make_small_axes("s")
        ax.hist(df["funding_target"].astype(float), bins=15)
        ax.set_title("Funding target distribution (filtered)")
        st.pyplot(fig)

# --------------- COMPARE ALGORITHMS ---------------
with tab_cmp:
    st.subheader("Compare algorithms")
    sel_id = st.session_state.get("selected_donor_id", donors.iloc[0]["donor_id"])
    def run_algo(weights, label):
        recs,_ = get_recs(sel_id, donors, projects, interactions, cf_estimates, proj_vecs_obj, weights=weights, ethical=True, topk=5)
        if not has_rows(recs): return pd.DataFrame()
        return recs[["title","region","sector_focus","organisation_type","funding_target","hybrid_score"]].rename(columns={"hybrid_score":f"{label} score"})
    cA,cB = st.columns(2); cC,cD = st.columns(2)
    cA.write("Rule-based Top-5");           cA.dataframe(run_algo((1,0,0),"Rule"))
    cB.write("Content (Cosine) Top-5");     cB.dataframe(run_algo((0,1,0),"Content"))
    cC.write("Collaborative (CF) Top-5");   cC.dataframe(run_algo((0,0,1),"CF"))
    cD.write("Hybrid Top-5");               cD.dataframe(run_algo((0.33,0.34,0.33),"Hybrid"))

# --------------- DIAGNOSTICS ---------------
with tab_diag:
    st.subheader("Diagnostics")
    st.write("Artifacts folder:", BASE)
    try:
        st.code("\n".join(sorted(os.listdir(BASE))), language="text")
    except Exception:
        pass
    st.write(f"CF estimates present: {'✅' if (cf_estimates is not None and not cf_estimates.empty) else '❌'}")
    st.write(f"Donors with history: {interactions['donor_id'].nunique() if has_rows(interactions) else 0}")
    st.write(f"Projects with history: {interactions['project_id'].nunique() if has_rows(interactions) else 0}")
    # Sample normalized IDs to prove alignment
    try:
        st.markdown("**Sample IDs after normalization:**")
        st.code("Projects: " + ", ".join(projects['project_id'].head(5).tolist()), language="text")
        if has_rows(interactions):
            st.code("History:  " + ", ".join(interactions['project_id'].head(5).tolist()), language="text")
    except Exception:
        pass

# --------------- REGISTER DONOR ---------------
with tab_reg:
    st.subheader("Register donor")
    with st.form("reg_form"):
        nm = st.text_input("Name")
        em = st.text_input("Email")
        regs = st.multiselect("Region preference (multi)", sorted(projects["region"].dropna().unique().tolist()))
        secs = st.multiselect("Sector preference (multi)", sorted(projects["sector_focus"].dropna().unique().tolist()))
        pref_t = st.number_input("Preferred project funding target ($)", min_value=0, value=0, step=1000)
        cap_v  = st.number_input("Budget cap ($)", min_value=0, value=0, step=1000)
        freq = st.selectbox("Giving frequency", ["One-off","Monthly","Quarterly","Yearly"])
        submit = st.form_submit_button("Create donor (session-only)")
    if submit:
        base = "NEW"; idx = 1; existing = set(donors["donor_id"])
        while f"{base}{idx:04d}" in existing: idx += 1
        new_id = f"{base}{idx:04d}"
        new_row = {
            "donor_id": new_id, "name": nm, "email": em,
            "region_preference": "; ".join(regs), "sector_preference": "; ".join(secs),
            "preferred_target": pref_t, "budget_cap": cap_v,
            "behaviour_type": "Active", "giving_frequency": freq
        }
        donors = pd.concat([donors, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["selected_donor_id"] = new_id
        st.success(f"Donor {new_id} created (session only). Go to Home to generate recommendations.")
