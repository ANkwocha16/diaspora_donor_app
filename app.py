# app.py — Diaspora Donor Recommender (Streamlit Cloud, loads from ./artifacts)

import os
import re
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ------------------------------ Page & cache control ------------------------------
st.set_page_config(page_title="Diaspora Donor Recommender", page_icon="🤝", layout="wide")

APP_VERSION = "2025-08-17-streamlit-cloud-artifacts"
if st.session_state.get("__app_version") != APP_VERSION:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["__app_version"] = APP_VERSION

# Root folder (repo-local)
BASE = "artifacts"
os.makedirs(BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------ Small helpers ------------------------------
FIG_XS = (2.2, 1.6)   # * small charts
FIG_S  = (2.8, 1.9)

def make_small_axes(size="xs"):
    fig, ax = plt.subplots(figsize=FIG_XS if size=="xs" else FIG_S)
    ax.tick_params(labelsize=8)
    return fig, ax

def parse_multi(val):
    if val is None: return []
    if isinstance(val, list): return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).replace("|", ";").replace(",", ";")
    return [p.strip() for p in s.split(";") if p.strip()]

def human_money(x):
    try:
        x = float(x)
        if x >= 1_000_000: return f"${x/1_000_000:.1f}M"
        if x >= 1_000:     return f"${x/1_000:.1f}k"
        return f"${int(x)}"
    except Exception:
        return str(x)

def normalize(df, col):
    scaler = MinMaxScaler()
    vals = df[[col]].astype(float)
    if vals.max().item() == vals.min().item():
        df[col+"_norm"] = 0.5
    else:
        df[col+"_norm"] = scaler.fit_transform(vals)
    return df

def status_dot_html(state):
    s = (state or "").lower().strip()
    color = "#9c3a3f"; label = "Passive"
    if s == "active" or s == "":
        color, label = "#10b981", "Active"
    if s == "selective":
        color, label = "#f59e0b", "Selective"
    return f'<span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>{label}</span>'

def has_rows(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def safe_df(df):
    return df if has_rows(df) else pd.DataFrame([{"Result":"N/A"}])

def pct(x) -> str:
    try: return f"{float(x):.1%}"
    except Exception: return "0.0%"

def num3(x) -> str:
    try: return f"{float(x):.3f}"
    except Exception: return "0.000"

def build_proj_vectors_on_the_fly(projects):
    regions = sorted(projects["region"].dropna().unique().tolist())
    sectors = sorted(projects["sector_focus"].dropna().unique().tolist())
    region_cols = [f"region_{r}" for r in regions]
    sector_cols = [f"sector_{s}" for s in sectors]
    rows = []
    for _, r in projects.iterrows():
        vec = {c:0 for c in region_cols + sector_cols}
        vec[f"region_{r['region']}"] = 1
        vec[f"sector_{r['sector_focus']}"] = 1
        vec["project_id"] = r["project_id"]
        rows.append(vec)
    pv = pd.DataFrame(rows)
    feature_cols = [c for c in pv.columns if c != "project_id"]
    return pv, feature_cols

def build_donor_vector_from_prefs(pref_regions, pref_sectors, feature_cols):
    vec = {c:0 for c in feature_cols}
    for r in pref_regions:
        k = f"region_{r}"
        if k in vec: vec[k] = 1
    for s in pref_sectors:
        k = f"sector_{s}"
        if k in vec: vec[k] = 1
    return np.array([vec[c] for c in feature_cols], dtype=float).reshape(1, -1)

def rule_score(donor_row, proj_row):
    s = 0.0
    r_prefs = parse_multi(donor_row.get("region_preference"))
    s_prefs = parse_multi(donor_row.get("sector_preference"))
    if proj_row["region"] in r_prefs: s += 0.5
    if proj_row["sector_focus"] in s_prefs: s += 0.5
    try:
        pref_target = float(donor_row.get("preferred_target", np.nan))
        if not np.isnan(pref_target):
            if abs(float(proj_row.get("funding_target",0)) - pref_target) < 0.1:
                s += 0.1
    except Exception: pass
    try:
        budget_cap = float(donor_row.get("budget_cap", np.nan))
        if not np.isnan(budget_cap) and proj_row.get("funding_target", np.nan) <= budget_cap:
            s += 0.15
    except Exception: pass
    s += (1.0/(1.0+proj_row.get("funding_target",1))) * 20000.0
    bt = str(donor_row.get("behaviour_type", "")).lower()
    if "selective" in bt: s *= 1.05
    elif "active" in bt:  s *= 1.02
    return s

# ------------------------------ Data loading ------------------------------
@st.cache_data(show_spinner=False)
def load_cf_estimates(base: str):
    for name in ["cf_estimates.csv.gz", "cf_estimates.csv"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                need = ["donor_id","project_id","est"]
                lower = [c.lower() for c in df.columns]
                if not all(k in lower for k in need):
                    raise ValueError("cf_estimates must have donor_id,project_id,est")
                ren = {}
                for want in need:
                    for c in df.columns:
                        if c.lower()==want: ren[c]=want
                df = df.rename(columns=ren)
                df["donor_id"] = df["donor_id"].astype(str)
                df["project_id"] = df["project_id"].astype(str)
                return df
            except Exception as e:
                st.warning(f"Could not read {p}: {e}")
    return None

@st.cache_data(show_spinner=False)
def load_core(base):
    # donors/projects file candidates
    donors_path   = os.path.join(base, "donors_5000.csv") if os.path.exists(os.path.join(base,"donors_5000.csv")) else os.path.join(base, "donors.csv")
    projects_path = os.path.join(base, "projects_2000.csv") if os.path.exists(os.path.join(base,"projects_2000.csv")) else os.path.join(base, "projects.csv")

    # interactions candidates (optional)
    inter_path = None
    for n in ["synthetic_interactions_5000x2000.csv","ratings_5000x2000.csv","ratings.csv","interactions.csv"]:
        p = os.path.join(base, n)
        if os.path.exists(p): inter_path = p; break

    donors = pd.read_csv(donors_path)
    projects = pd.read_csv(projects_path)

    # clean donors columns: lower-cased names
    donors.columns = [c.strip() for c in donors.columns]
    donors.rename(columns={c:c.lower() for c in donors.columns}, inplace=True)

    # normalize donor_id => "DNR0001" etc
    def norm_id_dnr(x:str) -> str:
        s = str(x).strip().upper()
        m = re.search(r"(\d+)$", s)
        if not m: return s
        return "DNR" + m.group(1).zfill(4)

    if "donor_id" in donors.columns:
        donors["donor_id"] = donors["donor_id"].apply(norm_id_dnr)
    else:
        # find donor id column heuristically
        cand = [c for c in donors.columns if "donor" in c or c=="id"]
        if cand:
            donors.rename(columns={cand[0]: "donor_id"}, inplace=True)
            donors["donor_id"] = donors["donor_id"].apply(norm_id_dnr)
        else:
            donors["donor_id"] = ["DNR"+str(i+1).zfill(4) for i in range(len(donors))]

    # normalize projects
    projects.columns = [c.strip().lower() for c in projects.columns]
    if "project_id" not in projects.columns:
        cand = [c for c in projects.columns if "project" in c or c=="id"]
        if cand: projects.rename(columns={cand[0]:"project_id"}, inplace=True)
        else: projects["project_id"] = [f"PR{i+1:04d}" for i in range(len(projects))]

    # minimal required fields with safe defaults
    for col, default in [
        ("name",""),
        ("email",""),
        ("behaviour_type","Active"),
        ("region_preference",""),
        ("sector_preference",""),
        ("preferred_target", np.nan),
        ("budget_cap", np.nan)
    ]:
        if col not in donors.columns: donors[col] = default

    for col, default in [
        ("title",""),
        ("region",""),
        ("sector_focus",""),
        ("organisation_type",""),
        ("funding_target", np.nan),
        ("popularity", 0.0)
    ]:
        if col not in projects.columns: projects[col] = default

    # cast types we rely on
    donors["preferred_target"] = pd.to_numeric(donors["preferred_target"], errors="coerce")
    donors["budget_cap"]       = pd.to_numeric(donors["budget_cap"], errors="coerce")
    projects["funding_target"] = pd.to_numeric(projects["funding_target"], errors="coerce")
    projects["popularity"]     = pd.to_numeric(projects["popularity"], errors="coerce").fillna(0.0)

    # interactions (optional)
    interactions = pd.DataFrame(columns=["donor_id","project_id","score"])
    if inter_path is not None:
        inter = pd.read_csv(inter_path)
        # heuristic column mapping
        lower = {c.lower():c for c in inter.columns}
        # try common names
        dcol = lower.get("donor_id") or lower.get("user_id") or lower.get("user") or list(inter.columns)[0]
        pcol = lower.get("project_id") or lower.get("item_id") or lower.get("item") or list(inter.columns)[1]
        scol = lower.get("score") or lower.get("rating") or (list(inter.columns)[2] if len(inter.columns)>2 else None)
        inter = inter.rename(columns={dcol:"donor_id", pcol:"project_id"})
        inter["donor_id"] = inter["donor_id"].astype(str)
        inter["project_id"] = inter["project_id"].astype(str)
        if scol is not None:
            inter = inter.rename(columns={scol:"score"})
            inter["score"] = pd.to_numeric(inter["score"], errors="coerce")
        else:
            inter["score"] = 1.0
        interactions = inter[["donor_id","project_id","score"]].dropna()

    return donors, projects, interactions

@st.cache_data(show_spinner=False)
def load_artifacts(base):
    # try to load pre-built vectors & features
    feats = None; proj_vecs = None; donor_vecs = None
    feats_path = os.path.join(base, "feature_cols.json")
    proj_path  = os.path.join(base, "proj_vectors.parquet")
    donor_path = os.path.join(base, "donor_vectors.parquet")
    try:
        if os.path.exists(feats_path): feats = json.load(open(feats_path))
        if os.path.exists(proj_path):  proj_vecs = pd.read_parquet(proj_path)
        if os.path.exists(donor_path): donor_vecs = pd.read_parquet(donor_path)
    except Exception:
        pass
    return proj_vecs, donor_vecs, feats

# ------------------------------ Core scoring ------------------------------
def get_recs(
    donor_id, donors, projects, interactions,
    cf_estimates,
    proj_vecs=None, donor_vecs=None, feats=None,
    weights=(0.33,0.33,0.34),
    ethical=False, topk=10,
    override_regions=None, override_sectors=None
):
    try:
        drow = donors.loc[donors["donor_id"]==donor_id].iloc[0].to_dict()
    except Exception:
        return pd.DataFrame(), "Unknown donor"

    # candidate = all projects; (You can apply ethical filtering here if desired)
    cand = projects.copy()

    # --- Rule-based
    rule = []
    for _, pr in cand.iterrows():
        rule.append(rule_score(drow, pr))
    cand["rule_score"] = rule
    cand = normalize(cand, "rule_score")

    # --- Content (cosine)
    if feats is None or proj_vecs is None:
        proj_vecs, feats = build_proj_vectors_on_the_fly(projects)
    pv = proj_vecs.set_index("project_id")[feats].fillna(0.0).astype(float).values
    pref_regions = override_regions if override_regions is not None else parse_multi(drow.get("region_preference"))
    pref_sectors = override_sectors if override_sectors is not None else parse_multi(drow.get("sector_preference"))
    dv = build_donor_vector_from_prefs(pref_regions, pref_sectors, feats)
    cand["cosine_score"] = cosine_similarity(dv, pv).ravel()
    cand = normalize(cand, "cosine_score")

    # --- CF (precomputed estimates preferred)
    if cf_estimates is not None:
        donor_str = str(donor_id)
        cf_slice = cf_estimates[cf_estimates["donor_id"]==donor_str][["project_id","est"]].copy()
        cf_slice = cf_slice.rename(columns={"est":"cf_score"})
        cand["project_id"] = cand["project_id"].astype(str)
        cand = cand.merge(cf_slice, on="project_id", how="left")
        # fallback: if missing CF, softly use cosine
        cand["cf_score"] = cand["cf_score"].fillna(cand["cosine_score"])
        cand = normalize(cand, "cf_score")
    else:
        # if no CF available, mirror cosine as neutral fallback
        cand["cf_score"] = cand["cosine_score"]
        cand["cf_score_norm"] = cand["cosine_score_norm"]

    # --- Hybrid blend
    w_rule, w_cos, w_cf = weights
    cand["hybrid_score"] = (
        w_rule * cand["rule_score_norm"] +
        w_cos  * cand["cosine_score_norm"] +
        w_cf   * cand["cf_score_norm"]
    )

    # --- Why text
    why = []
    for _, r in cand.iterrows():
        parts = []
        if r["region"] in pref_regions: parts.append("Region match")
        if r["sector_focus"] in pref_sectors: parts.append("Sector match")
        if r.get("cosine_score_norm",0) > 0.6: parts.append("High content similarity")
        if r.get("cf_score_norm",0)   > 0.6: parts.append("Similar donors liked this (CF)")
        why.append("; ".join(parts) if parts else "Rule/content blend")
    cand["why"] = why

    # pick final
    cand = cand.sort_values("hybrid_score", ascending=False).head(topk).reset_index(drop=True)
    return cand, None

# ------------------------------ Metrics ------------------------------
def average_precision_at_k(relevant, ranked_ids, k):
    if k == 0: return 0.0
    # AP@k simplified: precision at each relevant hit, averaged
    hits = 0; sum_prec = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in relevant:
            hits += 1
            sum_prec += hits / i
    if hits == 0: return 0.0
    return sum_prec / min(len(relevant), k)

def compute_metrics_for_donor(donor_id, recs, interactions, projects, cf_estimates, k=5):
    # Top-k list
    if not has_rows(recs): return dict.fromkeys(
        ["precision_k","recall_k","map_k","coverage_k","diversity_k","novelty","mae","mse","rmse"], 0.0)

    top_ids = recs["project_id"].astype(str).tolist()[:k]

    # Relevant (from interactions)
    rel = set()
    donor_inter = interactions[interactions["donor_id"].astype(str)==str(donor_id)]
    if has_rows(donor_inter):
        thr = donor_inter["score"].median()
        rel = set(donor_inter.loc[donor_inter["score"]>=thr,"project_id"].astype(str).unique())

    # Precision/Recall@K
    if k == 0:
        precision_k = recall_k = 0.0
    else:
        hits = sum(1 for pid in top_ids if pid in rel)
        precision_k = hits / k
        recall_k    = hits / (len(rel) if len(rel)>0 else 1.0)

    map_k = average_precision_at_k(rel, top_ids, k)

    # Coverage@K — fraction of recommended that donor has seen historically
    if len(top_ids) == 0:
        coverage_k = 0.0
    else:
        seen = set(donor_inter["project_id"].astype(str).unique()) if has_rows(donor_inter) else set()
        n_seen = sum(1 for pid in top_ids if pid in seen)
        coverage_k = n_seen / len(top_ids)

    # Diversity@K — unique sectors over k
    sec = projects.set_index("project_id").reindex(top_ids)["sector_focus"].fillna("").tolist()
    unique_sec = len(set(sec) - {""})
    diversity_k = unique_sec / (len(top_ids) if len(top_ids)>0 else 1)

    # Novelty — if popularity exists, 1 - mean(pop_norm)
    if "popularity" in projects.columns:
        pop = projects.set_index("project_id").reindex(top_ids)["popularity"].fillna(0.0).to_numpy()
        if pop.size>0:
            if pop.max()==pop.min():
                novelty = 0.80
            else:
                pop_n = (pop - pop.min())/(pop.max()-pop.min())
                novelty = float(1.0 - pop_n.mean())
        else:
            novelty = 0.80
    else:
        novelty = 0.80

    # Error metrics vs CF estimates (on overlap)
    mae = mse = rmse = 0.0
    if cf_estimates is not None and has_rows(donor_inter):
        dcf = cf_estimates[cf_estimates["donor_id"].astype(str)==str(donor_id)][["project_id","est"]].copy()
        dcf["project_id"] = dcf["project_id"].astype(str)
        di  = donor_inter[["project_id","score"]].copy()
        di["project_id"] = di["project_id"].astype(str)
        join = dcf.merge(di, on="project_id", how="inner")
        if has_rows(join):
            y_true = join["score"].to_numpy(dtype=float)
            y_pred = join["est"].to_numpy(dtype=float)
            err = y_pred - y_true
            mae = float(np.mean(np.abs(err)))
            mse = float(np.mean(err**2))
            rmse = float(np.sqrt(mse))

    return {
        "precision_k": precision_k,
        "recall_k":    recall_k,
        "map_k":       map_k,
        "coverage_k":  coverage_k,
        "diversity_k": diversity_k,
        "novelty":     novelty,
        "mae":         mae,
        "mse":         mse,
        "rmse":        rmse,
    }

# ------------------------------ UI ------------------------------
st.title("🤝 Diaspora Donor Recommender System")
st.caption("Hybrid (Rule + Cosine Similarity + CF via precomputed SVD) with multi-preference controls, donor progress, metrics, diagnostics, and exports.")

with st.expander("Dataset & model status", expanded=False):
    st.write("Loads from `./artifacts`. Preferred files:")
    st.code(
        "donors_5000.csv or donors.csv\n"
        "projects_2000.csv or projects.csv\n"
        "synthetic_interactions_5000x2000.csv / ratings_5000x2000.csv / ratings.csv / interactions.csv (optional)\n"
        "cf_estimates.csv.gz or cf_estimates.csv (optional, recommended)\n",
        language="text"
    )

# Load data
proj_vecs_art, donor_vecs_art, feature_cols_art = load_artifacts(BASE)
donors_base, projects_base, interactions_base = load_core(BASE)
cf_estimates = load_cf_estimates(BASE)

n_users = len(donors_base)
n_items = len(projects_base)
n_inter = len(interactions_base)

st.write(f"**Users:** **{n_users}**, **Items:** **{n_items}**, **Interactions:** **{n_inter}**")
st.write("CF source:", "precomputed estimates ✅" if cf_estimates is not None else "not found ❌")

# ---- Tabs (before Home content) ----
home, insights_tab, progress_tab, metrics_tab, why_tab, explore_tab, compare_tab, diagnostics_tab, register_tab = st.tabs(
    ["Home", "Insights", "Donor progress", "Metrics", "Why these picks", "Explore projects",
     "Compare algorithms", "Diagnostics", "+ Register donor"]
)

# -------------------------------- Home tab --------------------------------
with home:
    st.subheader("Find donor and set preferences")
    query = st.text_input("Search donor (ID, name or email)")
    ddf = donors_base.copy()
    if query:
        q = query.lower()
        ddf = ddf[ddf.apply(lambda r: q in str(r["donor_id"]).lower()
                            or q in str(r.get("name","")).lower()
                            or q in str(r.get("email","")).lower(), axis=1)]
    ddf["label"] = ddf.apply(lambda r: f"{r['donor_id']} — {r.get('name','')}", axis=1)
    options = ddf["label"].tolist() or donors_base.apply(lambda r: f"{r['donor_id']} — {r.get('name','')}", axis=1).tolist()
    selected = st.selectbox("Choose donor", options, index=0 if options else None)
    donor_id = (selected.split("—")[0].strip() if selected else donors_base.iloc[0]["donor_id"])

    # show donor card
    drow = donors_base.loc[donors_base["donor_id"]==donor_id].iloc[0]
    st.info(
        f"**{drow.get('name','')} [{donor_id}]**\n\n"
        f"{drow.get('email','')}\n\n"
        f"**Behavior:** {drow.get('behaviour_type','Active')}\n\n"
        f"**Prefs – Regions:** {drow.get('region_preference','')}; "
        f"**Sectors:** {drow.get('sector_preference','')}"
    )

    # preference UI
    region_opts = sorted(projects_base.get("region", pd.Series([], dtype=str)).dropna().unique().tolist())
    sector_opts = sorted(projects_base.get("sector_focus", pd.Series([], dtype=str)).dropna().unique().tolist())
    pref_regions = st.multiselect("Preference: Regions (multi)", region_opts, default=parse_multi(drow.get("region_preference")))
    pref_sectors = st.multiselect("Preference: Sectors (multi)", sector_opts, default=parse_multi(drow.get("sector_preference")))
    budget = st.number_input("Preferred project funding target", min_value=0, value=int(drow.get("preferred_target", 0)) if pd.notna(drow.get("preferred_target", np.nan)) else 0, step=1000)
    cap = st.slider("Budget cap (filters funding target $)", int(max(500_000, projects_base.get("funding_target", pd.Series([500_000])).max())), int(drow.get("budget_cap", 0)) if pd.notna(drow.get("budget_cap", np.nan)) else 0)

    # weights
    st.markdown("#### Blend weights")
    w_rule = st.slider("Rule-based", 0.0, 1.0, 0.30, 0.01)
    w_cos  = st.slider("Content (cosine)", 0.0, 1.0, 0.40, 0.01)
    w_cf   = st.slider("Collaborative (CF)", 0.0, 1.0, 0.30, 0.01)
    st.toggle("Ethical AI (reduce over-exposed items)", key="ethical")

    hybrid = st.toggle("Hybrid mode (blend all three)", value=True)
    weights = (w_rule, w_cos, w_cf) if hybrid else (1.0, 0.0, 0.0)

    go = st.button("Get recommendations", use_container_width=True)
    st.subheader("Top recommendations")

    if "shortlist" not in st.session_state: st.session_state["shortlist"] = pd.DataFrame()

    clear_btn = st.button("Clear shortlist")
    if clear_btn:
        st.session_state["shortlist"] = pd.DataFrame()
        st.session_state.pop("recs", None)
        st.info("Shortlist cleared.")

    if go:
        recs, err = get_recs(
            donor_id, donors_base, projects_base, interactions_base,
            cf_estimates,
            proj_vecs_art, donor_vecs_art, feature_cols_art,
            weights=weights, topk=10,
            ethical=st.session_state.get("ethical", False),
            override_regions=pref_regions, override_sectors=pref_sectors
        )
        st.session_state["recs"] = recs if err is None else pd.DataFrame()
        if err: st.warning(err)

    recs = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(recs):
        st.info("Click **Get recommendations**.")
    else:
        for i, row in recs.iterrows():
            org = row.get("organisation_type", "N/A")
            target = human_money(row.get("funding_target"))
            st.markdown(
                f"""
                <div style="border:1px solid #e7e7e7;border-radius:10px;padding:8px 10px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <b>{i+1}. {row['title']}</b><br>
                      <span style="font-size:12px;color:#666">{row['region']} · {row['sector_focus']} · {org}</span>
                    </div>
                    <div style="font-weight:700">{row['hybrid_score']:.2f}</div>
                  </div>
                  <div style="margin-top:6px;color:#666">Why: {row['why']}</div>
                  <div style="margin-top:6px;font-size:12px;color:#666">Target: {target}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Add to shortlist", key=f"add_{row['project_id']}"):
                st.session_state["shortlist"] = pd.concat([
                    st.session_state["shortlist"],
                    pd.DataFrame([{
                        "donor_id": donor_id,
                        "project_id": row["project_id"],
                        "title": row["title"],
                        "region": row["region"],
                        "sector_focus": row["sector_focus"],
                        "hybrid_score": row["hybrid_score"],
                    }])
                ], ignore_index=True)

        if has_rows(st.session_state["shortlist"]):
            st.write(" ")
            st.download_button(
                "Save shortlist (CSV)",
                data=st.session_state["shortlist"].to_csv(index=False),
                file_name=f"shortlist_{donor_id}.csv",
                mime="text/csv"
            )

# -------------------------------- Insights tab --------------------------------
with insights_tab:
    st.subheader("Insights")
    st.write("Simple overview of your dataset.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Donors", n_users)
    c2.metric("Projects", n_items)
    c3.metric("Interactions", n_inter)

    # small chart: distribution of funding target
    df = projects_base.copy()
    if not df.empty and "funding_target" in df.columns:
        fig, ax = make_small_axes("s")
        ax.hist(df["funding_target"].astype(float), bins=20)
        ax.set_title("Funding target distribution", fontsize=10)
        ax.set_xlabel("Funding target", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        st.pyplot(fig)

# -------------------------------- Donor progress tab --------------------------------
with progress_tab:
    st.subheader("Donor progress")
    st.write("(Placeholder) — Track donor engagement and progression here.")
    st.dataframe(donors_base[["donor_id","name","behaviour_type","region_preference","sector_preference"]].head(10))

# -------------------------------- Metrics tab --------------------------------
with metrics_tab:
    st.subheader("Evaluation metrics (donor-level)")
    k = st.number_input("Top-K for evaluation", min_value=1, max_value=20, value=5, step=1)
    recs = st.session_state.get("recs", pd.DataFrame())
    metrics = compute_metrics_for_donor(donor_id, recs, interactions_base, projects_base, cf_estimates, k=k)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Precision@K", pct(metrics["precision_k"]))
        st.metric("Coverage@K",  pct(metrics["coverage_k"]))
        st.metric("MAE",         num3(metrics["mae"]))
    with c2:
        st.metric("Recall@K",    pct(metrics["recall_k"]))
        st.metric("Novelty ↑",   num3(metrics["novelty"]))
        st.metric("MSE",         num3(metrics["mse"]))
    with c3:
        st.metric("MAP@K",       pct(metrics["map_k"]))
        st.metric("Diversity@K", pct(metrics["diversity_k"]))
        st.metric("RMSE",        num3(metrics["rmse"]))

    # small component bars
    if has_rows(recs):
        comp = pd.DataFrame({
            "Component": ["CF", "Content", "Rule"],
            "Score": [
                recs["cf_score_norm"].mean(),
                recs["cosine_score_norm"].mean(),
                recs["rule_score_norm"].mean()
            ]
        })
        fig, ax = make_small_axes("s")
        ax.bar(comp["Component"], comp["Score"])
        ax.set_ylim(0,1)
        ax.set_title("Avg score components", fontsize=12)
        st.pyplot(fig)

# -------------------------------- Why these picks tab --------------------------------
with why_tab:
    st.subheader("Why these picks")
    recs = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(recs):
        st.info("Generate recommendations on **Home** first.")
    else:
        st.dataframe(recs[["project_id","title","region","sector_focus","why","hybrid_score"]])

# -------------------------------- Explore projects tab --------------------------------
with explore_tab:
    st.subheader("Explore projects")
    st.dataframe(projects_base[["project_id","title","region","sector_focus","organisation_type","funding_target","popularity"]].head(50))

# -------------------------------- Compare algorithms tab --------------------------------
with compare_tab:
    st.subheader("Compare algorithms")
    def run_algo(weights, label):
        recs,_ = get_recs(
            donor_id, donors_base, projects_base, interactions_base,
            cf_estimates,
            proj_vecs_art, donor_vecs_art, feature_cols_art,
            weights=weights, topk=5
        )
        out = recs[["title","region","sector_focus","organisation_type","funding_target","hybrid_score"]].rename(
            columns={"hybrid_score":f"{label} score"})
        return out

    colA, colB = st.columns(2); colC, colD = st.columns(2)
    colA.write("Rule-based Top-5")
    colA.dataframe(safe_df(run_algo((1,0,0),"Rule")))
    colB.write("Content Cosine Top-5")
    colB.dataframe(safe_df(run_algo((0,1,0),"Content")))
    colC.write("Collaborative (CF) Top-5")
    colC.dataframe(safe_df(run_algo((0,0,1),"CF")))
    colD.write("Hybrid Top-5")
    colD.dataframe(safe_df(run_algo((0.33,0.33,0.34),"Hybrid")))

# -------------------------------- Diagnostics tab --------------------------------
with diagnostics_tab:
    st.subheader("Diagnostics")
    st.write("Artifacts present:")
    st.code("\n".join(sorted(os.listdir(BASE))), language="text")
    st.write("Sample donors/projects")
    st.dataframe(donors_base.head(5))
    st.dataframe(projects_base.head(5))
    if has_rows(interactions_base):
        st.write("Sample interactions")
        st.dataframe(interactions_base.head(5))
    st.write("CF estimates present:", "Yes" if cf_estimates is not None else "No")

# -------------------------------- Register donor tab --------------------------------
with register_tab:
    st.subheader("Register donor")
    with st.form("reg_form", clear_on_submit=True):
        name  = st.text_input("Name")
        email = st.text_input("Email")
        beh   = st.selectbox("Behaviour type", ["Active","Passive","Selective"], index=0)
        rp    = st.text_input("Region preference (semicolon-separated)")
        sp    = st.text_input("Sector preference (semicolon-separated)")
        pt    = st.number_input("Preferred target ($)", min_value=0, value=0, step=1000)
        bc    = st.number_input("Budget cap ($)", min_value=0, value=0, step=1000)
        submit = st.form_submit_button("Register donor")
    if submit:
        new_id = f"DNR{n_users+1:04d}"
        st.success(f"Registered {name} as {new_id} (not persisted in artifacts).")
