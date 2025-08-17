# app.py — Diaspora Donor Recommender (Streamlit Cloud, loads from ./artifacts)
import os, json, time, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib

# ----------------------------- Page & cache control -----------------------------
st.set_page_config(page_title="Diaspora Donor Recommender", page_icon="🤝", layout="wide")

APP_VERSION = "2025-08-17-streamlit-cloud-artifacts-v2"
if st.session_state.get("_app_version") != APP_VERSION:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["_app_version"] = APP_VERSION

# Inline CF training toggle (off on Streamlit Cloud)
# On local dev: export ALLOW_INLINE_TRAIN=1  (or set in Advanced Settings -> Secrets)
ALLOW_INLINE_TRAIN = os.environ.get("ALLOW_INLINE_TRAIN", "0") == "1"

# Root folder (repo-local)
BASE = "artifacts"
os.makedirs(BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- Small helpers -----------------------------
FIG_XS = (2.0, 1.3)
FIG_S  = (2.4, 1.5)
PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
           "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ab"]

def take_colors(n):
    if n <= 0: return []
    reps = (n // len(PALETTE)) + 1
    return (PALETTE * reps)[:n]

def parse_multi(val):
    if val is None: return []
    if isinstance(val, list): return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).replace('|',';').replace(',',';')
    return [p.strip() for p in s.split(';') if p.strip()]

def human_money(x):
    try:
        x = float(x)
        if x >= 1_000_000: return f"${x/1_000_000:.1f}M"
        if x >= 1_000: return f"${x/1_000:.1f}k"
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
    color = "#9ca3af"; label = "Passive"
    if s == "active" or s == "":
        color, label = "#10b981", "Active"
    if s == "selective":
        color, label = "#f59e0b", "Selective"
    return f'<span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>{label}</span>'

def has_rows(df):
    return isinstance(df, pd.DataFrame) and not df.empty

def safe_df(df):
    return df if has_rows(df) else pd.DataFrame([{"Result":"N/A"}])

def ensure_cols(df, cols, name="df"):
    """
    Ensure that required columns are present in the dataframe.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required cols: {missing}. Found: {list(df.columns)}")

def load_csv_or_parquet(path):
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)

# ----------------------------- Data loading -----------------------------
@st.cache_data(show_spinner=True)
def load_core(base):
    donors_path = os.path.join(base, "donors_5000.csv") if os.path.exists(os.path.join(base,"donors_5000.csv")) else os.path.join(base,"donors.csv")
    projects_path = os.path.join(base, "projects_2000.csv") if os.path.exists(os.path.join(base,"projects_2000.csv")) else os.path.join(base,"projects.csv")
    inter_path = None
    for n in ["synthetic_interactions_5000x2000.csv","ratings_5000x2000.csv","ratings.csv","interactions.csv"]:
        p = os.path.join(base, n)
        if os.path.exists(p): inter_path = p; break

    donors = load_csv_or_parquet(donors_path)
    projects = load_csv_or_parquet(projects_path)
    interactions = load_csv_or_parquet(inter_path) if inter_path else pd.DataFrame(columns=["Donor_ID","Project_ID","Score"])

    # tidy cols
    donors.columns = [c.strip() for c in donors.columns]
    projects.columns = [c.strip() for c in projects.columns]
    projects = projects.rename(columns={c: c.strip().lower() for c in projects.columns})

    # interactions robust rename
    if not interactions.empty:
        interactions.columns = [c.strip() for c in interactions.columns]
        low = {c.lower():c for c in interactions.columns}
        # unify to Donor_ID / Project_ID / Score
        dcol = low.get("donor_id", low.get("donorid", low.get("user_id", None)))
        icol = low.get("project_id", low.get("item_id", None))
        scol = low.get("score", low.get("rating", None))
        if not all([dcol, icol, scol]):
            raise ValueError(f"interactions needs donor_id/project_id/score columns. Found: {list(interactions.columns)}")
        interactions = interactions.rename(columns={dcol:"Donor_ID", icol:"Project_ID", scol:"Score"})[["Donor_ID","Project_ID","Score"]]
        interactions["Score"] = pd.to_numeric(interactions["Score"], errors="coerce").fillna(0)

        def norm_id_dnr(x: str) -> str:
            s = str(x).strip().upper()
            m = re.search(r"(\d+)$", s)
            if not m: return s
            digits = m.group(1)
            return "DNR" + digits.zfill(4)
        interactions["Donor_ID"] = interactions["Donor_ID"].apply(norm_id_dnr)
        interactions["Project_ID"] = interactions["Project_ID"].astype(str).str.strip().str.upper()
    else:
        interactions = pd.DataFrame(columns=["Donor_ID","Project_ID","Score"])

    # donors
    ensure_cols(donors, ["donor_id","name","email","region_preference","sector_preference"], "donors")
    def norm_id_donor(x: str) -> str:
        s = str(x).strip().upper()
        m = re.search(r"(\d+)$", s)
        if not m: return s
        digits = m.group(1)
        return "DNR" + digits.zfill(4)
    donors["donor_id"] = donors["donor_id"].apply(norm_id_donor)

    # projects
    ensure_cols(projects, ["project_id","title","region","sector_focus","funding_target","organisation_type","popularity"], "projects")
    projects["funding_target"] = pd.to_numeric(projects["funding_target"], errors="coerce").fillna(projects["funding_target"].median())
    projects["popularity"]     = pd.to_numeric(projects["popularity"], errors="coerce").fillna(0)
    projects["project_id"]     = projects["project_id"].astype(str).str.upper().str.strip()

    return donors, projects, interactions

@st.cache_resource(show_spinner=False)
def load_artifacts(base):
    paths = {
        "svd": os.path.join(base, "svd_model.joblib"),
        "proj": os.path.join(base, "proj_vectors.parquet"),
        "donor": os.path.join(base, "donor_vectors.parquet"),
        "feats": os.path.join(base, "feature_cols.json"),
    }
    svd = proj_vecs = donor_vecs = feats = None
    if os.path.exists(paths["svd"]):
        try: svd = joblib.load(paths["svd"])
        except Exception: svd = None
    if os.path.exists(paths["proj"]):
        proj_vecs = load_csv_or_parquet(paths["proj"])
    if os.path.exists(paths["donor"]):
        donor_vecs = load_csv_or_parquet(paths["donor"])
    if os.path.exists(paths["feats"]):
        feats = json.load(open(paths["feats"], "r"))
    return svd, proj_vecs, donor_vecs, feats

# Optional inline SVD (disabled by default on Cloud)
@st.cache_resource(show_spinner=True)
def train_svd_inline(interactions):
    if not ALLOW_INLINE_TRAIN:
        return None
    try:
        from surprise import Dataset, Reader, SVD
    except Exception:
        return None
    if interactions is None or interactions.empty:
        return None
    reader = Reader(rating_scale=(float(interactions["Score"].min()), float(interactions["Score"].max())))
    data = Dataset.load_from_df(interactions[["Donor_ID","Project_ID","Score"]], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=60, n_epochs=25, biased=True, random_state=42)
    model.fit(trainset)
    return model

# ----------------------------- Vector builders & rule scoring -----------------------------
def build_proj_vectors_on_the_fly(projects):
    regions = sorted(projects["region"].dropna().unique().tolist())
    sectors = sorted(projects["sector_focus"].dropna().unique().tolist())
    region_cols = [f"region__{r}" for r in regions]
    sector_cols = [f"sector__{s}" for s in sectors]
    rows = []
    for _, r in projects.iterrows():
        vec = {c:0 for c in region_cols + sector_cols}
        vec[f"region__{r['region']}"] = 1
        vec[f"sector__{r['sector_focus']}"] = 1
        vec["project_id"] = r["project_id"]
        rows.append(vec)
    pv = pd.DataFrame(rows)
    feature_cols = [c for c in pv.columns if c != "project_id"]
    return pv, feature_cols

def build_donor_vector_from_prefs(pref_regions, pref_sectors, feature_cols):
    vec = {c:0 for c in feature_cols}
    for r in pref_regions:
        k = f"region__{r}"
        if k in vec: vec[k] = 1
    for s in pref_sectors:
        k = f"sector__{s}"
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
            if abs(float(proj_row.get("funding_target",0)) - pref_target) <= 0.2*pref_target:
                s += 0.1
    except Exception: pass
    try:
        budget_cap = float(donor_row.get("budget_cap", np.nan))
        if not np.isnan(budget_cap) and proj_row.get("funding_target", 1) <= budget_cap:
            s += 0.15
    except Exception: pass
    s += (1.0/(1.0+proj_row.get("funding_target",1))) * 20000.0
    s += float(proj_row.get("popularity",0)) * 0.02
    bt = str(donor_row.get("behaviour_type","")).lower()
    if "selective" in bt: s *= 1.05
    elif "active" in bt: s *= 1.02
    return s

# ----------------------------- Recommendations -----------------------------
def get_recs(
    donor_id, donors, projects, interactions, svd, proj_vecs, donor_vecs, feats,
    weights, filters, ethical=True, topk=10, override_regions=None, override_sectors=None
):
    drow = donors[donors["donor_id"] == donor_id]
    if drow.empty: return pd.DataFrame(), "Unknown donor_id"
    drow = drow.iloc[0].copy()

    cand = projects.copy()
    if ethical and "popularity" in cand.columns and len(cand) > 0:
        p90 = cand["popularity"].quantile(0.9)
        cand = cand[cand["popularity"] <= p90].copy()
    if filters.get("region"): cand = cand[cand["region"].isin(filters["region"])].copy()
    if filters.get("sector"): cand = cand[cand["sector_focus"].isin(filters["sector"])].copy()
    if filters.get("budget") is not None: cand = cand[cand["funding_target"] <= filters["budget"]].copy()
    if cand.empty: return pd.DataFrame(), "No projects left after filtering."

    # Rule
    cand["rule_score"] = [rule_score(drow, prow) for _, prow in cand.iterrows()]
    cand = normalize(cand, "rule_score")

    # Content (cosine)
    if feats is None or proj_vecs is None:
        proj_vecs, feats = build_proj_vectors_on_the_fly(projects)
    pv = proj_vecs.set_index("project_id").loc[cand["project_id"], feats].fillna(0.0).astype(float).values
    pref_regions = override_regions if override_regions is not None else parse_multi(drow.get("region_preference"))
    pref_sectors = override_sectors if override_sectors is not None else parse_multi(drow.get("sector_preference"))
    dv = build_donor_vector_from_prefs(pref_regions, pref_sectors, feats)
    cand["cosine_score"] = cosine_similarity(pv, dv).ravel()
    cand = normalize(cand, "cosine_score")

    # Collaborative (SVD prediction) — fallback to content if missing
    cf = []
    if svd is not None:
        for pid in cand["project_id"]:
            try: cf.append(svd.predict(donor_id, pid).est)
            except Exception: cf.append(np.nan)
    else:
        cf = [np.nan]*len(cand)
    cand["cf_score"] = pd.Series(cf).fillna(cand["cosine_score"])
    cand = normalize(cand, "cf_score")
    if cand["cf_score_norm"].nunique() <= 1:
        cand["cf_score_norm"] = (0.7*cand["cosine_score_norm"] + 0.3*cand["rule_score_norm"])

    # Hybrid
    w_rule, w_cos, w_cf = weights
    cand["hybrid_score"] = w_rule*cand["rule_score_norm"] + w_cos*cand["cosine_score_norm"] + w_cf*cand["cf_score_norm"]

    # Why
    why = []
    for _, r in cand.iterrows():
        parts = []
        if r["region"] in parse_multi(drow.get("region_preference")): parts.append("Region match")
        if r["sector_focus"] in parse_multi(drow.get("sector_preference")): parts.append("Sector match")
        if r["cosine_score_norm"] > 0.6: parts.append("High content similarity")
        if r["cf_score_norm"] > 0.6: parts.append("Similar donors liked this")
        if not parts: parts = ["Strong blended score"]
        why.append("; ".join(parts))
    cand["why"] = why

    cols = ["project_id","title","region","sector_focus","funding_target","organisation_type","popularity",
            "rule_score_norm","cosine_score_norm","cf_score_norm","hybrid_score","why"]
    recs = cand[cols].fillna(0).sort_values("hybrid_score", ascending=False).head(topk).reset_index(drop=True)
    return recs, None

# ----------------------------- Load data & artifacts -----------------------------
donors_base, projects, interactions = load_core(BASE)
svd_art, proj_vecs_art, donor_vecs_art, feats_art = load_artifacts(BASE)

# Keep a working donors copy in session
if "donors" not in st.session_state:
    st.session_state["donors"] = donors_base.copy()
donors = st.session_state["donors"]

# ----------------------------- UI: Header -----------------------------
st.title("🤝 Diaspora Donor Recommender System")
st.caption("Hybrid (Rule + Cosine Similarity + CF via SVD) with multi-preference controls, donor progress, metrics, diagnostics, and exports.")

with st.expander("Dataset and model status", expanded=False):
    n_users = donors["donor_id"].nunique()
    n_items = projects["project_id"].nunique()
    n_inter = 0 if interactions is None or interactions.empty else len(interactions)
    sparsity = 1.0 - (n_inter / max(1, (n_users * n_items)))
    st.write(f"Users: **{n_users}**, Items: **{n_items}**, Interactions: **{n_inter}**, Sparsity: **{sparsity:.6f}**")
    st.write("CF model:", "pre-trained loaded ✅" if svd_art is not None else "not found ❌")
    if not ALLOW_INLINE_TRAIN:
        st.caption("Inline CF training is disabled on this deployment.")

# ----------------------------- Tabs FIRST -----------------------------
tab_home, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Home", "📊 Insights", "🧭 Donor progress", "📈 Metrics", "💡 Why these picks",
    "🔎 Explore projects", "🧠 Compare algorithms", "🧪 Diagnostics"
])

# ----------------------------- HOME (Find donor + recs) -----------------------------
with tab_home:
    left, right = st.columns([0.42, 0.58])

    with left:
        st.subheader("Find donor and set preferences")

        query = st.text_input("Search donor (ID, name or email)")
        ddf = donors.copy()
        if query:
            q = query.lower()
            ddf = ddf[ddf.apply(lambda r: q in str(r["donor_id"]).lower() or q in str(r.get("name","")).lower() or q in str(r.get("email","")).lower(), axis=1)]

        hist_ids = set(interactions["Donor_ID"].unique()) if has_rows(interactions) else set()
        ddf["label"] = ddf.apply(lambda r: f"{r['donor_id']} - {r.get('name','')}" + (" ✅" if r["donor_id"] in hist_ids else ""), axis=1)
        options = ddf["label"].tolist() or donors.apply(lambda r: f"{r['donor_id']} - {r.get('name','')}", axis=1).tolist()

        default_id = st.session_state.get("selected_donor_id")
        def_label = None
        if default_id is not None:
            row = ddf[ddf["donor_id"] == default_id]
            if not row.empty: def_label = row["label"].iloc[0]
        default_index = options.index(def_label) if def_label in options else 0

        donor_label = st.selectbox("Choose donor", options, index=default_index)
        donor_id = donor_label.split(" - ")[0].strip()
        drow = donors[donors["donor_id"] == donor_id].iloc[0]

        # donor card
        address = None
        for c in ["address","location","city","country"]:
            if c in donors.columns and pd.notna(drow.get(c)):
                address = f"{c.title()}: {drow.get(c)}"; break
        pref_regions_text = "; ".join(parse_multi(drow.get("region_preference")))
        pref_sectors_text = "; ".join(parse_multi(drow.get("sector_preference")))
        budget_cap = drow.get("budget_cap","N/A")
        freq = drow.get("giving_frequency","N/A")
        dot = status_dot_html(drow.get("behaviour_type"))
        st.markdown(f"""
        <div style="border:1px solid #e7e7e7;border-radius:10px;padding:8px 10px;background:#fafafa">
          <div style="font-weight:700">{drow.get('name','')} <span style="font-weight:400;color:#666">({drow.get('donor_id','')})</span></div>
          <div style="color:#555">{drow.get('email','')}</div>
          <div style="color:#555">{address or ''}</div>
          <div style="margin-top:6px">Behavior: {dot}</div>
          <div style="margin-top:6px;font-size:13px;color:#555">
            Prefs — Regions: <b>{pref_regions_text or '—'}</b> • Sectors: <b>{pref_sectors_text or '—'}</b><br>
            Budget cap: <b>{human_money(budget_cap)}</b> • Frequency: <b>{freq}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # preference editors
        all_regions = sorted(projects["region"].dropna().unique().tolist())
        all_sectors = sorted(projects["sector_focus"].dropna().unique().tolist())
        default_regions = [r for r in parse_multi(drow.get("region_preference")) if r in all_regions]
        default_sectors = [s for s in parse_multi(drow.get("sector_preference")) if s in all_sectors]
        ui_regions = st.multiselect("Preference: Regions (multi)", options=all_regions, default=default_regions)
        ui_sectors = st.multiselect("Preference: Sectors (multi)", options=all_sectors, default=default_sectors)

        if ui_regions != default_regions or ui_sectors != default_sectors:
            donors.loc[donors["donor_id"]==donor_id, "region_preference"] = "; ".join(ui_regions)
            donors.loc[donors["donor_id"]==donor_id, "sector_preference"] = "; ".join(ui_sectors)
            drow = donors[donors["donor_id"] == donor_id].iloc[0]
            st.session_state["donors"] = donors

        pref_target = st.number_input(
            "Preferred project funding target", min_value=0,
            value=int(drow.get("preferred_target", 0)) if pd.notna(drow.get("preferred_target", np.nan)) else 0, step=1000
        )
        donors.loc[donors["donor_id"]==donor_id, "preferred_target"] = pref_target
        st.session_state["donors"] = donors

        budget = st.slider("Budget cap (filters funding target ≤)", 0, int(projects["funding_target"].max()),
                           int(projects["funding_target"].quantile(0.5)), 1000)

        st.markdown("**Hybrid weights**")
        w_rule = st.slider("Rule-based", 0.0, 1.0, 0.30, 0.05)
        w_cos  = st.slider("Content (Cosine)", 0.0, 1.0, 0.40, 0.05)
        w_cf   = st.slider("Collaborative (SVD)", 0.0, 1.0, 0.30, 0.05)
        ethical = st.toggle("Ethical AI (reduce over-exposed items)", value=True)
        hybrid_mode = st.toggle("Hybrid mode (blend all three)", value=True)

        # optional inline training (disabled by default on Cloud)
        if ALLOW_INLINE_TRAIN:
            cf_mode = st.radio("Collaborative filtering source",
                               ["Use pre-trained (fast)", "Train inline from interactions"], index=0)
            if cf_mode == "Train inline from interactions":
                if "svd_inline" not in st.session_state:
                    with st.spinner("Training SVD from interactions…"):
                        st.session_state["svd_inline"] = train_svd_inline(interactions)
        else:
            st.caption("CF training disabled on this deployment (using pre-trained model).")

        # persist UI for right column
        st.session_state["ui_regions"] = ui_regions
        st.session_state["ui_sectors"] = ui_sectors
        st.session_state["budget"]     = budget
        st.session_state["w_rule"]      = w_rule
        st.session_state["w_cos"]       = w_cos
        st.session_state["w_cf"]        = w_cf
        st.session_state["ethical"]     = ethical
        st.session_state["hybrid_mode"] = hybrid_mode
        st.session_state["selected_donor_id"] = donor_id

        go = st.button("Get recommendations", type="primary")
        clear_btn = st.button("Clear shortlist")

    with right:
        st.subheader("Top recommendations")
        if "shortlist" not in st.session_state:
            st.session_state["shortlist"] = pd.DataFrame()
        if clear_btn:
            st.session_state["shortlist"] = pd.DataFrame()
            st.info("Shortlist cleared.")

        # read UI from session with fallbacks
        donor_id = st.session_state.get("selected_donor_id", donors["donor_id"].iloc[0])
        drow = donors[donors["donor_id"] == donor_id].iloc[0]
        ui_regions = st.session_state.get("ui_regions", parse_multi(drow.get("region_preference")))
        ui_sectors = st.session_state.get("ui_sectors", parse_multi(drow.get("sector_preference")))
        budget     = st.session_state.get("budget", int(projects["funding_target"].quantile(0.5)))
        w_rule     = st.session_state.get("w_rule", 0.30)
        w_cos      = st.session_state.get("w_cos", 0.40)
        w_cf       = st.session_state.get("w_cf", 0.30)
        ethical    = st.session_state.get("ethical", True)
        hybrid_mode = st.session_state.get("hybrid_mode", True)

        if go:
            weights = (w_rule,w_cos,w_cf) if hybrid_mode else (1.0,0.0,0.0)
            svd_inline = st.session_state.get("svd_inline") if ALLOW_INLINE_TRAIN else None
            svd_to_use = svd_inline if svd_inline is not None else svd_art
            recs, err = get_recs(
                donor_id, donors, projects, interactions, svd_to_use,
                None, None, None,
                weights, {"region": ui_regions, "sector": ui_sectors, "budget": budget},
                ethical=ethical, topk=10, override_regions=ui_regions, override_sectors=ui_sectors
            )
            st.session_state["recs"] = recs if err is None else pd.DataFrame()
            if err: st.warning(err)

        recs = st.session_state.get("recs", pd.DataFrame())
        if not has_rows(recs):
            st.info("Click **Get recommendations**.")
        else:
            for i,row in recs.iterrows():
                org = row.get("organisation_type","N/A")
                target = human_money(row.get("funding_target"))
                st.markdown(f"""
                <div style="border:1px solid #e7e7e7;border-radius:10px;padding:8px 10px;">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <b>{i+1}. {row['title']}</b><br>
                      <span style="font-size:12px;color:#666">{row['region']} • {row['sector_focus']} • {org} • Target {target}</span>
                    </div>
                    <div style="font-weight:700">{row['hybrid_score']:.2f}</div>
                  </div>
                  <div style="margin-top:6px;color:#666">Why: {row['why']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Add to shortlist", key=f"add_{row['project_id']}"):
                    st.session_state["shortlist"] = pd.concat([st.session_state["shortlist"], pd.DataFrame([row])], ignore_index=True)

            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1:
                if st.button("Save shortlist (CSV)"):
                    p = os.path.join(OUTPUT_DIR, f"shortlist_{donor_id}_{int(time.time())}.csv")
                    st.session_state["shortlist"].to_csv(p, index=False); st.success(f"Saved: {p}")
            with c2:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
                def export_pdf(recs_df, donor_row, path):
                    c = canvas.Canvas(path, pagesize=A4); w,h = A4; y = h-40
                    c.setFont("Helvetica-Bold", 14); c.drawString(40,y,"Diaspora Donor Recommender — Top Picks"); y-=22
                    c.setFont("Helvetica", 10); c.drawString(40,y,f"Donor: {donor_row.get('name','')} ({donor_row.get('donor_id','')})"); y-=16
                    c.drawString(40,y,f"Prefs: {donor_row.get('region_preference','')} / {donor_row.get('sector_preference','')}"); y-=24
                    for i,row in recs_df.iterrows():
                        if y<80: c.showPage(); y=h-40
                        c.setFont("Helvetica-Bold",11); c.drawString(40,y,f"{i+1}. {row['title']}"); y-=14
                        c.setFont("Helvetica",10); c.drawString(40,y,f"{row['region']} • {row['sector_focus']} • {row['organisation_type']} | Target {int(row['funding_target'])} | Score {row['hybrid_score']:.2f}"); y-=18
                    c.save()
                if st.button("Save shortlist (PDF)"):
                    p = os.path.join(OUTPUT_DIR, f"shortlist_{donor_id}_{int(time.time())}.pdf")
                    export_pdf(st.session_state["shortlist"], drow, p); st.success(f"Saved: {p}")
            with c3:
                st.download_button("Download current results (CSV)", data=recs.to_csv(index=False),
                                   file_name=f"recs_{donor_id}.csv", mime="text/csv")

# ----------------------------- INSIGHTS -----------------------------
with tab1:
    st.subheader("Quick insights")
    r = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(r):
        st.info("Generate recommendations first.")
    else:
        reg_counts = r["region"].value_counts()
        fig1, ax1 = plt.subplots(figsize=FIG_S)
        ax1.barh(reg_counts.index, reg_counts.values, color=take_colors(len(reg_counts)))
        ax1.set_title("Regions in recommended list"); ax1.invert_yaxis()
        st.pyplot(fig1)

        sec_counts = r["sector_focus"].value_counts()
        fig2, ax2 = plt.subplots(figsize=FIG_S)
        ax2.pie(sec_counts.values, labels=None, startangle=90, colors=take_colors(len(sec_counts)))
        centre_circle = plt.Circle((0,0), 0.55, fc='white'); fig2.gca().add_artist(centre_circle)
        ax2.set_title("Sectors (share of top picks)")
        st.pyplot(fig2)
        st.caption("Legend: " + ", ".join([f"{lab} ({val})" for lab, val in zip(sec_counts.index.tolist(), sec_counts.values.tolist())]))

        fig3, ax3 = plt.subplots(figsize=FIG_S)
        ax3.hist(r["funding_target"].astype(float), bins=12, color=PALETTE[4], edgecolor="white")
        ax3.set_title("Funding target distribution"); ax3.set_xlabel("Target"); ax3.set_ylabel("Count")
        st.pyplot(fig3)

# ----------------------------- DONOR PROGRESS -----------------------------
with tab2:
    st.subheader("Donor progress and giving")
    donor_id = st.session_state.get("selected_donor_id", donors["donor_id"].iloc[0])
    drow = donors[donors["donor_id"]==donor_id].iloc[0]

    current_bt = drow.get("behaviour_type") if "behaviour_type" in donors.columns else "active"
    if pd.isna(current_bt) or str(current_bt).strip()=="":
        current_bt = "active"
    st.markdown("**Current status:** " + status_dot_html(current_bt), unsafe_allow_html=True)
    new_bt = st.selectbox("Behaviour type", ["active","passive","selective"], index=["active","passive","selective"].index(current_bt))
    if new_bt != current_bt:
        donors.loc[donors["donor_id"]==donor_id, "behaviour_type"] = new_bt
        st.session_state["donors"] = donors
        st.success("Behaviour updated.")

    hist = interactions[interactions["Donor_ID"] == donor_id] if has_rows(interactions) else pd.DataFrame()

    avg_gift = drow.get("avg_gift", np.nan)
    lifetime_given = drow.get("lifetime_given", np.nan)
    if (pd.isna(avg_gift) or pd.isna(lifetime_given)) and has_rows(hist):
        alpha = 0.05
        f_map = projects.set_index("project_id")["funding_target"].to_dict()
        amounts = []
        for _, row_h in hist.iterrows():
            ft = float(f_map.get(row_h["Project_ID"], 0))
            cap = float(drow.get("budget_cap", ft)) if pd.notna(drow.get("budget_cap", np.nan)) else ft
            est = min(ft, cap) * alpha * float(row_h["Score"])
            amounts.append(est)
        if amounts:
            if pd.isna(avg_gift): avg_gift = float(np.mean(amounts))
            if pd.isna(lifetime_given): lifetime_given = float(np.sum(amounts))

    a,b,c,d = st.columns(4)
    a.metric("Known interactions", 0 if not has_rows(hist) else len(hist))
    b.metric("Unique projects", 0 if not has_rows(hist) else hist["Project_ID"].nunique())
    c.metric("Avg gift (est.)", human_money(avg_gift if pd.notna(avg_gift) else 0))
    d.metric("Lifetime given (est.)", human_money(lifetime_given if pd.notna(lifetime_given) else 0))

# ----------------------------- METRICS -----------------------------
with tab3:
    st.subheader("Evaluation metrics (donor-level)")
    donor_id = st.session_state.get("selected_donor_id", donors["donor_id"].iloc[0])
    r = st.session_state.get("recs", pd.DataFrame())

    if not has_rows(r):
        st.info("Generate recommendations first.")
    else:
        thr_mode = st.selectbox(
            "Relevance threshold for evaluation",
            ["Median per donor", "60th percentile per donor", "Fixed ≥ 0.6"],
            index=0
        )
        custom_thr = None
        if thr_mode == "Fixed ≥ 0.6":
            custom_thr = st.slider("Fixed threshold", 0.0, 1.0, 0.60, 0.01)

        hist_d = interactions[interactions["Donor_ID"] == donor_id] if has_rows(interactions) else pd.DataFrame()
        donor_has_history = has_rows(hist_d)

        rel_thr = None
        if donor_has_history:
            scores = hist_d["Score"].astype(float)
            if thr_mode == "Median per donor":
                rel_thr = float(scores.median())
            elif thr_mode == "60th percentile per donor":
                rel_thr = float(scores.quantile(0.60))
            else:
                rel_thr = float(custom_thr if custom_thr is not None else 0.60)

        relevant_items = set()
        if donor_has_history and rel_thr is not None:
            relevant_items = set(hist_d.loc[hist_d["Score"].astype(float) >= rel_thr, "Project_ID"].astype(str))

        k = st.selectbox("Top-K for evaluation", [5, 10], index=0)
        topk_df = r.head(k).copy()
        topk_df["project_id"] = topk_df["project_id"].astype(str)
        topk_list = topk_df["project_id"].tolist()

        precisionk = 0.0; recallk = 0.0; mapk = 0.0
        overlap_flags = []
        if donor_has_history and relevant_items:
            hits = [pid for pid in topk_list if pid in relevant_items]
            precisionk = len(hits) / float(k)
            recallk = len(hits) / float(len(relevant_items)) if relevant_items else 0.0

            running_sum = 0.0; hit_count = 0
            for idx, pid in enumerate(topk_list, start=1):
                is_hit = (pid in relevant_items)
                overlap_flags.append("✓" if is_hit else "–")
                if is_hit:
                    hit_count += 1
                    running_sum += hit_count / idx
            mapk = running_sum / float(min(len(relevant_items), k)) if relevant_items else 0.0
        else:
            overlap_flags = ["–"] * len(topk_list)

        coveragek = 0.0
        if has_rows(interactions) and len(topk_list) > 0:
            seen_items = set(interactions["Project_ID"].astype(str).unique())
            n_seen = sum(int(pid in seen_items) for pid in topk_list)
            coveragek = n_seen / len(topk_list)

        mae = mse = rmse = 0.0
        if svd_art is not None and donor_has_history:
            y_true, y_pred = [], []
            for _, row_h in hist_d.iterrows():
                pid = str(row_h["Project_ID"])
                try:
                    pred = svd_art.predict(donor_id, pid).est
                    if pd.notna(pred) and pd.notna(row_h["Score"]):
                        y_true.append(float(row_h["Score"]))
                        y_pred.append(float(pred))
                except Exception:
                    continue
            if len(y_true) >= 1:
                y_true = np.array(y_true, dtype=float)
                y_pred = np.array(y_pred, dtype=float)
                abs_err = np.abs(y_pred - y_true)
                sq_err  = (y_pred - y_true)**2
                mae = float(abs_err.mean()); mse = float(sq_err.mean()); rmse = float(np.sqrt(sq_err.mean()))

        novelty = float((1.0/(1.0 + r["popularity"].fillna(0))).mean()) if has_rows(r) else 0.0
        diversity = (r["sector_focus"].nunique() / max(1, projects["sector_focus"].nunique())) if has_rows(r) else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision@K", f"{precisionk*100:.1f}%")
        col2.metric("Recall@K", f"{recallk*100:.1f}%")
        col3.metric("MAP@K", f"{mapk*100:.1f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Coverage@K (CF)", f"{coveragek*100:.1f}%")
        col5.metric("Novelty ↑", f"{novelty:.2f}")
        col6.metric("Diversity@K", f"{diversity*100:.1f}%")

        with st.expander("Preview overlap", expanded=False):
            if len(topk_list) == 0:
                st.write("No recommendations yet.")
            else:
                prev = topk_df[["project_id","title","region","sector_focus","hybrid_score"]].copy()
                prev.insert(1, "Relevant?", overlap_flags)
                st.dataframe(prev, use_container_width=True)

# ----------------------------- WHY THESE PICKS -----------------------------
with tab4:
    st.subheader("Why these picks")
    r = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(r):
        st.info("Generate recommendations first.")
    else:
        for i,row in r.iterrows():
            comp = pd.DataFrame({"Rule":[row["rule_score_norm"]],
                                 "Content":[row["cosine_score_norm"]],
                                 "CF":[row["cf_score_norm"]]})
            st.markdown(f"**{i+1}. {row['title']}** — {row['region']} • {row['sector_focus']} • {row['organisation_type']} • Target {human_money(row['funding_target'])}")
            st.caption(f"Why matched: {row['why']}")
            fig, ax = plt.subplots(figsize=FIG_XS)
            comp.T[0].fillna(0).plot(kind="bar", ax=ax, color=take_colors(3))
            ax.set_ylim(0,1); ax.set_title("Score contribution"); ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)

# ----------------------------- EXPLORE PROJECTS -----------------------------
with tab5:
    st.subheader("Explore projects")
    c1,c2,c3,c4 = st.columns(4)
    with c1: reg_sel = st.multiselect("Region", sorted(projects["region"].dropna().unique().tolist()))
    with c2: sec_sel = st.multiselect("Sector", sorted(projects["sector_focus"].dropna().unique().tolist()))
    with c3: org_sel = st.multiselect("Org type", sorted(projects["organisation_type"].dropna().unique().tolist()))
    with c4: max_budget = st.number_input("Max funding target", min_value=0, value=int(projects["funding_target"].max()))
    cols_to_show = st.multiselect("Columns to display", options=list(projects.columns),
                                  default=["project_id","title","region","sector_focus","organisation_type","funding_target","popularity"])
    search = st.text_input("Search in title")
    df = projects.copy()
    if reg_sel: df = df[df["region"].isin(reg_sel)]
    if sec_sel: df = df[df["sector_focus"].isin(sec_sel)]
    if org_sel: df = df[df["organisation_type"].isin(org_sel)]
    if max_budget: df = df[df["funding_target"] <= max_budget]
    if search: df = df[df["title"].str.contains(search, case=False, na=False)]
    st.write(f"{len(df)} projects")
    st.dataframe(df[cols_to_show])

    if not df.empty and "funding_target" in df.columns:
        figE, axE = plt.subplots(figsize=FIG_S)
        axE.hist(df["funding_target"].astype(float), bins=15, color=PALETTE[2], edgecolor="white")
        axE.set_title("Funding target distribution (filtered)")
        st.pyplot(figE)

# ----------------------------- COMPARE ALGORITHMS -----------------------------
with tab6:
    st.subheader("Compare algorithms")
    donor_id = st.session_state.get("selected_donor_id", donors["donor_id"].iloc[0])

    def run_algo(weights, label):
        recs,_ = get_recs(donor_id, donors, projects, interactions, svd_art, None, None, None, weights,
                          {"region":[], "sector":[], "budget":None}, ethical=False, topk=5,
                          override_regions=parse_multi(donors.loc[donors["donor_id"]==donor_id,"region_preference"].iloc[0]),
                          override_sectors=parse_multi(donors.loc[donors["donor_id"]==donor_id,"sector_preference"].iloc[0]))
        if not has_rows(recs): return pd.DataFrame()
        out = recs[["title","region","sector_focus","organisation_type","funding_target","hybrid_score"]].copy()
        out.rename(columns={"hybrid_score":f"{label} score"}, inplace=True)
        return out

    colA,colB = st.columns(2)
    colC,colD = st.columns(2)
    colA.write("Rule-based Top-5"); colA.dataframe(safe_df(run_algo((1,0,0), "Rule")))
    colB.write("Content Cosine Top-5"); colB.dataframe(safe_df(run_algo((0,1,0), "Content")))
    colC.write("Collaborative (SVD) Top-5"); colC.dataframe(safe_df(run_algo((0,0,1), "CF")))
    colD.write("Hybrid Top-5"); colD.dataframe(safe_df(run_algo((0.33,0.33,0.34), "Hybrid")))

# ----------------------------- DIAGNOSTICS -----------------------------
with tab7:
    st.subheader("Data diagnostics")
    donors_ok = has_rows(donors)
    inter_ok = has_rows(interactions)

    total_donors = len(donors) if donors_ok else 0
    total_projects = len(projects) if has_rows(projects) else 0
    total_interactions = 0 if not inter_ok else len(interactions)

    hist_ids = set(interactions["Donor_ID"].astype(str)) if inter_ok else set()
    hist_proj = set(interactions["Project_ID"].astype(str)) if inter_ok else set()
    donors_ids = set(donors["donor_id"].astype(str)) if donors_ok else set()
    proj_ids = set(projects["project_id"].astype(str)) if has_rows(projects) else set()

    donors_with_hist_ids = donors_ids.intersection(hist_ids)
    projects_with_hist_ids = proj_ids.intersection(hist_proj)

    labels = ["All donors", "Donors\nwith history", "All projects", "Projects\nwith history"]
    values = [total_donors, len(donors_with_hist_ids), total_projects, len(projects_with_hist_ids)]

    fig, ax = plt.subplots(figsize=FIG_S)
    ax.bar(labels, values, color=take_colors(len(labels)))
    ax.set_title("Coverage (overlap with interactions)")
    ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)
    ax.margins(x=0.05)
    plt.tight_layout()
    st.pyplot(fig)

    if donors_ok and len(donors_with_hist_ids) > 0:
        sample_ids = list(donors_with_hist_ids)[:10]
        sample_df = donors[donors["donor_id"].isin(sample_ids)][["donor_id","name","email","region_preference","sector_preference"]]
        st.markdown("**Sample donors with history (first 10):**")
        st.dataframe(sample_df)
    else:
        st.info("No overlapping donor IDs found yet.")
