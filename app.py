# app.py — Diaspora Donor Recommender (Streamlit Cloud, precomputed CF in artifacts/)
import os, json, time, re
import numpy as np, pandas as pd
import streamlit as st, matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Optional PDF export (won't crash if missing)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# -------------------- PAGE / CACHE --------------------
st.set_page_config(page_title="Diaspora Donor Recommender", page_icon="🤝", layout="wide")
APP_VERSION = "2025-08-17-streamlit-cloud-precomputed-cf"
if st.session_state.get("_app_version") != APP_VERSION:
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["_app_version"] = APP_VERSION

BASE = "artifacts"  # everything lives here on Streamlit Cloud
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- SMALL VIZ / UX --------------------
FIG_XS = (3.0, 2.0)   # smaller charts
FIG_S  = (3.6, 2.4)
SMALL_TITLE = 10
SMALL_LABEL = 8

# -------------------- HELPERS --------------------
def parse_multi(val):
    if val is None: return []
    if isinstance(val, list): return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).replace("|",";").replace(",",";")
    return [p.strip() for p in s.split(";") if p.strip()]

def normalize(df, col):
    scaler = MinMaxScaler()
    vals = df[[col]].astype(float)
    if vals.max().item() == vals.min().item():
        df[col+"_norm"] = 0.5
    else:
        df[col+"_norm"] = scaler.fit_transform(vals)
    return df

def human_money(x):
    try:
        x = float(x)
        if x >= 1_000_000: return f"${x/1_000_000:.1f}M"
        if x >= 1_000: return f"${x/1_000:.1f}k"
        return f"${int(x)}"
    except Exception:
        return str(x)

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

def load_csv_or_parquet(path):
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        return pd.read_csv(path)
    return pd.read_parquet(path)

# -------------------- LOAD DATA --------------------
@st.cache_data(show_spinner=False)
def load_core(base):
    # donors / projects
    donors_path   = os.path.join(base, "donors_5000.csv") if os.path.exists(os.path.join(base,"donors_5000.csv")) else os.path.join(base, "donors.csv")
    projects_path = os.path.join(base, "projects_2000.csv") if os.path.exists(os.path.join(base,"projects_2000.csv")) else os.path.join(base, "projects.csv")
    donors = load_csv_or_parquet(donors_path)
    projects = load_csv_or_parquet(projects_path)

    # optional interactions (for metrics only)
    inter_path = None
    for n in ["synthetic_interactions_5000x2000.csv","interactions.csv","ratings.csv"]:
        p = os.path.join(base, n)
        if os.path.exists(p): inter_path = p; break
    interactions = load_csv_or_parquet(inter_path) if inter_path else pd.DataFrame(columns=["Donor_ID","Project_ID","Score"])

    # normalize columns
    donors.columns = [c.strip() for c in donors.columns]
    projects.columns = [c.strip().lower() for c in projects.columns]
    interactions.columns = [c.strip() for c in interactions.columns]

    # projects numeric
    if "funding_target" in projects.columns:
        projects["funding_target"] = pd.to_numeric(projects["funding_target"], errors="coerce").fillna(projects["funding_target"].median() if "funding_target" in projects else 0)
    if "popularity" in projects.columns:
        projects["popularity"] = pd.to_numeric(projects["popularity"], errors="coerce").fillna(0)

    # interactions schema unify
    if has_rows(interactions):
        interactions = interactions.rename(columns={c:c.lower() for c in interactions.columns})
        # We accept donor_id / project_id / score (any casing)
        # map common names
        col_map = {}
        if "donor_id" not in interactions.columns:
            for c in interactions.columns:
                if c.lower() in ["donor","user","user_id","uid"]: col_map[c] = "donor_id"
        if "project_id" not in interactions.columns:
            for c in interactions.columns:
                if c.lower() in ["item","item_id","pid","project"]: col_map[c] = "project_id"
        if "score" not in interactions.columns:
            for c in interactions.columns:
                if c.lower() in ["rating","rank","value"]: col_map[c] = "score"
        if col_map:
            interactions = interactions.rename(columns=col_map)
        # keep only needed
        keep = [c for c in ["donor_id","project_id","score"] if c in interactions.columns]
        interactions = interactions[keep] if keep else pd.DataFrame(columns=["donor_id","project_id","score"])
        if has_rows(interactions):
            interactions["donor_id"] = interactions["donor_id"].astype(str)
            interactions["project_id"] = interactions["project_id"].astype(str)
            interactions["score"] = pd.to_numeric(interactions["score"], errors="coerce").fillna(0)

    # precomputed CF (donor_id, project_id, est)
    cf_path = os.path.join(base, "cf_estimates.csv.gz")
    if os.path.exists(cf_path):
        cf_df = pd.read_csv(cf_path)
        cf_df.columns = [c.strip().lower() for c in cf_df.columns]
        # Accept columns like donor_id/project_id/est
        must = {"donor_id","project_id","est"}
        if not must.issubset(set(cf_df.columns)):
            # try alternate naming
            alt = {}
            for c in cf_df.columns:
                if c.lower() in ["user","user_id","uid","donor"]: alt[c] = "donor_id"
                if c.lower() in ["item","item_id","pid","project"]: alt[c] = "project_id"
                if c.lower() in ["prediction","pred","score","estimate"]: alt[c] = "est"
            cf_df = cf_df.rename(columns=alt)
        if must.issubset(set(cf_df.columns)):
            cf_ok = True
            cf_df["donor_id"] = cf_df["donor_id"].astype(str)
            cf_df["project_id"] = cf_df["project_id"].astype(str)
            cf_df["est"] = pd.to_numeric(cf_df["est"], errors="coerce")
        else:
            cf_ok = False
            cf_df = pd.DataFrame(columns=["donor_id","project_id","est"])
    else:
        cf_ok = False
        cf_df = pd.DataFrame(columns=["donor_id","project_id","est"])

    return donors, projects, interactions, cf_df, cf_ok

donors, projects, interactions, cf_df, cf_ok = load_core(BASE)

# -------------------- SIMPLE CONTENT VECTORS --------------------
def build_proj_vectors_on_the_fly(projects_df):
    # One-hot on (region, sector_focus)
    regions = sorted(projects_df["region"].dropna().unique().tolist()) if "region" in projects_df.columns else []
    sectors = sorted(projects_df["sector_focus"].dropna().unique().tolist()) if "sector_focus" in projects_df.columns else []
    region_cols = [f"region__{r}" for r in regions]
    sector_cols = [f"sector__{s}" for s in sectors]
    rows = []
    for _, r in projects_df.iterrows():
        vec = {c:0 for c in region_cols + sector_cols}
        if "region" in r and pd.notna(r["region"]) and f"region__{r['region']}" in vec:
            vec[f"region__{r['region']}"] = 1
        if "sector_focus" in r and pd.notna(r["sector_focus"]) and f"sector__{r['sector_focus']}" in vec:
            vec[f"sector__{r['sector_focus']}"] = 1
        vec["project_id"] = str(r["project_id"])
        rows.append(vec)
    pv = pd.DataFrame(rows)
    feature_cols = [c for c in pv.columns if c != "project_id"]
    return pv, feature_cols

# -------------------- SCORING --------------------
def rule_score(donor_row, proj_row):
    s = 0.0
    r_prefs = parse_multi(donor_row.get("region_preference"))
    s_prefs = parse_multi(donor_row.get("sector_preference"))
    if "region" in proj_row and proj_row["region"] in r_prefs: s += 0.5
    if "sector_focus" in proj_row and proj_row["sector_focus"] in s_prefs: s += 0.5
    try:
        pref_target = float(donor_row.get("preferred_target", np.nan))
        if not np.isnan(pref_target) and "funding_target" in proj_row:
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

def build_donor_vector_from_prefs(pref_regions, pref_sectors, feature_cols):
    vec = {c:0 for c in feature_cols}
    for r in pref_regions:
        k = f"region__{r}"
        if k in vec: vec[k] = 1
    for s in pref_sectors:
        k = f"sector__{s}"
        if k in vec: vec[k] = 1
    return np.array([vec[c] for c in feature_cols], dtype=float).reshape(1, -1)

def get_cf_score_for_pairs(cf_lookup, donor_id, project_ids):
    # cf_lookup: dict[(donor_id, project_id)] -> est
    out = []
    for pid in project_ids:
        est = cf_lookup.get((donor_id, str(pid)))
        out.append(np.nan if est is None else est)
    return pd.Series(out, index=project_ids)

def get_recs(
    donor_id, donors_df, projects_df, interactions_df, cf_df, cf_ok,
    weights=(0.3,0.4,0.3), filters=None, ethical=True, topk=10,
    override_regions=None, override_sectors=None, proj_vecs=None, feats=None
):
    drow = donors_df[donors_df["donor_id"]==donor_id]
    if drow.empty: return pd.DataFrame(), "Unknown donor_id"
    drow = drow.iloc[0].copy()

    cand = projects_df.copy()
    # ethical filter: downweight very popular; here we simply cap popularity
    if ethical and "popularity" in cand.columns and len(cand) > 0:
        p90 = cand["popularity"].quantile(0.9)
        cand = cand[cand["popularity"] <= p90].copy()

    if filters:
        if filters.get("region"): cand = cand[cand["region"].isin(filters["region"])].copy()
        if filters.get("sector"): cand = cand[cand["sector_focus"].isin(filters["sector"])].copy()
        if filters.get("budget") is not None: cand = cand[cand["funding_target"] <= filters["budget"]].copy()
    if cand.empty: return pd.DataFrame(), "No projects left after filtering."

    # Rule score
    cand["rule_score"] = [rule_score(drow, prow) for _, prow in cand.iterrows()]
    cand = normalize(cand, "rule_score")

    # Content score (cosine) from one-hot region/sector
    if proj_vecs is None or feats is None:
        proj_vecs, feats = build_proj_vectors_on_the_fly(projects_df)
    pv = proj_vecs.set_index("project_id").reindex(cand["project_id"].astype(str)).fillna(0.0).astype(float).values
    pref_regions = override_regions if override_regions is not None else parse_multi(drow.get("region_preference"))
    pref_sectors = override_sectors if override_sectors is not None else parse_multi(drow.get("sector_preference"))
    dv = build_donor_vector_from_prefs(pref_regions, pref_sectors, feats)
    cand["cosine_score"] = cosine_similarity(pv, dv).ravel()
    cand = normalize(cand, "cosine_score")

    # CF score: from precomputed estimates
    if cf_ok and has_rows(cf_df):
        cf_lookup = {(str(r["donor_id"]), str(r["project_id"])): r["est"] for _, r in cf_df.iterrows()}
        series_cf = get_cf_score_for_pairs(cf_lookup, str(donor_id), cand["project_id"].astype(str).tolist())
        cand["cf_score"] = series_cf.values
    else:
        cand["cf_score"] = np.nan
    # backfill with content if CF missing for any
    mask_nan = cand["cf_score"].isna()
    if mask_nan.any():
        cand.loc[mask_nan, "cf_score"] = cand.loc[mask_nan, "cosine_score"]
    cand = normalize(cand, "cf_score")

    # Blend
    w_rule, w_cos, w_cf = weights
    cand["hybrid_score"] = w_rule*cand["rule_score_norm"] + w_cos*cand["cosine_score_norm"] + w_cf*cand["cf_score_norm"]

    # Why text
    why = []
    for _, r in cand.iterrows():
        parts = []
        if r["region"] in pref_regions: parts.append("Region match")
        if r["sector_focus"] in pref_sectors: parts.append("Sector match")
        if r["cosine_score_norm"] > 0.6: parts.append("High content similarity")
        if r["cf_score_norm"] > 0.6 and cf_ok: parts.append("Similar donors liked this (CF)")
        if not parts: parts = ["Strong blended score"]
        why.append("; ".join(parts))
    cand["why"] = why

    cols = ["project_id","title","region","sector_focus","funding_target","organisation_type","popularity",
            "rule_score_norm","cosine_score_norm","cf_score_norm","hybrid_score","why"]
    recs = cand[cols].fillna(0).sort_values("hybrid_score", ascending=False).head(topk).reset_index(drop=True)
    return recs, None

# -------------------- TABS FIRST (as requested) --------------------
tab_home, tab_insights, tab_progress, tab_metrics, tab_why, tab_explore, tab_compare, tab_diag = st.tabs(
    ["Home", "Insights", "Donor progress", "Metrics", "Why these picks", "Explore projects", "Compare algorithms", "Diagnostics"]
)

# -------------------- HOME (find donor + recs) --------------------
with tab_home:
    st.subheader("Dataset & model status")
    n_users = donors["donor_id"].nunique() if "donor_id" in donors.columns else len(donors)
    n_items = projects["project_id"].nunique() if "project_id" in projects.columns else len(projects)
    st.write(f"Users: **{n_users}**, Items: **{n_items}**")
    st.write("CF source:", "precomputed estimates ✅" if cf_ok else "not found ❌")

    left, right = st.columns([0.42, 0.58])
    with left:
        st.subheader("Find donor and set preferences")

        # search box
        query = st.text_input("Search donor (ID, name or email)")
        ddf = donors.copy()
        if query:
            q = query.lower()
            def row_match(r):
                return (q in str(r.get("donor_id","")).lower()) or (q in str(r.get("name","")).lower()) or (q in str(r.get("email","")).lower())
            ddf = ddf[ddf.apply(row_match, axis=1)]

        # label with tick if donor appears in interactions
        hist_ids = set(interactions["donor_id"].astype(str)) if has_rows(interactions) and "donor_id" in interactions.columns else set()
        if "donor_id" in ddf.columns:
            ddf["label"] = ddf.apply(lambda r: f"{r['donor_id']} - {r.get('name','')}" + (" ✅" if str(r['donor_id']) in hist_ids else ""), axis=1)
            options = ddf["label"].tolist()
        else:
            st.error("`donor_id` column missing in donors file.")
            options = []

        if not options:
            st.stop()

        donor_label = st.selectbox("Choose donor", options, index=0)
        donor_id = donor_label.split(" - ")[0].strip()
        drow = donors[donors["donor_id"]==donor_id].iloc[0]

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
        all_regions = sorted(projects["region"].dropna().unique().tolist()) if "region" in projects.columns else []
        all_sectors = sorted(projects["sector_focus"].dropna().unique().tolist()) if "sector_focus" in projects.columns else []
        default_regions = [r for r in parse_multi(drow.get("region_preference")) if r in all_regions]
        default_sectors = [s for s in parse_multi(drow.get("sector_preference")) if s in all_sectors]
        ui_regions = st.multiselect("Preference: Regions (multi)", options=all_regions, default=default_regions)
        ui_sectors = st.multiselect("Preference: Sectors (multi)", options=all_sectors, default=default_sectors)

        pref_target = st.number_input("Preferred project funding target", min_value=0, value=int(drow.get("preferred_target", 0)) if pd.notna(drow.get("preferred_target", np.nan)) else 0, step=1000)
        budget = st.slider("Budget cap (filters funding target ≤)", 0, int(projects["funding_target"].max() if "funding_target" in projects.columns else 100000), int(projects["funding_target"].quantile(0.5) if "funding_target" in projects.columns else 10000), 1000)

        st.markdown("**Hybrid weights**")
        w_rule = st.slider("Rule-based", 0.0, 1.0, 0.30, 0.05)
        w_cos  = st.slider("Content (Cosine)", 0.0, 1.0, 0.40, 0.05)
        w_cf   = st.slider("Collaborative (CF)", 0.0, 1.0, 0.30, 0.05)

        ethical = st.toggle("Ethical AI (reduce over-exposed items)", value=True)
        hybrid_mode = st.toggle("Hybrid mode (blend all three)", value=True)

        go = st.button("Get recommendations", type="primary")
        clear_btn = st.button("Clear shortlist")

    with right:
        st.subheader("Top recommendations")
        if "shortlist" not in st.session_state: st.session_state["shortlist"] = pd.DataFrame()
        if clear_btn:
            st.session_state["shortlist"] = pd.DataFrame()
            st.info("Shortlist cleared.")

        if go:
            weights = (w_rule,w_cos,w_cf) if hybrid_mode else (1.0,0.0,0.0)
            recs, err = get_recs(
                donor_id, donors, projects, interactions, cf_df, cf_ok,
                weights=weights,
                filters={"region": ui_regions, "sector": ui_sectors, "budget": budget},
                ethical=ethical, topk=10,
                override_regions=ui_regions, override_sectors=ui_sectors
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
                with st.container():
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
                if REPORTLAB_OK:
                    def export_pdf(recs_df, donor_row, path):
                        c = canvas.Canvas(path, pagesize=A4); w,h = A4; y = h-40
                        c.setFont("Helvetica-Bold", 14); c.drawString(40,y,"Diaspora Donor Recommender — Top Picks"); y-=22
                        c.setFont("Helvetica", 10); c.drawString(40,y,f"Donor: {donor_row.get('name','')} ({donor_row.get('donor_id','')})"); y-=16
                        c.drawString(40,y,f"Prefs: {donor_row.get('region_preference','')} / {donor_row.get('sector_preference','')}"); y-=24
                        for i,row in recs_df.iterrows():
                            if y<80: c.showPage(); y=h-40
                            c.setFont("Helvetica-Bold",11); c.drawString(40,y,f"{i+1}. {row['title']}"); y-=14
                            c.setFont("Helvetica",10); c.drawString(40,y,f"{row['region']} • {row['sector_focus']} • {row.get('organisation_type','')} | Target {int(row.get('funding_target',0))} | Score {row.get('hybrid_score',0):.2f}"); y-=18
                        c.save()
                    if st.button("Save shortlist (PDF)"):
                        p = os.path.join(OUTPUT_DIR, f"shortlist_{donor_id}_{int(time.time())}.pdf")
                        export_pdf(st.session_state["shortlist"], drow, p); st.success(f"Saved: {p}")
                else:
                    st.caption("Install `reportlab` to enable PDF export.")
            with c3:
                st.download_button("Download current results (CSV)", data=recs.to_csv(index=False), file_name=f"recs_{donor_id}.csv", mime="text/csv")

# -------------------- INSIGHTS --------------------
with tab_insights:
    st.subheader("Quick insights")
    r = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(r):
        st.info("Generate recommendations first.")
    else:
        # Regions bar
        if "region" in r.columns:
            reg_counts = r["region"].value_counts()
            fig1, ax1 = plt.subplots(figsize=FIG_S)
            ax1.barh(reg_counts.index, reg_counts.values)
            ax1.set_title("Regions in recommended list", fontsize=SMALL_TITLE)
            ax1.tick_params(axis='both', labelsize=SMALL_LABEL)
            ax1.invert_yaxis()
            st.pyplot(fig1)

        # Sectors pie
        if "sector_focus" in r.columns:
            sec_counts = r["sector_focus"].value_counts()
            fig2, ax2 = plt.subplots(figsize=FIG_S)
            ax2.pie(sec_counts.values, labels=None, startangle=90)
            centre_circle = plt.Circle((0,0), 0.55, fc='white')
            fig2.gca().add_artist(centre_circle)
            ax2.set_title("Sectors (share of top picks)", fontsize=SMALL_TITLE)
            st.pyplot(fig2)
            st.caption("Legend: " + ", ".join([f"{lab} ({val})" for lab, val in zip(sec_counts.index.tolist(), sec_counts.values.tolist())]))

        # Funding targets hist
        if "funding_target" in r.columns:
            fig3, ax3 = plt.subplots(figsize=FIG_S)
            ax3.hist(r["funding_target"].astype(float), bins=12)
            ax3.set_title("Funding target distribution", fontsize=SMALL_TITLE)
            ax3.set_xlabel("Target"); ax3.set_ylabel("Count")
            ax3.tick_params(axis='both', labelsize=SMALL_LABEL)
            st.pyplot(fig3)

# -------------------- DONOR PROGRESS --------------------
with tab_progress:
    st.subheader("Donor progress & giving")
    # Simple snapshot: just show current donor if available
    recs = st.session_state.get("recs", pd.DataFrame())
    st.dataframe(donors.head())

# -------------------- METRICS --------------------
with tab_metrics:
    st.subheader("Evaluation metrics (donor-level)")

    # choose donor
    if "donor_id" not in donors.columns:
        st.info("No donor_id column.")
    else:
        donor_id_sel = st.selectbox("Choose donor for metrics", donors["donor_id"].astype(str).tolist(), index=0)

        # Top-K table
        r = st.session_state.get("recs", pd.DataFrame())
        topk = r.head(10).copy() if has_rows(r) else pd.DataFrame()

        # Build relevant set from interactions for this donor
        hist_d = pd.DataFrame()
        relevant = set()
        if has_rows(interactions):
            hist_d = interactions[interactions["donor_id"].astype(str)==str(donor_id_sel)].copy()
            if has_rows(hist_d) and "score" in hist_d.columns:
                thr = hist_d["score"].median()
                relevant = set(hist_d.loc[hist_d["score"]>=thr, "project_id"].astype(str))

        # Precision/Recall/MAP
        k = st.selectbox("Top-K for evaluation", [5,10], index=0)
        precisionk = recallk = mapk = 0.0
        overlap_flags = []
        if has_rows(topk) and relevant:
            top_ids = topk.head(k)["project_id"].astype(str).tolist()
            hits = [pid for pid in top_ids if pid in relevant]
            precisionk = len(hits) / float(k)
            recallk = len(hits) / float(len(relevant)) if relevant else 0.0

            running_sum = 0.0; hit_count = 0
            for idx, pid in enumerate(top_ids, start=1):
                is_hit = (pid in relevant)
                overlap_flags.append("✓" if is_hit else "–")
                if is_hit:
                    hit_count += 1
                    running_sum += hit_count / idx
            mapk = running_sum / float(min(len(relevant), k)) if relevant else 0.0
        elif has_rows(topk):
            overlap_flags = ["–"] * len(topk.head(k))

        # Coverage@K based on CF availability
        coveragek = 0.0
        if has_rows(topk) and cf_ok:
            seen_pairs = set((str(d), str(p)) for d,p in zip(cf_df["donor_id"], cf_df["project_id"]))
            top_ids = topk.head(k)["project_id"].astype(str).tolist()
            n_seen = sum((str(donor_id_sel), pid) in seen_pairs for pid in top_ids)
            coveragek = n_seen / float(len(top_ids)) if top_ids else 0.0

        # Error metrics (MAE/MSE/RMSE) on overlap between history and cf_estimates
        mae = mse = rmse = 0.0
        if has_rows(hist_d) and cf_ok:
            # Join on (donor_id, project_id)
            hist_d["donor_id"] = hist_d["donor_id"].astype(str)
            hist_d["project_id"] = hist_d["project_id"].astype(str)
            sub_cf = cf_df[(cf_df["donor_id"].astype(str)==str(donor_id_sel))][["project_id","est"]].copy()
            joined = hist_d.merge(sub_cf, on="project_id", how="inner")
            if has_rows(joined) and "score" in joined.columns and "est" in joined.columns:
                y_true = pd.to_numeric(joined["score"], errors="coerce").dropna()
                y_pred = pd.to_numeric(joined.loc[y_true.index, "est"], errors="coerce")
                y_pred = y_pred.loc[y_true.index]
                if len(y_true) > 0:
                    err = (y_pred - y_true)
                    mae = float(np.mean(np.abs(err)))
                    mse = float(np.mean(err**2))
                    rmse = float(np.sqrt(mse))

        c1,c2,c3 = st.columns(3)
        c1.metric("Precision@K", f"{precisionk*100:.1f}%")
        c2.metric("Recall@K", f"{recallk*100:.1f}%")
        c3.metric("MAP@K", f"{mapk*100:.1f}%")
        d1,d2,d3 = st.columns(3)
        d1.metric("Coverage@K (CF)", f"{coveragek*100:.1f}%")
        d2.metric("MAE", f"{mae:.3f}")
        d3.metric("RMSE", f"{rmse:.3f}")

        # Mini overlap preview
        with st.expander("Top-K overlap preview"):
            if has_rows(topk):
                prev = topk.head(k)[["project_id","title","region","sector_focus","hybrid_score"]].copy()
                prev.insert(1, "Relevant?", overlap_flags[:len(prev)])
                st.dataframe(prev, use_container_width=True)
            else:
                st.info("No recommendations yet.")

# -------------------- WHY THESE PICKS --------------------
with tab_why:
    st.subheader("Why these picks")
    r = st.session_state.get("recs", pd.DataFrame())
    if not has_rows(r):
        st.info("Generate recommendations first.")
    else:
        for i,row in r.iterrows():
            comp = pd.DataFrame({"Rule":[row["rule_score_norm"]], "Content":[row["cosine_score_norm"]], "CF":[row["cf_score_norm"]]})
            st.markdown(f"**{i+1}. {row['title']}** — {row['region']} • {row['sector_focus']} • {row.get('organisation_type','')} • Target {human_money(row.get('funding_target',0))}")
            st.caption(f"Why matched: {row['why']}")
            fig, ax = plt.subplots(figsize=FIG_XS)
            comp.T[0].fillna(0).plot(kind="bar", ax=ax)
            ax.set_ylim(0,1)
            ax.set_title("Score contribution", fontsize=SMALL_TITLE)
            ax.tick_params(axis='x', labelsize=SMALL_LABEL, rotation=0)
            ax.tick_params(axis='y', labelsize=SMALL_LABEL)
            st.pyplot(fig)

# -------------------- EXPLORE PROJECTS --------------------
with tab_explore:
    st.subheader("Explore projects")
    c1,c2,c3,c4 = st.columns(4)
    with c1: reg_sel = st.multiselect("Region", sorted(projects["region"].dropna().unique().tolist()) if "region" in projects.columns else [])
    with c2: sec_sel = st.multiselect("Sector", sorted(projects["sector_focus"].dropna().unique().tolist()) if "sector_focus" in projects.columns else [])
    with c3: org_sel = st.multiselect("Org type", sorted(projects["organisation_type"].dropna().unique().tolist()) if "organisation_type" in projects.columns else [])
    with c4: max_budget = st.number_input("Max funding target", min_value=0, value=int(projects["funding_target"].max()) if "funding_target" in projects.columns else 0)
    cols_to_show = st.multiselect("Columns to display", options=list(projects.columns), default=[c for c in ["project_id","title","region","sector_focus","organisation_type","funding_target","popularity"] if c in projects.columns])
    search = st.text_input("Search in title")
    df = projects.copy()
    if reg_sel and "region" in df.columns: df = df[df["region"].isin(reg_sel)]
    if sec_sel and "sector_focus" in df.columns: df = df[df["sector_focus"].isin(sec_sel)]
    if org_sel and "organisation_type" in df.columns: df = df[df["organisation_type"].isin(org_sel)]
    if max_budget and "funding_target" in df.columns: df = df[df["funding_target"] <= max_budget]
    if search and "title" in df.columns: df = df[df["title"].str.contains(search, case=False, na=False)]
    st.write(f"{len(df)} projects")
    st.dataframe(df[cols_to_show])

    if not df.empty and "funding_target" in df.columns:
        figE, axE = plt.subplots(figsize=FIG_S)
        axE.hist(df["funding_target"].astype(float), bins=15)
        axE.set_title("Funding target distribution (filtered)", fontsize=SMALL_TITLE)
        axE.tick_params(axis='both', labelsize=SMALL_LABEL)
        st.pyplot(figE)

# -------------------- COMPARE ALGORITHMS --------------------
with tab_compare:
    st.subheader("Compare algorithms")
    def run_algo(weights, label):
        recs,_ = get_recs(
            donors["donor_id"].astype(str).iloc[0], donors, projects, interactions, cf_df, cf_ok,
            weights=weights, filters={"region":[], "sector":[],"budget":None}, ethical=False, topk=5
        )
        if not has_rows(recs): return pd.DataFrame()
        out = recs[["title","region","sector_focus","organisation_type","funding_target","hybrid_score"]].copy()
        out.rename(columns={"hybrid_score":f"{label} score"}, inplace=True)
        return out

    colA,colB = st.columns(2)
    colC,colD = st.columns(2)
    colA.write("Rule-based Top-5"); colA.dataframe(safe_df(run_algo((1,0,0), "Rule")))
    colB.write("Content Cosine Top-5"); colB.dataframe(safe_df(run_algo((0,1,0), "Content")))
    colC.write("Collaborative (CF) Top-5"); colC.dataframe(safe_df(run_algo((0,0,1), "CF")))
    colD.write("Hybrid Top-5"); colD.dataframe(safe_df(run_algo((0.33,0.33,0.34), "Hybrid")))

# -------------------- DIAGNOSTICS --------------------
with tab_diag:
    st.subheader("Diagnostics")
    st.write({
        "cf_loaded": bool(cf_ok),
        "cf_rows": int(len(cf_df)) if has_rows(cf_df) else 0,
        "donors": int(len(donors)),
        "projects": int(len(projects)),
        "interactions_rows": int(len(interactions)) if has_rows(interactions) else 0,
        "reportlab_installed": REPORTLAB_OK,
        "app_version": APP_VERSION
    })
