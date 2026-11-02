"""Customer Churn Intelligence - Streamlit dashboard.
Run:  python -m streamlit run app.py
Needs: streamlit, pandas, scikit-learn, joblib (altair ships with streamlit)
"""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "outputs" / "models"
TABLES = ROOT / "outputs" / "tables"
RAW = ROOT / "data" / "Telco_Customer_Churn_Dataset.csv"
ADDONS = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

st.set_page_config(page_title="Churn Intelligence", page_icon="📡", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }
#MainMenu, footer { visibility: hidden; }

.hero {
  background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 45%, #7c3aed 100%);
  border-radius: 22px; padding: 38px 42px; color: #fff; margin-bottom: 22px;
  box-shadow: 0 18px 40px rgba(30,58,138,.35); position: relative; overflow: hidden;
}
.hero:after { content:""; position:absolute; right:-80px; top:-80px; width:320px; height:320px;
  background: radial-gradient(circle, rgba(255,255,255,.18), transparent 70%); border-radius:50%; }
.hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing:-.5px; color:#fff; }
.hero p  { font-size: 1.05rem; opacity: .85; margin: 0; color:#fff; }
.pill { display:inline-block; background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
  padding: 4px 14px; border-radius: 999px; font-size:.78rem; margin: 14px 8px 0 0; color:#fff; }

.kpi { background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.22); border-radius:16px;
  padding: 18px 20px; transition: transform .2s, box-shadow .2s; }
.kpi:hover { transform: translateY(-4px); box-shadow: 0 10px 24px rgba(0,0,0,.15); }
.kpi .v { font-size: 1.9rem; font-weight: 800; background: linear-gradient(90deg,#3b82f6,#8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi .l { font-size:.82rem; opacity:.7; text-transform: uppercase; letter-spacing:.8px; }

.card { background: rgba(128,128,128,.07); border:1px solid rgba(128,128,128,.22); border-radius:18px;
  padding: 22px 24px; margin-bottom: 14px; }
.card h4 { margin: 0 0 8px 0; }
.badge { display:inline-block; padding:6px 18px; border-radius:999px; font-weight:700; color:#fff; font-size:.95rem; }
.action { border-left: 4px solid #8b5cf6; background: rgba(139,92,246,.10); padding: 10px 14px;
  border-radius: 8px; margin: 8px 0; font-size:.95rem; }

div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(90deg,#2563eb,#7c3aed); color:#fff; border:0; border-radius: 12px;
  padding: .7rem 1.6rem; font-weight:700; width:100%; transition: all .2s; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px); box-shadow: 0 8px 20px rgba(124,58,237,.45); color:#fff; }
button[data-baseweb="tab"] { font-weight:600; font-size:1rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- model + helpers
@st.cache_resource
def load_model():
    info = json.load(open(MODELS / "best_model_info.json"))
    try:
        return joblib.load(MODELS / "best_model.joblib"), info
    except Exception:  # scikit-learn version mismatch -> retrain quickly
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        tr = pd.read_csv(ROOT / "data" / "train.csv")
        pipe = Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k="all")),
                         ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                                                        class_weight="balanced", random_state=42, n_jobs=-1))])
        return pipe.fit(tr.drop(columns=["customerID", "Churn"]), tr["Churn"]), info


@st.cache_data
def load_raw():
    d = pd.read_csv(RAW)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    d["Churned"] = (d["Churn"] == "Yes").astype(int)
    return d


def encode(raw: pd.DataFrame, columns) -> pd.DataFrame:
    d = raw.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    for c in ADDONS:
        d[c] = d[c].replace({"No internet service": "No", "No phone service": "No"})
    o = pd.DataFrame(index=d.index)
    o["gender"] = (d["gender"] == "Male").astype(int)
    o["SeniorCitizen"] = d["SeniorCitizen"].astype(int)
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + ADDONS:
        o[c] = (d[c] == "Yes").astype(int)
    o["tenure"], o["MonthlyCharges"], o["TotalCharges"] = d["tenure"], d["MonthlyCharges"], d["TotalCharges"]
    o["InternetService_Fiber optic"] = (d["InternetService"] == "Fiber optic").astype(int)
    o["InternetService_No"] = (d["InternetService"] == "No").astype(int)
    o["Contract_One year"] = (d["Contract"] == "One year").astype(int)
    o["Contract_Two year"] = (d["Contract"] == "Two year").astype(int)
    o["PaymentMethod_Credit card (automatic)"] = (d["PaymentMethod"] == "Credit card (automatic)").astype(int)
    o["PaymentMethod_Electronic check"] = (d["PaymentMethod"] == "Electronic check").astype(int)
    o["PaymentMethod_Mailed check"] = (d["PaymentMethod"] == "Mailed check").astype(int)
    return o[list(columns)]


def risk(p):
    if p >= 0.5: return "HIGH RISK", "#ef4444"
    if p >= 0.3: return "MEDIUM RISK", "#f59e0b"
    return "LOW RISK", "#10b981"


def gauge(p, color):
    arc = 314.16
    return f"""
    <svg viewBox="0 0 240 150" style="width:100%;max-width:340px;display:block;margin:auto">
      <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0%" stop-color="#10b981"/>
        <stop offset="55%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#ef4444"/></linearGradient></defs>
      <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="rgba(128,128,128,.25)" stroke-width="20" stroke-linecap="round"/>
      <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="{color}" stroke-width="20" stroke-linecap="round"
            stroke-dasharray="{p*arc:.1f} {arc}"/>
      <text x="120" y="100" text-anchor="middle" font-size="40" font-weight="800" fill="{color}">{p:.0%}</text>
      <text x="120" y="125" text-anchor="middle" font-size="11" fill="gray" letter-spacing="2">CHURN SCORE</text>
      <text x="20" y="143" text-anchor="middle" font-size="9" fill="gray">0</text>
      <text x="220" y="143" text-anchor="middle" font-size="9" fill="gray">100</text>
    </svg>"""


model, info = load_model()
cols = info["all_features"]
df = load_raw()

# ----------------------------------------------------------------------------- hero + KPIs
st.markdown(f"""
<div class="hero">
  <h1>📡 Churn Intelligence</h1>
  <p>Predict which telecom customers are about to leave - and what to do about it.</p>
  <span class="pill">🤖 {info['best_model']}</span><span class="pill">📈 ROC-AUC 0.85</span>
  <span class="pill">🎯 Recall 78%</span><span class="pill">🏢 SaiKet Systems · Data Science Internship</span>
</div>""", unsafe_allow_html=True)

k = st.columns(4)
for c, (v, l) in zip(k, [(f"{len(df):,}", "Customers analysed"), (f"{df.Churned.mean():.1%}", "Overall churn rate"),
                         (f"${df.MonthlyCharges.mean():.0f}", "Avg monthly bill"), ("0.847", "Model ROC-AUC")]):
    c.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📂 Batch scoring", "📊 Insights", "ℹ️ About"])

# ----------------------------------------------------------------------------- Predict
with tab1:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.form("customer"):
            st.markdown("##### 👤 Customer profile")
            a, b, c, d_ = st.columns(4)
            gender = a.selectbox("Gender", ["Female", "Male"])
            senior = b.selectbox("Senior", ["No", "Yes"])
            partner = c.selectbox("Partner", ["No", "Yes"])
            dependents = d_.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 6)

            st.markdown("##### 🌐 Services")
            a, b, c = st.columns(3)
            internet = a.selectbox("Internet", ["Fiber optic", "DSL", "No"])
            phone = b.selectbox("Phone service", ["Yes", "No"])
            multi = c.selectbox("Multiple lines", ["No", "Yes"])
            a, b, c = st.columns(3)
            sec = a.selectbox("Online security", ["No", "Yes"])
            backup = b.selectbox("Online backup", ["No", "Yes"])
            devprot = c.selectbox("Device protection", ["No", "Yes"])
            a, b, c = st.columns(3)
            tech = a.selectbox("Tech support", ["No", "Yes"])
            tv = b.selectbox("Streaming TV", ["No", "Yes"])
            movies = c.selectbox("Streaming movies", ["No", "Yes"])

            st.markdown("##### 💳 Contract & billing")
            a, b = st.columns(2)
            contract = a.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            pay = b.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
            a, b, c = st.columns(3)
            paperless = a.selectbox("Paperless billing", ["Yes", "No"])
            monthly = b.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, 0.5)
            total = c.number_input("Total charges ($)", 0.0, 9000.0, 510.0, 10.0)
            go = st.form_submit_button("🔮  Predict churn risk")

    with right:
        if go:
            row = pd.DataFrame([{
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
                "InternetService": internet, "OnlineSecurity": sec, "OnlineBackup": backup,
                "DeviceProtection": devprot, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
                "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": pay,
                "MonthlyCharges": monthly, "TotalCharges": total}])
            p = float(model.predict_proba(encode(row, cols))[:, 1][0])
            label, color = risk(p)
            st.markdown(f"""<div class="card" style="text-align:center">
                {gauge(p, color)}
                <div style="margin-top:8px"><span class="badge" style="background:{color}">{label}</span></div>
                <p style="opacity:.65;font-size:.82rem;margin-top:12px">Annual revenue at stake:
                <b>${monthly*12:,.0f}</b></p></div>""", unsafe_allow_html=True)

            tips = []
            if contract == "Month-to-month": tips.append("📝 Offer a discount or free add-on for a 12-month contract.")
            if pay == "Electronic check": tips.append("💳 Give a small credit for switching to automatic payment.")
            if internet != "No" and tech == "No": tips.append("🛠️ Offer a free Tech Support / Online Security trial.")
            if internet == "Fiber optic": tips.append("📞 Proactive service-quality check-in for fiber customers.")
            if tenure <= 12: tips.append("👋 New customer - schedule onboarding calls at day 30 and day 90.")
            if p >= 0.3 and tips:
                st.markdown("##### 🎯 Recommended retention actions")
                for t in tips: st.markdown(f'<div class="action">{t}</div>', unsafe_allow_html=True)
            elif p < 0.3:
                st.success("This customer looks stable - no special action needed.")
            st.caption("Score from a class-weighted model: use it as a risk ranking (≥50% flagged), not an exact probability.")
        else:
            st.markdown("""<div class="card" style="text-align:center;padding:60px 20px">
                <div style="font-size:3rem">🔮</div><h4>Your prediction appears here</h4>
                <p style="opacity:.65">Fill in the customer details and press <b>Predict churn risk</b>.</p></div>""",
                        unsafe_allow_html=True)

# ----------------------------------------------------------------------------- Batch
with tab2:
    st.markdown("#### Score a whole customer file")
    st.write("Upload a CSV with the original Telco columns. Try it with `data/Telco_Customer_Churn_Dataset.csv`.")
    up = st.file_uploader("Drop CSV here", type="csv")
    if up:
        raw = pd.read_csv(up)
        need = {"gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", *ADDONS,
                "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"}
        missing = need - set(raw.columns)
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            res = raw.copy()
            res["churn_score"] = model.predict_proba(encode(raw, cols))[:, 1].round(3)
            res["risk"] = pd.cut(res.churn_score, [-1, .3, .5, 2], labels=["Low", "Medium", "High"])
            hi = res[res.risk == "High"]
            m = st.columns(3)
            m[0].metric("Customers scored", f"{len(res):,}")
            m[1].metric("High-risk", f"{len(hi):,}", f"{len(hi)/len(res):.0%} of base", delta_color="inverse")
            m[2].metric("Annual revenue at risk", f"${hi.MonthlyCharges.sum()*12:,.0f}")
            dist = res.risk.value_counts().reindex(["Low", "Medium", "High"]).reset_index()
            dist.columns = ["risk", "customers"]
            st.altair_chart(alt.Chart(dist).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X("risk:N", sort=["Low", "Medium", "High"], title=None), y=alt.Y("customers:Q", title=None),
                color=alt.Color("risk:N", legend=None, scale=alt.Scale(domain=["Low", "Medium", "High"],
                                                                       range=["#10b981", "#f59e0b", "#ef4444"]))
            ).properties(height=220), use_container_width=True)
            st.dataframe(res.sort_values("churn_score", ascending=False), use_container_width=True, height=360)
            st.download_button("⬇️ Download predictions", res.to_csv(index=False), "churn_predictions.csv", "text/csv")

# ----------------------------------------------------------------------------- Insights
with tab3:
    st.markdown("#### What drives churn in this dataset")
    base = df.Churned.mean()

    def bar(frame, x, title, sort=None):
        g = frame.groupby(x, observed=True).Churned.mean().reset_index()
        ch = alt.Chart(g).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("Churned:Q", title="Churn rate", axis=alt.Axis(format="%")),
            color=alt.Color("Churned:Q", legend=None, scale=alt.Scale(scheme="redyellowblue", reverse=True)),
            tooltip=[x, alt.Tooltip("Churned:Q", format=".1%")])
        rule = alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(strokeDash=[5, 4], color="gray").encode(y="y:Q")
        return (ch + rule).properties(height=280, title=title)

    d2 = df.copy()
    d2["Tenure band"] = pd.cut(d2.tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"])
    a, b = st.columns(2)
    a.altair_chart(bar(d2, "Contract", "Churn by contract type"), use_container_width=True)
    b.altair_chart(bar(d2, "Tenure band", "Churn by tenure"), use_container_width=True)
    a, b = st.columns(2)
    a.altair_chart(bar(d2, "PaymentMethod", "Churn by payment method"), use_container_width=True)
    b.altair_chart(bar(d2, "InternetService", "Churn by internet service"), use_container_width=True)

    fi = TABLES / "task5_permutation_importance.csv"
    if fi.exists():
        imp = pd.read_csv(fi).head(10)
        st.markdown("#### Top model drivers (permutation importance)")
        st.altair_chart(alt.Chart(imp).mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, color="#7c3aed").encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("mean_auc_drop:Q", title="Drop in AUC when shuffled"),
            tooltip=["feature", alt.Tooltip("mean_auc_drop:Q", format=".4f")]).properties(height=330),
            use_container_width=True)

# ----------------------------------------------------------------------------- About
with tab4:
    st.markdown("""
<div class="card"><h4>About this project</h4>
Customer churn analysis and prediction on the Telco dataset (7,043 customers).</div>
<div class="card"><h4>Method</h4>
Data cleaning &amp; encoding → EDA → customer segmentation → model comparison (Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting) with 5-fold CV tuning → test-set evaluation (ROC-AUC 0.847, recall 0.78,
precision 0.53) → business recommendations.</div>
<div class="card"><h4>Use responsibly</h4>
Scores rank customers by risk; they are not exact probabilities. Feature importances show association, not causation.
Validate any retention offer with an A/B test before a full roll-out.</div>
""", unsafe_allow_html=True)