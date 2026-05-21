import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Analyzer",
    page_icon="👥",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / "src" / "best_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()
explainer = shap.TreeExplainer(model)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(data_path)
    return df

df_raw = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/diversity.png", width=80)
st.sidebar.title("HR Attrition Analyzer")
st.sidebar.markdown("Predict which employees are at risk of leaving — and why.")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🔍 Predict Employee", "📊 Fleet Risk View"])

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("👥 HR Attrition Analyzer")
    st.markdown("### Key insights from the IBM HR dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", "1,470")
    col2.metric("Attrition Rate", "16.1%", delta="-3.2% vs industry avg", delta_color="inverse")
    col3.metric("Model Recall", "74%", delta="+19% after tuning")
    col4.metric("Top Risk Factor", "OverTime")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        dept = df_raw.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        dept.columns = ['Department', 'Attrition Rate (%)']
        fig = px.bar(dept, x='Department', y='Attrition Rate (%)',
                     title='Attrition rate by department',
                     color='Attrition Rate (%)', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ot = df_raw.groupby('OverTime')['Attrition'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        ot.columns = ['OverTime', 'Attrition Rate (%)']
        fig2 = px.bar(ot, x='OverTime', y='Attrition Rate (%)',
                      title='Attrition rate by overtime',
                      color='Attrition Rate (%)', color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(df_raw, x='Attrition', y='MonthlyIncome',
                      title='Monthly income vs attrition',
                      color='Attrition',
                      color_discrete_map={'Yes': '#E24B4A', 'No': '#378ADD'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.histogram(df_raw, x='YearsAtCompany', color='Attrition',
                            barmode='overlay', nbins=20,
                            title='Years at company by attrition',
                            color_discrete_map={'Yes': '#E24B4A', 'No': '#378ADD'},
                            opacity=0.7)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — Predict Employee
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Predict Employee":
    st.title("🔍 Employee Risk Predictor")
    st.markdown("Fill in the employee details to get their attrition risk and explanation.")

    with st.form("employee_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal**")
            age = st.slider("Age", 18, 60, 30)
            marital = st.selectbox("Marital Status", [0, 1, 2],
                                   format_func=lambda x: ["Single", "Married", "Divorced"][x])
            distance = st.slider("Distance from Home (km)", 1, 29, 5)
            num_companies = st.slider("Num Companies Worked", 0, 9, 2)

        with col2:
            st.markdown("**Job**")
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            job_sat = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                   format_func=lambda x: ["Low", "Medium", "High", "Very High"][x-1])
            env_sat = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                   format_func=lambda x: ["Low", "Medium", "High", "Very High"][x-1])
            overtime = st.selectbox("OverTime", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
            dept = st.selectbox("Department", [0, 1, 2],
                                format_func=lambda x: ["HR", "R&D", "Sales"][x])

        with col3:
            st.markdown("**Compensation & Tenure**")
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            years_company = st.slider("Years at Company", 0, 40, 3)
            years_manager = st.slider("Years with Current Manager", 0, 17, 2)
            total_working = st.slider("Total Working Years", 0, 40, 5)
            business_travel = st.selectbox("Business Travel", [0, 1, 2],
                                           format_func=lambda x: ["Non-Travel", "Travel Rarely", "Travel Frequently"][x])

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            'Age': age, 'BusinessTravel': business_travel,
            'DailyRate': 800, 'Department': dept,
            'DistanceFromHome': distance, 'Education': 3,
            'EducationField': 2, 'EnvironmentSatisfaction': env_sat,
            'Gender': 0, 'HourlyRate': 65,
            'JobInvolvement': 3, 'JobLevel': job_level,
            'JobRole': 2, 'JobSatisfaction': job_sat,
            'MaritalStatus': marital, 'MonthlyIncome': monthly_income,
            'MonthlyRate': 14000, 'NumCompaniesWorked': num_companies,
            'OverTime': overtime, 'PercentSalaryHike': 14,
            'PerformanceRating': 3, 'RelationshipSatisfaction': 3,
            'StockOptionLevel': stock_option, 'TotalWorkingYears': total_working,
            'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3,
            'YearsAtCompany': years_company, 'YearsInCurrentRole': 2,
            'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': years_manager
        }])

        prob = model.predict_proba(input_data)[:, 1][0]
        risk_pct = round(prob * 100, 1)

        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            if risk_pct >= 60:
                st.error(f"🔴 HIGH RISK: {risk_pct}% chance of leaving")
                st.markdown("**Immediate action recommended.**")
            elif risk_pct >= 35:
                st.warning(f"🟡 MEDIUM RISK: {risk_pct}% chance of leaving")
                st.markdown("**Monitor this employee closely.**")
            else:
                st.success(f"🟢 LOW RISK: {risk_pct}% chance of leaving")
                st.markdown("**Employee appears stable.**")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={'text': "Attrition Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#E24B4A' if risk_pct >= 60 else '#F5A623' if risk_pct >= 35 else '#1D9E75'},
                    'steps': [
                        {'range': [0, 35], 'color': '#E1F5EE'},
                        {'range': [35, 60], 'color': '#FAEEDA'},
                        {'range': [60, 100], 'color': '#FDEAEA'}
                    ]
                }
            ))
            gauge.update_layout(height=250, margin=dict(t=40, b=0))
            st.plotly_chart(gauge, use_container_width=True)

        with col2:
            shap_vals = explainer.shap_values(input_data)[0]
            shap_df = pd.DataFrame({
                'Feature': input_data.columns,
                'SHAP Value': shap_vals
            }).reindex(pd.Series(shap_vals).abs().sort_values(ascending=False).index).head(10)

            shap_df['Color'] = shap_df['SHAP Value'].apply(
                lambda x: '#E24B4A' if x > 0 else '#378ADD'
            )

            fig = px.bar(shap_df, x='SHAP Value', y='Feature',
                         orientation='h',
                         title='Why? — Top factors for this employee',
                         color='Color',
                         color_discrete_map='identity')
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                xaxis_title='Red = pushes toward leaving | Blue = pushes toward staying'
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — Fleet Risk View
# ══════════════════════════════════════════════════════════════
elif page == "📊 Fleet Risk View":
    st.title("📊 Fleet Risk View")
    st.markdown("Attrition risk scores for all employees in the dataset.")

    from sklearn.preprocessing import LabelEncoder

    df_model = df_raw.copy()
    df_model.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
    df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})
    cat_cols = df_model.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df_model[col] = le.fit_transform(df_model[col])

    X_all = df_model.drop('Attrition', axis=1)
    probs = model.predict_proba(X_all)[:, 1]

    df_display = df_raw[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'YearsAtCompany']].copy()
    df_display['Attrition Risk (%)'] = (probs * 100).round(1)
    df_display['Risk Level'] = pd.cut(
        df_display['Attrition Risk (%)'],
        bins=[0, 35, 60, 100],
        labels=['🟢 Low', '🟡 Medium', '🔴 High']
    )
    df_display = df_display.sort_values('Attrition Risk (%)', ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk", len(df_display[df_display['Risk Level'] == '🔴 High']))
    col2.metric("🟡 Medium Risk", len(df_display[df_display['Risk Level'] == '🟡 Medium']))
    col3.metric("🟢 Low Risk", len(df_display[df_display['Risk Level'] == '🟢 Low']))

    st.divider()

    dept_filter = st.multiselect(
        "Filter by department",
        options=df_display['Department'].unique(),
        default=df_display['Department'].unique()
    )

    filtered = df_display[df_display['Department'].isin(dept_filter)]
    st.dataframe(filtered, use_container_width=True, height=500)
