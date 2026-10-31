import streamlit as st
import os
import sys
from pathlib import Path

from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from utils.input_options import (
    EDUCATION_LEVEL_OPTIONS,
    EMPLOYMENT_STATUS_OPTIONS,
    GENDER_OPTIONS,
    GRADE_SUBGRADE_OPTIONS,
    LOAN_PURPOSE_OPTIONS,
    MARITAL_STATUS_OPTIONS,
    calculate_installment,
    collect_constraint_notes,
)
from utils.modeling import load_or_train_pipeline, predict_credit_risk
from utils.reporting import infer_decision_label, parse_report_sections

st.set_page_config(page_title="Lending Advisor", page_icon="", layout="wide")

@st.cache_resource
def load_model_pipeline():
    return load_or_train_pipeline()

def get_ml_prediction(input_data: dict):
    pipeline = load_model_pipeline()
    return predict_credit_risk(input_data, pipeline)



#  Sidebar 
st.sidebar.title(" Lending Advisor")
st.sidebar.info(
    "This page uses an agentic AI workflow to generate a structured "
    "lending assessment report with regulatory references."
)
st.sidebar.markdown("###  Agent Workflow")
st.sidebar.markdown(
    "1. **Risk Analyzer** — Evaluates borrower profile\n"
    "2. **Regulation Retriever** — Fetches relevant guidelines (RAG)\n"
    "3. **Report Generator** — Produces structured assessment"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Framework:** LangGraph")
st.sidebar.markdown("**LLM:** Llama 3.3 70B (Groq)")
st.sidebar.markdown("**RAG:** FAISS + SentenceTransformers with TF-IDF fallback")
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** Ravleen Singh, Anurag Pandey, Ansh Tomar, Himanshu Chauhan")


#  Main UI 
st.title(" Agentic Lending Decision Support")
st.markdown("### AI-powered credit assessment with regulatory compliance")
st.markdown("---")

# Load Groq API key - try Streamlit secrets first (for Cloud), then .env (for local development)
groq_key = os.getenv('GROQ_API_KEY')
if not groq_key:
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        groq_key = None

if not groq_key:
    st.error(
        "⚠️ **GROQ_API_KEY not configured**\n\n"
        "**For local development:**\n"
        "Create a `.env` file in project root:\n"
        "```\nGROQ_API_KEY=your_api_key_here\n```\n\n"
        "**For Streamlit Cloud:**\n"
        "Add secret in Streamlit Cloud settings:\n"
        "1. Go to app settings → Secrets\n"
        "2. Add: `GROQ_API_KEY=your_api_key_here`\n\n"
        "Get free API key: https://console.groq.com"
    )
    st.stop()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, key="adv_age")
    gender = st.selectbox("Gender", GENDER_OPTIONS, key="adv_gender")
    marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS, key="adv_ms")
    education_level = st.selectbox("Education Level", EDUCATION_LEVEL_OPTIONS, key="adv_edu")
    employment_status = st.selectbox("Employment Status", EMPLOYMENT_STATUS_OPTIONS, key="adv_emp")

with col2:
    st.subheader(" Financial Information")
    annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, key="adv_inc")
    monthly_income = annual_income / 12
    st.caption("Monthly income is derived automatically from annual income to stay consistent with the training data.")
    debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=3.0, value=0.15, step=0.01, format="%.2f", key="adv_dti")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, key="adv_cs")
    num_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, max_value=100, value=5, key="adv_oa")
    total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0, value=50000, key="adv_tcl")
    current_balance = st.number_input("Current Balance ($)", min_value=0, value=10000, key="adv_cb")

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader(" Loan Details")
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000, key="adv_la")
    loan_purpose = st.selectbox("Loan Purpose", LOAN_PURPOSE_OPTIONS, key="adv_lp")
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.1, format="%.2f", key="adv_ir")
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=36, step=1, key="adv_lt")
    installment = calculate_installment(loan_amount, interest_rate, loan_term)
    grade_subgrade = st.selectbox("Grade/Subgrade", GRADE_SUBGRADE_OPTIONS, key="adv_gs")

with col4:
    st.subheader(" Credit History")
    delinquency_history = st.number_input("Delinquency History", min_value=0, max_value=100, value=0, key="adv_dh")
    public_records = st.number_input("Public Records", min_value=0, max_value=100, value=0, key="adv_pr")
    num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, max_value=100, value=0, key="adv_nd")

st.markdown("---")

preview_input = {
    'age': age,
    'marital_status': marital_status,
    'employment_status': employment_status,
    'annual_income': annual_income,
    'debt_to_income_ratio': debt_to_income_ratio,
    'credit_score': credit_score,
    'loan_amount': loan_amount,
    'interest_rate': interest_rate,
    'loan_term': loan_term,
    'num_of_open_accounts': num_of_open_accounts,
    'total_credit_limit': total_credit_limit,
    'current_balance': current_balance,
    'delinquency_history': delinquency_history,
    'public_records': public_records,
    'num_of_delinquencies': num_of_delinquencies,
}
constraint_notes = collect_constraint_notes(preview_input)
if constraint_notes:
    st.warning("Assessment note: the current input includes values the model did not see often or at all during training.\n\n- " + "\n- ".join(constraint_notes))

_, col_btn, _ = st.columns([1, 1, 1])
with col_btn:
    run_button = st.button(" Generate Lending Assessment", use_container_width=True)

if run_button:
    os.environ['GROQ_API_KEY'] = groq_key

    input_data = {
        'age': age, 'gender': gender, 'marital_status': marital_status,
        'education_level': education_level, 'annual_income': annual_income,
        'monthly_income': monthly_income, 'employment_status': employment_status,
        'debt_to_income_ratio': debt_to_income_ratio, 'credit_score': credit_score,
        'loan_amount': loan_amount, 'loan_purpose': loan_purpose,
        'interest_rate': interest_rate, 'loan_term': loan_term,
        'installment': installment, 'grade_subgrade': grade_subgrade,
        'num_of_open_accounts': num_of_open_accounts, 'total_credit_limit': total_credit_limit,
        'current_balance': current_balance, 'delinquency_history': delinquency_history,
        'public_records': public_records, 'num_of_delinquencies': num_of_delinquencies
    }

    with st.spinner('Running ML model...'):
        prediction, repayment_prob, default_prob = get_ml_prediction(input_data)

    borrower_data = {**input_data,
                     'ml_prediction': prediction,
                     'repayment_probability': repayment_prob,
                     'default_probability': default_prob}

    st.markdown("---")
    st.markdown("###  ML Model Result")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if prediction == 1:
            st.success(" LOW RISK")
        else:
            st.error(" HIGH RISK")
    with col_r2:
        st.metric("Repayment Probability", f"{repayment_prob*100:.1f}%")
    with col_r3:
        st.metric("Default Probability", f"{default_prob*100:.1f}%")

    st.markdown("---")
    st.markdown("###  Running Agentic Workflow...")

    progress = st.progress(0)
    status = st.empty()

    try:
        from agent.nodes import risk_analyzer_node, regulation_retriever_node, report_generator_node
        from agent.state import AgentState

        state = {'borrower': borrower_data, 'risk_summary': None,
                 'retrieved_regulations': None, 'final_report': None, 'messages': []}

        status.info(" Step 1/3 — Risk Analyzer: Evaluating borrower profile...")
        progress.progress(15)
        state.update(risk_analyzer_node(state))

        status.info(" Step 2/3 — Regulation Retriever: Fetching relevant guidelines...")
        progress.progress(50)
        state.update(regulation_retriever_node(state))

        status.info(" Step 3/3 — Report Generator: Producing structured assessment...")
        progress.progress(80)
        state.update(report_generator_node(state))

        result = {'risk_summary': state['risk_summary'],
                  'retrieved_regulations': state['retrieved_regulations'],
                  'final_report': state['final_report']}

        progress.progress(100)
        status.success(" Assessment complete!")

        st.markdown("---")
        st.markdown("##  Lending Assessment Report")

        report_sections = parse_report_sections(result['final_report'])

        borrower_section = report_sections.get("BORROWER PROFILE & RISK ANALYSIS")
        if borrower_section:
            with st.expander(" Borrower Profile & Risk Analysis", expanded=True):
                st.markdown(borrower_section)

        decision_section = report_sections.get("LENDING DECISION")
        if decision_section:
            with st.expander(" Lending Decision", expanded=True):
                if infer_decision_label(decision_section, prediction) == "APPROVE":
                    st.success(decision_section)
                else:
                    st.error(decision_section)

        regulatory_section = report_sections.get("REGULATORY REFERENCES")
        if regulatory_section:
            with st.expander(" Regulatory References", expanded=True):
                st.markdown(regulatory_section)

        disclaimer_section = report_sections.get("LEGAL DISCLAIMER")
        if disclaimer_section:
            with st.expander(" Legal Disclaimer", expanded=False):
                st.warning(disclaimer_section)

        with st.expander(" View Retrieved Regulations (RAG Sources)", expanded=False):
            st.text(result['retrieved_regulations'])

    except Exception as e:
        progress.empty()
        status.error(f"Agent error: {str(e)}")
        st.exception(e)
