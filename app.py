import streamlit as st
from PIL import Image
from models.model_params import accuracy, roc_auc
from utils.runtime import PROJECT_ROOT
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

st.set_page_config(page_title="Credit Risk Scoring", page_icon="", layout="wide")

@st.cache_resource
def load_model_pipeline():
    return load_or_train_pipeline()


#  Sidebar 
st.sidebar.title("ℹ About")
st.sidebar.info(
    "This application predicts credit risk and loan repayment probability "
    "using Random Forest machine learning trained on 20,000 historical loan applications."
)

st.sidebar.markdown("###  Model Performance")
try:
    img_cm = Image.open(PROJECT_ROOT / 'visualizations' / 'final_confusion_matrix.png')
    st.sidebar.image(img_cm, caption='Confusion Matrix', use_container_width=True)
    img_roc = Image.open(PROJECT_ROOT / 'visualizations' / 'final_roc_curve.png')
    st.sidebar.image(img_roc, caption='ROC Curve', use_container_width=True)
except Exception:
    st.sidebar.warning("Visualizations not found. Run train_model.py first.")

st.sidebar.markdown("###  Features")
st.sidebar.markdown(
    "- Real-time predictions\n"
    f"- {accuracy*100:.2f}% validation accuracy\n"
    "- Random Forest algorithm\n"
    "- 21 input features"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** Ravleen Singh, Anurag Pandey, Ansh Tomar, Himanshu Chauhan")
st.sidebar.markdown("**GitHub:** [View Repository](https://github.com/ravleensingh/credit-risk-scoring)")
st.sidebar.markdown("**Project:** GenAI Capstone - Milestone 1")


#  Main UI 
def main():
    st.title(" Credit Risk Scoring System")
    st.markdown("### Predict loan repayment probability using Machine Learning")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS)
        education_level = st.selectbox("Education Level", EDUCATION_LEVEL_OPTIONS)
        employment_status = st.selectbox("Employment Status", EMPLOYMENT_STATUS_OPTIONS)

    with col2:
        st.subheader(" Financial Information")
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        monthly_income = annual_income / 12
        st.caption("Monthly income is derived automatically from annual income to stay consistent with the training data.")
        debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=3.0, value=0.15, step=0.01, format="%.2f")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        num_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, max_value=100, value=5)
        total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0, value=50000)
        current_balance = st.number_input("Current Balance ($)", min_value=0, value=10000)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(" Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
        loan_purpose = st.selectbox("Loan Purpose", LOAN_PURPOSE_OPTIONS)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.1, format="%.2f")
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=36, step=1)
        installment = calculate_installment(loan_amount, interest_rate, loan_term)
        grade_subgrade = st.selectbox("Grade/Subgrade", GRADE_SUBGRADE_OPTIONS)

    with col4:
        st.subheader(" Credit History")
        delinquency_history = st.number_input("Delinquency History", min_value=0, max_value=100, value=0)
        public_records = st.number_input("Public Records", min_value=0, max_value=100, value=0)
        num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, max_value=100, value=0)

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
        st.warning("Prediction note: the current input includes values the model did not see often or at all during training.\n\n- " + "\n- ".join(constraint_notes))

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        predict_button = st.button(" Assess Credit Risk", use_container_width=True)

    if predict_button:
        input_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education_level': education_level,
            'annual_income': annual_income,
            'monthly_income': monthly_income,
            'employment_status': employment_status,
            'debt_to_income_ratio': debt_to_income_ratio,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'interest_rate': interest_rate,
            'loan_term': loan_term,
            'installment': installment,
            'grade_subgrade': grade_subgrade,
            'num_of_open_accounts': num_of_open_accounts,
            'total_credit_limit': total_credit_limit,
            'current_balance': current_balance,
            'delinquency_history': delinquency_history,
            'public_records': public_records,
            'num_of_delinquencies': num_of_delinquencies
        }

        with st.spinner('Analyzing credit risk...'):
            pipeline = load_model_pipeline()
            prediction, repayment_probability, default_probability = predict_credit_risk(input_data, pipeline)

        st.markdown("---")

        col_result1, col_result2 = st.columns(2)
        with col_result1:
            if prediction == 1:
                st.success(" LOW RISK - Loan Likely to be Paid Back")
                st.balloons()
            else:
                st.error(" HIGH RISK - Loan Default Likely")

        with col_result2:
            st.metric("Repayment Probability", f"{repayment_probability*100:.2f}%")
            st.metric("Default Probability", f"{default_probability*100:.2f}%")

    st.markdown("---")
    st.markdown("###  Model Information")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Training Samples", "20,000")
    with col_info2:
        st.metric("Algorithm", "Random Forest")
    with col_info3:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col_info4:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")


main()
