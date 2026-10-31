from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState
from agent.rag import retrieve
import os


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def _get_llm():
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError('GROQ_API_KEY is not configured.')

    return ChatGroq(
        model=os.environ.get('GROQ_MODEL', DEFAULT_GROQ_MODEL),
        temperature=0.2,
        api_key=api_key
    )


def _unique_chunks_in_order(chunks: list[str]) -> list[str]:
    unique_chunks: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        normalized = chunk.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_chunks.append(normalized)

    return unique_chunks


def risk_analyzer_node(state: AgentState) -> dict:
    b = state['borrower']

    credit_util = (b['current_balance'] / b['total_credit_limit'] * 100) if b['total_credit_limit'] > 0 else 0
    installment_to_income = (b['installment'] / b['monthly_income'] * 100) if b['monthly_income'] > 0 else 0
    loan_to_income = b['loan_amount'] / b['annual_income'] if b['annual_income'] > 0 else 0

    risk_level = 'LOW' if b['ml_prediction'] == 1 else 'HIGH'

    system_prompt = """You are a senior credit risk analyst at a financial institution.
Analyze the borrower profile and produce a concise risk summary.
Focus on key risk drivers. Be factual and professional. Do not use bullet points.
Write 3-4 sentences maximum."""

    user_prompt = f"""Borrower Profile:
- Age: {b['age']}, Employment: {b['employment_status']}, Education: {b['education_level']}
- Annual Income: ${b['annual_income']:,.0f}, Monthly Income: ${b['monthly_income']:,.0f}
- Credit Score: {b['credit_score']} | DTI Ratio: {b['debt_to_income_ratio']*100:.1f}%
- Loan Amount: ${b['loan_amount']:,.0f} | Purpose: {b['loan_purpose']} | Grade: {b['grade_subgrade']}
- Interest Rate: {b['interest_rate']:.1f}% | Term: {b['loan_term']} months | Installment: ${b['installment']:,.2f}
- Delinquency History: {b['delinquency_history']} | Public Records: {b['public_records']} | Delinquencies: {b['num_of_delinquencies']}
- Open Accounts: {b['num_of_open_accounts']} | Credit Utilization: {credit_util:.1f}%
- Installment-to-Income Ratio: {installment_to_income:.1f}% | Loan-to-Income Ratio: {loan_to_income:.2f}x
- ML Model Prediction: {risk_level} RISK | Repayment Probability: {b['repayment_probability']*100:.1f}%

Write a professional risk summary for this borrower."""

    llm = _get_llm()
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    risk_summary = response.content.strip()

    return {
        'risk_summary': risk_summary,
        'messages': [HumanMessage(content=user_prompt), response]
    }


def regulation_retriever_node(state: AgentState) -> dict:
    b = state['borrower']

    query_parts = [
        f"credit score {b['credit_score']} risk classification",
        f"debt to income ratio {b['debt_to_income_ratio']*100:.0f}% lending guidelines",
        f"delinquency history default risk standards",
        f"employment status {b['employment_status']} income verification",
        f"loan amount {b['loan_amount']} income ratio guidelines",
        f"credit utilization open accounts fair lending"
    ]

    ordered_chunks: list[str] = []
    for query in query_parts:
        result = retrieve(query, k=2)
        for chunk in result.split('\n\n---\n\n'):
            ordered_chunks.append(chunk)

    unique_chunks = _unique_chunks_in_order(ordered_chunks)
    if not unique_chunks:
        unique_chunks = _unique_chunks_in_order([retrieve("general credit underwriting regulations", k=3)])

    regulations_text = '\n\n---\n\n'.join(unique_chunks[:5])

    return {'retrieved_regulations': regulations_text}


def report_generator_node(state: AgentState) -> dict:
    b = state['borrower']
    risk_level = 'LOW RISK' if b['ml_prediction'] == 1 else 'HIGH RISK'
    decision = 'APPROVE' if b['ml_prediction'] == 1 else 'DECLINE'

    system_prompt = """You are a senior lending officer generating a formal credit assessment report.
Structure your response with exactly these four sections using these exact headers:

## BORROWER PROFILE & RISK ANALYSIS
## LENDING DECISION
## REGULATORY REFERENCES
## LEGAL DISCLAIMER

Be professional, specific, and cite the provided regulatory context. Each section should be 2-4 sentences."""

    user_prompt = f"""Generate a structured lending assessment report using the information below.

RISK SUMMARY:
{state['risk_summary']}

REGULATORY CONTEXT:
{state['retrieved_regulations']}

KEY FACTS:
- ML Model Assessment: {risk_level}
- Repayment Probability: {b['repayment_probability']*100:.1f}%
- Default Probability: {b['default_probability']*100:.1f}%
- Credit Score: {b['credit_score']}
- DTI Ratio: {b['debt_to_income_ratio']*100:.1f}%
- Loan Grade: {b['grade_subgrade']}
- Recommended Decision: {decision}

Generate the four-section report now."""

    llm = _get_llm()
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    final_report = response.content.strip()

    return {
        'final_report': final_report,
        'messages': [HumanMessage(content=user_prompt), response]
    }
