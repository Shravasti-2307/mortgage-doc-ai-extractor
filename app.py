import streamlit as st
from groq import Groq
import json

st.set_page_config(
    page_title="Mortgage Document AI Extractor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
    <style>
    .arch-bar {
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 12px;
        color: #888;
        margin-bottom: 1rem;
    }
    .conf-high { color: #1D9E75; font-weight: 500; }
    .conf-med  { color: #BA7517; font-weight: 500; }
    .conf-low  { color: #E24B4A; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

st.title("🏠 Mortgage Document AI Extractor")
st.caption("Bronze → LLM Extraction → Gold Layer Output")

st.markdown(
    '<div class="arch-bar">'
    '📥 Bronze: raw input &nbsp;→&nbsp;'
    '🤖 LLM Extraction &nbsp;→&nbsp;'
    '📊 Gold: structured record'
    '</div>',
    unsafe_allow_html=True
)

SAMPLES = {
    "Loan Application": """UNIFORM RESIDENTIAL LOAN APPLICATION
Borrower: Sarah J. Mitchell
Property Address: 4821 Cypress Creek Lane, Austin, TX 78759
Loan Amount Requested: $485,000
Purchase Price: $540,000
Loan Purpose: Purchase
Loan Type: Conventional
Employment: Senior Software Engineer, Dell Technologies (6 years)
Annual Base Salary: $127,500
Credit Score: 742
Monthly Debt Obligations: $1,840
Down Payment: $55,000 (10.2%)
Requested Rate Type: 30-year fixed""",

    "W-2": """W-2 WAGE AND TAX STATEMENT 2024
Employer: Whole Foods Market Group Inc.
Employee: Marcus D. Thompson
Box 1 Wages: $68,400.00
Box 2 Federal tax withheld: $9,840.00
Box 3 Social security wages: $68,400.00
State: Texas""",

    "Appraisal": """UNIFORM RESIDENTIAL APPRAISAL REPORT
Property Address: 1102 Ridgewood Drive, Plano, TX 75025
Borrower: Elena Vasquez
Lender: Supreme Lending
Appraised Value: $612,000
Property Type: Single Family Residence
Year Built: 2018
Gross Living Area: 2,847 sq ft
Condition Rating: C2 (well-maintained)
Market Trend: Stable""",

    "Closing Disclosure": """CLOSING DISCLOSURE
Borrower: Jonathan Reed
Property Address: 812 Lakeview Court, Frisco, TX 75034
Loan Term: 30 years
Loan Type: FHA Fixed Rate
Loan Amount: $398,500
Interest Rate: 6.125%
Monthly Principal & Interest: $2,421.33
Estimated Cash to Close: $18,750
Closing Date: 05/12/2025
Lender: Supreme Lending
Escrow Included: Yes""",

    "Pay Stub": """EMPLOYEE PAY STATEMENT
Employee Name: Amanda Collins
Employer: Amazon Web Services
Pay Period: 04/01/2025 - 04/15/2025
Gross Pay: $5,842.00
Net Pay: $4,276.33
Federal Tax: $921.12
401(k) Deduction: $350.00
Position: Cloud Solutions Architect
Year-to-Date Gross: $58,420.00""",

    "Bank Statement": """CHASE BANK STATEMENT
Account Holder: Daniel Kim
Statement Period: March 1 - March 31, 2025
Beginning Balance: $24,810.22
Ending Balance: $31,442.91
Average Daily Balance: $27,338.15
Large Deposit Detected: $12,000 wire transfer on 03/18/2025
NSF Fees: None""",

    "Employment Verification": """VERIFICATION OF EMPLOYMENT
Employee: Rachel Thompson
Employer: Microsoft Corporation
Position: Senior Product Manager
Employment Start Date: June 14, 2019
Current Employment Status: Active Full-Time
Annual Salary: $158,000
Bonus Eligibility: 15%
Verification Date: April 28, 2025""",

    "Fraud Risk Scenario": """LOAN APPLICATION
Borrower: Michael Carter
Property Address: 9920 Willow Creek Blvd, Dallas, TX
Loan Amount Requested: $850,000
Annual Income Claimed: $420,000
Employer: Carter Consulting LLC
Employment Length: 3 months
Credit Score: 582
Down Payment: $4,000
Bank Statement Ending Balance: $1,284
Recent Large Deposit: $95,000 cash deposit
Requested Occupancy: Primary Residence"""
}

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Get yours free at console.groq.com"
)

sample = st.selectbox(
    "Load a sample document",
    ["— select —"] + list(SAMPLES.keys())
)

default_text = SAMPLES[sample] if sample != "— select —" else ""

doc_text = st.text_area(
    "Paste document text",
    value=default_text,
    height=240,
    placeholder="Paste any mortgage document here..."
)

SYSTEM_PROMPT = """You are a mortgage document AI extraction engine for Supreme Lending.
Extract structured fields and return ONLY valid JSON — no markdown, no backticks, no explanation.

Return exactly this structure:
{
  "document_type": "Loan Application | W-2 | Appraisal | Closing Disclosure | Pay Stub | Bank Statement | Employment Verification | Fraud Review | Other",
  "document_classification_confidence": 0.96,
  "fields": [
    {"key": "field name", "value": "extracted value or null", "confidence": 0.95}
  ],
  "overall_confidence": 0.92,
  "validation_flags": [
    "missing borrower income",
    "low credit score",
    "large unexplained deposit",
    "debt-to-income ratio risk",
    "employment history inconsistency"
  ],
  "extraction_notes": "1-2 sentence summary"
}

Extract 6-10 fields most important to a mortgage underwriter.
Confidence scoring:
- 0.95+ if explicitly stated
- 0.70-0.94 if inferred
- below 0.70 if uncertain

Validation flag guidance:
- Flag low credit score if credit score is below 620.
- Flag large unexplained deposit if a deposit appears unusually high relative to balances or income.
- Flag employment history risk if employment length is short or inconsistent.
- Flag missing borrower income if income is not present for income-based documents.
- Flag debt-to-income risk if monthly debt appears high relative to income.
- Flag missing property address if mortgage/property document lacks address.

Return ONLY the JSON object. No other text."""

def calculate_risk(flags):
    flags_text = " ".join(flags).lower()

    high_risk_keywords = [
        "large unexplained deposit",
        "low credit score",
        "fraud",
        "inconsistent",
        "employment history",
        "missing borrower income"
    ]

    medium_risk_keywords = [
        "missing",
        "debt-to-income",
        "uncertain",
        "inferred",
        "review"
    ]

    if any(keyword in flags_text for keyword in high_risk_keywords):
        return "High Risk"

    if any(keyword in flags_text for keyword in medium_risk_keywords):
        return "Medium Risk"

    return "Low Risk"

if st.button("⚡ Extract Fields", type="primary", disabled=not doc_text):
    if not api_key:
        st.error("Add your Groq API key in the sidebar.")
    else:
        raw = ""

        with st.status("Running extraction pipeline...", expanded=True) as status:
            st.write("📥 Ingesting document (Bronze layer)...")

            try:
                client = Groq(api_key=api_key)

                st.write("🤖 Calling extraction model...")

                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Extract fields from this mortgage document:\n\n{doc_text}"
                        }
                    ],
                    max_tokens=1200,
                    temperature=0.1
                )

                raw = response.choices[0].message.content.strip()

                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                raw = raw.strip()
                result = json.loads(raw)

                st.write("📊 Validating & writing to Gold layer...")
                status.update(label="✅ Extraction complete", state="complete")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"📄 {result.get('document_type', 'Unknown')}")

                with col2:
                    conf = result.get("overall_confidence", 0)
                    pct = int(conf * 100)
                    color = "conf-high" if pct >= 85 else "conf-med" if pct >= 65 else "conf-low"

                    st.markdown(
                        f'<p style="text-align:right;font-size:13px;color:#888;">Overall confidence</p>'
                        f'<p class="{color}" style="text-align:right;font-size:28px;">{pct}%</p>',
                        unsafe_allow_html=True
                    )

                class_conf = result.get("document_classification_confidence")
                if class_conf is not None:
                    st.caption(f"Document classification confidence: {int(class_conf * 100)}%")

                st.markdown("**Extracted fields — Gold layer**")

                fields = result.get("fields", [])

                for f in fields:
                    c = int(f.get("confidence", 0) * 100)
                    color = "#1D9E75" if c >= 85 else "#BA7517" if c >= 65 else "#E24B4A"

                    col_a, col_b, col_c = st.columns([3, 3, 1])

                    col_a.markdown(
                        f"<span style='font-size:13px;color:#888;'>{f.get('key', 'Unknown Field')}</span>",
                        unsafe_allow_html=True
                    )

                    col_b.markdown(
                        f"<span style='font-size:13px;font-weight:500;'>{f.get('value') or '—'}</span>",
                        unsafe_allow_html=True
                    )

                    col_c.markdown(
                        f"<span style='font-size:12px;color:{color};font-weight:500;'>{c}%</span>",
                        unsafe_allow_html=True
                    )

                    st.progress(c / 100)

                flags = result.get("validation_flags", [])

                st.markdown("### 🧠 Underwriting Risk Assessment")
                risk = calculate_risk(flags)
                st.metric("Risk Level", risk)

                if risk == "High Risk":
                    st.error("High-risk record detected. Human review recommended.")
                elif risk == "Medium Risk":
                    st.warning("Medium-risk record. Additional validation may be required.")
                else:
                    st.success("Low-risk record. No major validation issues detected.")

                if result.get("extraction_notes"):
                    st.caption(f"📝 {result['extraction_notes']}")

                if flags:
                    st.markdown("**⚠️ Validation flags**")
                    for flag in flags:
                        st.warning(flag)
                else:
                    st.success("✅ No validation flags — record passed all checks.")

                with st.expander("🔍 Audit log — raw model output"):
                    st.json(result)

            except json.JSONDecodeError:
                status.update(label="❌ Parse error", state="error")
                st.error("Model returned invalid JSON. Raw output:")
                st.code(raw)

            except Exception as e:
                status.update(label="❌ Extraction failed", state="error")
                st.error(f"Error: {str(e)}")
