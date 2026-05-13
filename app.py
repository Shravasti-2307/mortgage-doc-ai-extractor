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
Market Trend: Stable"""
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
    height=200,
    placeholder="Paste any mortgage document here..."
)

SYSTEM_PROMPT = """You are a mortgage document AI extraction engine for Supreme Lending.
Extract structured fields and return ONLY valid JSON — no markdown, no backticks, no explanation.

Return exactly this structure:
{
  "document_type": "Loan Application | W-2 | Appraisal | Closing Disclosure | Other",
  "fields": [
    {"key": "field name", "value": "extracted value or null", "confidence": 0.95}
  ],
  "overall_confidence": 0.92,
  "validation_flags": ["any anomalies or missing critical fields"],
  "extraction_notes": "1-2 sentence summary"
}

Extract 6-10 fields most important to a mortgage underwriter.
Confidence scoring: 0.95+ if explicitly stated, 0.70-0.94 if inferred, below 0.70 if uncertain.
Return ONLY the JSON object. No other text."""

if st.button("⚡ Extract Fields", type="primary", disabled=not doc_text):
    if not api_key:
        st.error("Add your Groq API key in the sidebar.")
    else:
        with st.status("Running extraction pipeline...", expanded=True) as status:
            st.write("📥 Ingesting document (Bronze layer)...")

            try:
                client = Groq(api_key=api_key)

                st.write("🤖 Calling extraction model...")
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract fields from this mortgage document:\n\n{doc_text}"}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )

                raw = response.choices[0].message.content.strip()

                # Strip markdown fences if model adds them anyway
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                result = json.loads(raw)

                st.write("📊 Validating & writing to Gold layer...")
                status.update(label="✅ Extraction complete", state="complete")

                # Doc type + overall confidence
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

                # Extracted fields
                st.markdown("**Extracted fields — Gold layer**")
                fields = result.get("fields", [])
                for f in fields:
                    c = int(f.get("confidence", 0) * 100)
                    color = "#1D9E75" if c >= 85 else "#BA7517" if c >= 65 else "#E24B4A"
                    col_a, col_b, col_c = st.columns([3, 3, 1])
                    col_a.markdown(
                        f"<span style='font-size:13px;color:#888;'>{f['key']}</span>",
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

                # Extraction notes
                if result.get("extraction_notes"):
                    st.caption(f"📝 {result['extraction_notes']}")

                # Validation flags
                flags = result.get("validation_flags", [])
                if flags:
                    st.markdown("**⚠️ Validation flags**")
                    for flag in flags:
                        st.warning(flag)
                else:
                    st.success("✅ No validation flags — record passed all checks.")

                # Audit log
                with st.expander("🔍 Audit log — raw model output"):
                    st.json(result)

            except json.JSONDecodeError:
                status.update(label="❌ Parse error", state="error")
                st.error("Model returned invalid JSON. Raw output:")
                st.code(raw)
            except Exception as e:
                status.update(label="❌ Extraction failed", state="error")
                st.error(f"Error: {str(e)}")
```

And your `requirements.txt` should be exactly:
```
groq
streamlit
