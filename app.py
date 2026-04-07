import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_HF_TOKEN = os.getenv("HF_TOKEN", "")


def classify_ticket(client: OpenAI, model_name: str, ticket_text: str) -> str:
    if not ticket_text.strip():
        return "general"

    prompt = f"""You are a STRICT classifier.

Classify the support ticket into ONLY ONE category:
- billing
- technical
- general

Rules:
- Output EXACTLY one word
- No explanation
- No sentence
- No extra text

Examples:
Ticket: I was charged twice
Answer: billing

Ticket: App is crashing
Answer: technical

Ticket: What are your pricing plans?
Answer: general

Now classify:

Ticket: {ticket_text}
Answer:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a ticket classification system. Respond only with: billing, technical, or general.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.0,
    )

    content = response.choices[0].message.content
    print("RAW RESPONSE:", response)
    if not content:
        return "general"

    category = content.strip().lower()

    if "billing" in category:
        return "billing"
    elif "technical" in category:
        return "technical"
    else:
        return "general"


@st.cache_resource(show_spinner=False)
def get_client(api_key: str, api_base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=api_base_url)


st.set_page_config(
    page_title="Ticket Router",
    page_icon="T",
    layout="wide",
)

st.markdown(
    """
<style>
.main h1 { font-weight: 700; letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: #eef2ff; color: #1f2937; font-size: 0.85rem; }
.card { padding: 1rem 1.25rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; }
.result { font-size: 1.1rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Customer Support Ticket Router")
st.write("Classify support tickets into billing, technical, or general.")

with st.sidebar:
    st.subheader("Settings")
    api_base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
    model_name = st.text_input("Model Name", value=DEFAULT_MODEL_NAME)
    token_input = st.text_input("HF Token", value="", type="password")

    if ENV_HF_TOKEN:
        st.caption("Environment token detected.")
    else:
        st.caption("No environment token found.")

    st.markdown("---")
    st.caption("Tip: set HF_TOKEN in Spaces secrets for a secure deploy.")

sample_tickets = [
    "I was charged twice for my subscription this month.",
    "The app crashes every time I try to upload a file.",
    "Do you offer annual pricing plans?",
]

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Ticket")
    selected_sample = st.selectbox("Sample tickets", ["(custom)"] + sample_tickets)
    if selected_sample == "(custom)":
        ticket_text = st.text_area("Ticket text", height=180, placeholder="Paste or type a support ticket...")
    else:
        ticket_text = st.text_area("Ticket text", value=selected_sample, height=180)

    classify_clicked = st.button("Classify", type="primary")

with col_right:
    st.subheader("Result")
    result_placeholder = st.empty()
    info_placeholder = st.empty()

if classify_clicked:
    api_key = token_input.strip() or ENV_HF_TOKEN

    if not api_key:
        st.error("HF_TOKEN is required. Add it in the sidebar or set it as an environment variable.")
    elif not ticket_text.strip():
        st.warning("Please provide a ticket to classify.")
    else:
        try:
            client = get_client(api_key=api_key, api_base_url=api_base_url)
            with st.spinner("Classifying ticket..."):
                category = classify_ticket(client, model_name, ticket_text)

            
            result_placeholder.markdown(
                f"<div class='card'><div class='badge'>Predicted</div><div class='result' style='color:black'>{category.title()}</div></div>",
                unsafe_allow_html=True,
            )
            info_placeholder.caption(f"Model: {model_name}")
        except Exception as exc:
            st.error(f"Classification failed: {exc}")

with st.expander("About this demo"):
    st.write(
        "This app mirrors the baseline inference logic and is compatible with Hugging Face Spaces. "
        "It uses the OpenAI client against a configurable API endpoint."
    )
