import streamlit as st
import os
from together import Together
from utils import (run_grading_pipeline, 
                   extract_and_read_files,
                   highlight_original_sentences,
                   clean_feedback_text,
                   create_pdf_with_highlights,
                   process_data,
                   system_message,
                   system_message_moderator,
                   system_message_tiebreaker)


# ==========================
# Simple Login System
# ==========================
def login():
    st.title("🔐 FMI Reflection Journal AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = {
            "fmi": "password123",
        }
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show login page if not authenticated
if not st.session_state.logged_in:
    login()
    st.stop()

#--- Initialize the Together Client with the API key ----#
# Automatically checks Environment Variables or Streamlit secrets.toml
api_key = st.secrets.get("TOGETHER_API", None)

# Fallback to manual input if not found
if not api_key:
    api_key = st.sidebar.text_input("Enter Together API Key", type="password")

if api_key:
    client = Together(api_key=api_key)
else:
    st.warning("Please provide a Together API Key to continue.")
    st.stop()

#------- create side bar --------#
with st.sidebar:
    st.subheader("FMI Stock Challenge Reflection Journal")
    st.info("""🤖 **Grading Pipeline**

1️⃣ **Primary:** MiniMaxAI/MiniMax-M3  
2️⃣ **Moderator**: GPT-OSS-120B  
3️⃣ **Tie-Breaker**: Qwen3.7-Max  

Automatically invokes Tie-Breaker only if component scores exceed tolerance.
            """)

# Hardcoded Rubric extracted from the PDF
rubric = """
Rubric for Individual Stock Challenge Assignment
Total: 150 marks | Weightage: 15% of Final Grade

Component: Trading Notes and Decisions
- Excellent (48–60 marks): Trading notes are highly insightful, with relevant reasonings and clearly structured for all the 16 trades. All 5 financial instruments discussed with strong justification and logical strategies. Demonstrates development of trading strategies over the trading period with evidence of market analysis, TA/FA, news-based decisions.
- Good (36–47 marks): Trading notes are insightful and structured for all the 16 trades. All 5 financial instruments discussed with good analysis. Shown progression or understanding of strategy over the trading period with attempts to link strategies and tools used, though depth or consistency may be lacking.
- Acceptable (30–35 marks): Basic explanation for all 16 trades with minimal insight. Some compulsory instruments may be missing or not justified adequately. Limited progression or understanding of strategy over the trading period.
- Unacceptable (0–29 marks): Minimal discussion for all 16 trades either not clearly explained or justified. Missing all compulsory instruments. Lacks understanding of trading experience over the trading period.

Component: 2 Useful CIQ Pro Functions
- Excellent (16–20 marks): Both CIQ Pro functions are clearly explained and well-integrated into trading decisions. Demonstrates initiative in exploring advanced features.
- Good (12–15 marks): Functions are relevant and briefly discussed. Some connection to trading strategies is shown.
- Acceptable (10–11 marks): Limited relevance or explanation. Functions discussed but with minimal linkage to trading outcomes.
- Unacceptable (0–9 marks): Functions not relevant or poorly explained. No connection to trading decisions.

Component: 1 Useful InvestingNote Function
- Excellent (8–10 marks): Function is highly relevant and explained clearly with screenshots. Well-integrated into trading decisions.
- Good (6–7 marks): Function discussed with reasonable relevance. Some application to trading decisions.
- Acceptable (5 marks): Limited explanation or unclear how function helped trading.
- Unacceptable (0–4 marks): Function not relevant or missing. No screenshot or context given.

Component: End-of-Challenge Reflection Journal
- Excellent (48–60 marks): Reflection is deep, personal, and insightful. Shows clear progression in learning and self-awareness. Strong examples of mistakes, improvements, and self-directed learning.
- Good (36–47 marks): Reflective and thoughtful writing. Identifies key learnings and shows moderate progression in mindset or skill. Adequate mentions aspects of becoming a self-directed learner.
- Acceptable (30–35 marks): Basic or surface-level reflection. Mentions learning but lacks personal insight or detail. Minimum mentions aspects of becoming a self-directed learner.
- Unacceptable (0–29 marks): Incomplete reflection. Lacks connection to personal growth or learning process.
"""

group_zip = st.sidebar.file_uploader(":gray[Upload a zip file (by Tutorial Group)]", type=['zip'], help='Zip file should contain students submission by Tutorial Group')
st.sidebar.write(":grey[Data is de-identified using UUIDs prior to AI analysis. These randomized identifiers ensure privacy during cloud processing, while original identities are restored locally only during the final reporting stage.]")

# Initialize variables outside the 'if' block so they are always defined
data = []
sid_map = {}

#--- extract text in docs and add to session state---#
if group_zip is not None:
    # 1. Detect if a NEW zip file was uploaded to reset the cache
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = group_zip.name

    if st.session_state.last_uploaded_file != group_zip.name:
        st.session_state.evaluation_results = {}
        st.session_state.extracted_contents = None
        st.session_state.sid_map = None
        st.session_state.last_uploaded_file = group_zip.name

    # 2. Cache the extraction process so it doesn't re-run on button clicks
    if "extracted_contents" not in st.session_state or st.session_state.extracted_contents is None:
        with st.spinner("Extracting and de-identifying files..."):
            st.session_state.extracted_contents, st.session_state.sid_map = extract_and_read_files(group_zip)

    extracted_contents = st.session_state.extracted_contents
    sid_map = st.session_state.sid_map

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = {}

    for key in extracted_contents:
        st.subheader(f":blue[{key}]")
        
        # 3. Only call the LLM pipeline if we haven't evaluated this student yet in this session
        if key not in st.session_state.evaluation_results:
            try:
                final_dict = run_grading_pipeline(
                    client=client,
                    report_text=extracted_contents[key][1],
                    key=key,
                    system_message_primary=system_message,
                    system_message_moderator=system_message_moderator,
                    system_message_tiebreaker=system_message_tiebreaker,
                    rubric=rubric
                )
                # Save to session state to prevent re-running LLM on button clicks
                st.session_state.evaluation_results[key] = final_dict
            except Exception as e:
                st.error(f"Error generating response: {e}")

        # 4. Display results if evaluation was successful
        if key in st.session_state.evaluation_results:
            actual_dict = st.session_state.evaluation_results[key]
            feedback_raw = actual_dict.get("Feedback", "")
            report_text = extracted_contents[key][1]
            highlighted_content = highlight_original_sentences(report_text, feedback_raw)
            clean_feedback = clean_feedback_text(feedback_raw)
            
            with st.expander(":grey[*Submitted report (Highlighted)*]"):
                st.markdown(highlighted_content, unsafe_allow_html=True)
                # Generate PDF in memory
                pdf_file = create_pdf_with_highlights(highlighted_content, key)
                if pdf_file:
                    st.download_button(
                        label="📥 Download Annotated PDF",
                        data=pdf_file,
                        file_name=f"Evaluation_{key}.pdf",
                        mime="application/pdf"
                    )            
            
            st.markdown("### AI Feedback")
            
            # Show pipeline status
            status_msg = actual_dict.get("Status", "Unknown Status")
            if "Arbitrated" in status_msg:
                st.warning(status_msg)
            else:
                st.success(status_msg)
                
            st.markdown(clean_feedback)

# 5. Build the summary dataframe directly from the cached session state
if st.session_state.get("evaluation_results"):
    st.subheader(":orange[Marks Summary]")
    data = list(st.session_state.evaluation_results.values())
    df = process_data(data, sid_map)
    st.dataframe(df)