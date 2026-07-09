import zipfile
from pathlib import Path
import shutil
import re
import pandas as pd
import streamlit as st
from pypdf import PdfReader
import mammoth
from xhtml2pdf import pisa
import io
import uuid
import ast
import json

# Add this near the top of utils.py with your other configurations
MODEL_PRICING = {
    # Prices are in USD per 1,000,000 tokens. Update these to match Together AI's current rates!
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {"input": 0.20, "output": 0.60}, 
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "Qwen/Qwen3.7-Max": {"input": 1.25, "output": 3.75}
}

# --- Model Configurations ---
PRIMARY_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
MODERATOR_MODEL = "openai/gpt-oss-120b"
TIEBREAKER_MODEL = "Qwen/Qwen3.7-Max"

TOLERANCES = {
    "Trading Notes and Decisions (60 marks)": 3,
    "2 Useful CIQ Pro Functions (20 marks)": 2,
    "1 Useful InvestingNote Function (10 marks)": 1,
    "End-of-Challenge Reflection Journal (60 marks)": 3
}

def check_tolerances(primary_dict, moderator_dict):
    discrepancies = []
    for key, tol in TOLERANCES.items():
        p_score = float(primary_dict.get(key, 0))
        m_score = float(moderator_dict.get(key, 0))
        if abs(p_score - m_score) > tol:
            discrepancies.append(key)
    return discrepancies

def call_llm(client, model_id, messages):
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
        max_tokens=4000,
        top_p=0.7,
        stream=False,
    )
    return response.choices[0].message.content, response.usage

def run_grading_pipeline(client, report_text, key, system_message_primary, system_message_moderator, system_message_tiebreaker, rubric):
    total_cost = 0.0  # Initialize cost tracker
    
    # 1. Primary Marker
    with st.spinner(f"[{key}] Primary Marker evaluating..."):
        msg_primary = [
            {"role": "system", "content": system_message_primary},
            {"role": "system", "content": f"Here are the marking rubrics: {rubric}"},
            {"role": "user", "content": f"Mark the following report for student identifier: {key}"},
            {"role": "user", "content": report_text},
            {"role": "user", "content": "Mark the report with high standard and be stringent when awarding marks."}
        ]
        primary_raw, primary_usage = call_llm(client, PRIMARY_MODEL, msg_primary)
        total_cost += inference_cost(primary_usage, MODEL_PRICING[PRIMARY_MODEL]["input"], MODEL_PRICING[PRIMARY_MODEL]["output"])
        primary_dict = safe_parse_dict(primary_raw)
        
    # 2. Moderator
    with st.spinner(f"[{key}] Moderator reviewing..."):
        msg_moderator = [
            {"role": "system", "content": system_message_moderator},
            {"role": "system", "content": f"Here are the marking rubrics: {rubric}"},
            {"role": "user", "content": f"Student Identifier: {key}\n\nReport:\n{report_text}"},
            {"role": "user", "content": f"Primary Marker's Evaluation:\n{json.dumps(primary_dict, indent=2)}"},
            {"role": "user", "content": "Review the primary marker's scores and provide your finalized scores."}
        ]
        moderator_raw, mod_usage = call_llm(client, MODERATOR_MODEL, msg_moderator)
        total_cost += inference_cost(mod_usage, MODEL_PRICING[MODERATOR_MODEL]["input"], MODEL_PRICING[MODERATOR_MODEL]["output"])
        moderator_dict = safe_parse_dict(moderator_raw)
        
    # 3. Check Tolerances
    discrepancies = check_tolerances(primary_dict, moderator_dict)
    
    if not discrepancies:
        primary_dict["Status"] = "✅ Accepted (Primary & Moderator Agree)"
        primary_dict["API Cost ($)"] = round(total_cost, 5) # Add cost to dict
        return primary_dict
    else:
        # 4. Tie-breaker
        with st.spinner(f"[{key}] Tie-Breaker resolving discrepancies..."):
            msg_tiebreaker = [
                {"role": "system", "content": system_message_tiebreaker},
                {"role": "system", "content": f"Here are the marking rubrics: {rubric}"},
                {"role": "user", "content": f"Student Identifier: {key}\n\nReport:\n{report_text}"},
                {"role": "user", "content": f"Disputed Components: {', '.join(discrepancies)}"},
                {"role": "user", "content": f"Primary Marker's Evaluation:\n{json.dumps(primary_dict, indent=2)}"},
                {"role": "user", "content": f"Moderator's Evaluation:\n{json.dumps(moderator_dict, indent=2)}"},
                {"role": "user", "content": "Provide the final, definitive scores and feedback."}
            ]
            tiebreaker_raw, tb_usage = call_llm(client, TIEBREAKER_MODEL, msg_tiebreaker)
            total_cost += inference_cost(tb_usage, MODEL_PRICING[TIEBREAKER_MODEL]["input"], MODEL_PRICING[TIEBREAKER_MODEL]["output"])
            
            final_dict = safe_parse_dict(tiebreaker_raw)
            final_dict["Status"] = f"⚖️ Arbitrated (Discrepancies in: {', '.join(discrepancies)})"
            final_dict["API Cost ($)"] = round(total_cost, 5) # Add cost to dict
            return final_dict

def inference_cost(usage, input_price, output_price):
    """
    usage: response.usage from Together SDK
    prices: USD per 1M tokens
    """
    input_cost = usage.prompt_tokens / 1_000_000 * input_price
    output_cost = usage.completion_tokens / 1_000_000 * output_price
    return input_cost + output_cost

STUDENT_ID_PATTERN = re.compile(r"\b[A-Za-z]\d{8}[A-Za-z]\b")

def generate_sid():
    return f"SID_{uuid.uuid4().hex[:8]}"

def deidentify_text(text, student_name, sid, sid_map):
    """
    Replace student name and student ID with SID (case-insensitive for name)
    """
    if student_name:
        name_pattern = re.compile(re.escape(student_name), re.IGNORECASE)
        text = name_pattern.sub(sid, text)

    matches = STUDENT_ID_PATTERN.findall(text)
    for student_id in matches:
        sid_map[sid]["student_id"] = student_id
        text = text.replace(student_id, sid)

    print(f"{student_name} ({''.join(matches)}) : {sid}")
    return text

def create_pdf_with_highlights(highlighted_html, student_name):
    html_content = f"""
    <h1>FMI Stock Challenge Evaluation</h1>
    <h2>Student: {student_name}</h2>
    {highlighted_html}
    """
    
    result = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html_content), dest=result)
    
    if pisa_status.err:
        return None
    
    result.seek(0)
    return result

def clean_feedback_text(feedback_text):
    """
    Removes the Original/Suggestion blocks from the feedback text
    so the summary remains high-level.
    """
    pattern = r'Original:\s*["“].*?["”]\s*Suggestion:\s*["“].*?["”]\s*'
    cleaned_text = re.sub(pattern, "", feedback_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
    return cleaned_text

def highlight_original_sentences(report_text, feedback_text):
    """
    Finds pairs of Original/Suggestion in feedback and embeds 
    the suggestion directly into the report text.
    """
    pattern = r'Original:\s*["“](.*?)["”].*?Suggestion:\s*["“](.*?)["”]'
    matches = re.finditer(pattern, feedback_text, re.DOTALL)
    
    highlighted_report = report_text
    
    for match in matches:
        original = match.group(1).strip()
        suggestion = match.group(2).strip()
        
        if original in highlighted_report:
            # Original gets the default yellow <mark>
            # Suggestion gets a custom soft green background with dark green text
            annotation_html = (
                f'<mark>{original}</mark> '
                f'<b>💡 Suggestion:</b> '
                f'<span style="background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 4px; font-style: italic;">{suggestion}</span>'
            )
            
            # Replaces only the first occurrence to prevent overlapping highlights 
            # if the student wrote the exact same sentence multiple times.
            highlighted_report = highlighted_report.replace(original, annotation_html, 1)
            
    return highlighted_report

def safe_parse_dict(text):
    """
    Safely extracts and parses a dictionary from LLM text output, 
    handling markdown blocks, conversational filler, and JSON fallbacks.
    """
    if not text:
        raise ValueError("Empty response from LLM")

    # 1. Strip Markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        lines = text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]  
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1] 
        text = '\n'.join(lines).strip()

    # 2. Extract the dictionary block using Regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        dict_str = match.group(0)
    else:
        dict_str = text  

    # 3. Attempt to parse using ast.literal_eval
    try:
        return ast.literal_eval(dict_str)
    except (SyntaxError, ValueError) as e:
        # 4. Fallback: Sometimes LLMs output strict JSON
        try:
            return json.loads(dict_str)
        except json.JSONDecodeError:
            pass
            
        raise ValueError(f"Failed to parse dictionary. AST Error: {e}\nRaw string snippet: {dict_str[:200]}...")

def extract_and_read_files(zip_path):
    # Initialize user_id in session state if it doesn't exist yet
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"temp_extract_{uuid.uuid4().hex[:8]}"
        
    extract_folder = st.session_state.user_id

    if Path(extract_folder).exists():
        shutil.rmtree(extract_folder)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    extracted_data = {}
    sid_map = {}

    for folder in Path(extract_folder).iterdir():
        if not folder.is_dir():
            continue

        folder_name = str(folder.relative_to(extract_folder))
        cleaned_text = re.sub(r"(?i)(TF\d+|Stock|Challenge|Reflection|FMI|Individual|_|\-)", " ", folder_name)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        student_name = cleaned_text.title()

        sid = generate_sid()
        sid_map[sid] = {"student_name": student_name, "student_id": None}

        for file in folder.glob("*.*"):
            if file.suffix.lower() == ".docx":
                def ignore_images(image):
                    return {}
                result = mammoth.convert_to_html(file, convert_image=ignore_images)
                data = result.value

            elif file.suffix.lower() == ".pdf":
                data = ""
                reader = PdfReader(file)
                for page in reader.pages:
                    data += page.extract_text()
            else:
                continue

            data = deidentify_text(data, student_name, sid, sid_map)
            extracted_data[sid] = [file.suffix.lower(), data]

    return extracted_data, sid_map

def process_data(data, sid_map):
    df = pd.DataFrame(data)
    df.insert(0, "SID", df["Student Name"])

    df["Student Name"] = (
        df["Student Name"]
        .map(lambda sid: sid_map.get(sid, {}).get("student_name", sid))
        .str.upper()
    )

    df['Total'] = (
        df['Trading Notes and Decisions (60 marks)'] + 
        df['2 Useful CIQ Pro Functions (20 marks)'] + 
        df['1 Useful InvestingNote Function (10 marks)'] + 
        df['End-of-Challenge Reflection Journal (60 marks)']
    )
    
    cols = [
        'SID',
        'Student Name', 
        'Trading Notes and Decisions (60 marks)', 
        '2 Useful CIQ Pro Functions (20 marks)', 
        '1 Useful InvestingNote Function (10 marks)',
        'End-of-Challenge Reflection Journal (60 marks)',
        'Total', 
        'API Cost ($)', # <-- Added Cost Column
        'Status',       # <-- Added Status Column
        'Summary'
    ]

    # Filter cols to only include those that actually exist in the dataframe 
    # (prevents errors if a key is somehow missing)
    valid_cols = [c for c in cols if c in df.columns]
    
    return df[valid_cols]

system_message = """
You are a strict, expert grader for the "FMI Individual Stock Challenge (April 2026)". 
Your task is to evaluate the submitted written reflection journal and report based on the provided marking rubric. 
The total assignment is worth 150 marks.

IMPORTANT LANGUAGE RULE:
- Do NOT refer to the writer as "the student" in your feedback.
- Always refer to the evaluated work as "the report", "the reflection", "the journal", or "the submission".
- Focus your comments on the quality of the evidence, analysis, decisions, and learning demonstrated in the report.
- Example:
  ❌ "The student acknowledges overconfidence..."
  ✅ "The reflection demonstrates acknowledgement of overconfidence..."
  ❌ "The student used RSI and EMA..."
  ✅ "The report demonstrates the use of RSI and EMA..."

### GRADING INSTRUCTIONS:
1. Evaluate exactly 4 components and assign marks within the rubric's ranges:
   - Trading Notes and Decisions (Max 60 marks)
   - 2 Useful CIQ Pro Functions (Max 20 marks)
   - 1 Useful InvestingNote Function (Max 10 marks)
   - End-of-Challenge Reflection Journal (Max 60 marks)

2. Verify Compliance:
   - Check if the report demonstrates sufficient trades (10-16).
   - Check if the report discusses all 5 compulsory financial instruments:
        • Large Cap
        • Penny Stock
        • REITs
        • Structured Warrant
        • DLC
   - If evidence is missing, penalize accordingly in the "Trading Notes and Decisions" section.

3. Be stringent and evidence-based.
   - Do not give uniform or overly rounded scores.
   - Marks must reflect the actual quality of evidence, depth of analysis, and application of financial concepts.
   - Justify all marks based on what is demonstrated in the report.

### FEEDBACK STRUCTURE (MANDATORY):
Your feedback MUST be structured using Markdown headings for each of the 4 components:
(e.g., ### 1. Trading Notes and Decisions)

For EVERY component:
- Provide at least one direct quote from the report that requires improvement.
- Follow the quote with a refined suggestion.

Use this exact format for sentence-level improvements to improve academic tone and financial terminology:

Original: "quoted sentence from the report"
Suggestion: "formal, refined version using specific financial terminology (e.g., TA/FA, risk management, specific instrument mechanics)"

### OUTPUT FORMAT:
Return your response as a dictionary with the following structure:

{    
    "Student Name": str,
    "Trading Notes and Decisions (60 marks)": float, 
    "2 Useful CIQ Pro Functions (20 marks)": float, 
    "1 Useful InvestingNote Function (10 marks)": float, 
    "End-of-Challenge Reflection Journal (60 marks)": float, 
    "Feedback": '''Multiline feedback here. Use double quotes inside if quoting report content.''',
    "Summary": "A concise, professional summary of the evaluation in strictly less than 100 words."
}

- Use double quotes (") for all dictionary keys and string values, except for the "Feedback" value.
- Enclose the "Feedback" value in triple single quotes (''') to preserve formatting and line breaks.
- Return only the dictionary and nothing else.
"""

system_message_moderator = """
You are an expert grading moderator for the "FMI Individual Stock Challenge (April 2026)". 
Your task is to review the evaluation provided by the Primary Marker and ensure the scores are fair, strict, and aligned with the rubric.
You will receive the student's report, the rubric, and the Primary Marker's scores and feedback.

INSTRUCTIONS:
1. Independently assess the 4 components based on the report and rubric.
2. Compare your assessment with the Primary Marker's scores.
3. Output your final agreed-upon scores. If you completely agree with the Primary Marker, output their exact scores. If you disagree, output your corrected scores.
4. Provide a brief "Moderator Notes" explaining any discrepancies or confirming the accuracy of the Primary Marker.

### OUTPUT FORMAT:
Return your response as a dictionary with the following structure:
{    
    "Student Name": str,
    "Trading Notes and Decisions (60 marks)": float, 
    "2 Useful CIQ Pro Functions (20 marks)": float, 
    "1 Useful InvestingNote Function (10 marks)": float, 
    "End-of-Challenge Reflection Journal (60 marks)": float, 
    "Feedback": '''The original feedback from the Primary Marker, optionally refined if you changed the scores significantly.''',
    "Moderator Notes": "Brief explanation of your review and any score adjustments.",
    "Summary": "A concise, professional summary of the evaluation in strictly less than 100 words."
}
- Use double quotes (") for all dictionary keys and string values, except for the "Feedback" value.
- Enclose the "Feedback" value in triple single quotes (''') to preserve formatting and line breaks.
- Return only the dictionary and nothing else.
"""

system_message_tiebreaker = """
You are the final expert arbiter for the "FMI Individual Stock Challenge (April 2026)".
Two previous graders (a Primary Marker and a Moderator) have evaluated the student's report but disagreed on certain components beyond the acceptable tolerance.
You will receive the student's report, the rubric, the Primary Marker's evaluation, and the Moderator's evaluation.

INSTRUCTIONS:
1. Review the disputed components carefully based on the rubric and the report.
2. Assign the final, definitive scores for ALL 4 components.
3. Provide comprehensive, high-quality feedback that reflects your final scoring.

### OUTPUT FORMAT:
Return your response as a dictionary with the following structure:
{    
    "Student Name": str,
    "Trading Notes and Decisions (60 marks)": float, 
    "2 Useful CIQ Pro Functions (20 marks)": float, 
    "1 Useful InvestingNote Function (10 marks)": float, 
    "End-of-Challenge Reflection Journal (60 marks)": float, 
    "Feedback": '''Final comprehensive feedback.''',
    "Summary": "Summarize the evaluation into a single, balanced paragraph of approximately 200 words. Highlight the student's key strengths, major weaknesses, and the most important areas for improvement across all assessment components. Maintain a professional, constructive, and objective tone. Do not repeat detailed examples, trade names, or rubric descriptions unless they are essential to the overall evaluation. Focus on overall performance, consistency, analytical depth, evidence provided, and reflection quality. End with a concise statement summarizing what the student should prioritize to achieve a higher grade."
}
- Use double quotes (") for all dictionary keys and string values, except for the "Feedback" value.
- Enclose the "Feedback" value in triple single quotes (''') to preserve formatting and line breaks.
- Return only the dictionary and nothing else.
"""