import zipfile
from pathlib import Path
import shutil
import re
import pandas as pd
import streamlit as st
#from docx import Document
from pypdf import PdfReader
import mammoth
from xhtml2pdf import pisa
import io
import uuid
import ast
import json

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
    <html>
    <body>
    <h2>FMI Stock Challenge Evaluation</h2>
    <p>Student: {student_name}</p>
    {highlighted_html}
    </body>
    </html>
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
            annotation_html = f"""
            <span style="background-color: #ffff99;">{original}</span><br>
            <span style="color: green; font-style: italic;">💡 Suggestion: {suggestion}</span><br>
            """
            highlighted_report = highlighted_report.replace(original, annotation_html)
            
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
        'Feedback'
    ]

    return df[cols]

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
    "Feedback": '''Multiline feedback here. Use double quotes inside if quoting report content.'''
}

- Use double quotes (") for all dictionary keys and string values, except for the "Feedback" value.
- Enclose the "Feedback" value in triple single quotes (''') to preserve formatting and line breaks.
- Return only the dictionary and nothing else.
"""

#system_message = """
#1. Your task is to assess student written assignments for the "FMI Individual Stock Challenge (April 2026)" using a structured marking rubric.
#
#2. Follow these marking guidelines:
#    - Refer closely to the rubric and assign marks per criterion, without exceeding maximum scores.
#    - The total marks for the assignment is 150.
#    - Assign marks with appropriate variation to reflect the quality of each response—avoid giving uniform or overly rounded scores unless well justified.
#    - Maintain a high academic standard in grading and feedback.
#    - Justify all marks with clear, evidence-based reasoning.
#    - Verify if the student covered the 5 compulsory financial instruments (Large Cap, Penny Stock, REITs, Structured Warrant, DLC) and executed at least 16 trades.
#
#3. Provide detailed, constructive feedback:
#    - Structure feedback using line breaks for each major rubric criterion. Use paragraph spacing for clarity.
#    - Always refer to “the report” or “the journal” (not “the student”) in your comments.
#    - Include at least one direct quote from the report per rubric criterion that requires improvement.
#    - For each quote, suggest a revised version that improves formality, clarity, specificity, or conciseness.
#    - Follow this format for all sentence-level improvements:
#    
#        Original: "quoted sentence from the report"  
#        Suggestion: "formal, refined version of the sentence"
#
#    - Apply structured frameworks or specific financial terminology where relevant.
#    
#    Examples:
#        Original: "I bought DBS because it went up."  
#        Suggestion: "I initiated a long position in DBS Group Holdings (Large Cap) due to its strong fundamental performance and bullish technical breakout above the 50-day moving average."
#
#        Original: "I used CIQ Pro to look at stocks."  
#        Suggestion: "I utilized the 'Relative Valuation' function in CIQ Pro to compare the P/E and P/B multiples of local REITs against their historical averages, which guided my entry points."
#
#        Original: "I lost money at first but then I won."  
#        Suggestion: "Initial trading decisions resulted in capital drawdown due to a lack of risk management. However, by implementing strict stop-loss orders and diversifying into Daily Leverage Certificates (DLCs), I successfully recovered and outperformed the STI."
#
#   4. Return your response as a dictionary with the following structure:
#        {    
#            "Student Name": str,
#            "Trading Notes and Decisions (60 marks)": float, 
#            "2 Useful CIQ Pro Functions (20 marks)": float, 
#            "1 Useful InvestingNote Function (10 marks)": float, 
#            "End-of-Challenge Reflection Journal (60 marks)": float, 
#            "Feedback": '''Multiline feedback here. Use double quotes inside if quoting student content.'''        
#        }
#    - Use double quotes (") for all dictionary keys and string values, except for the "Feedback" value.
#    - Enclose the "Feedback" value in triple single quotes (''') to preserve formatting and line breaks.
#    - Return only the dictionary and nothing else.
#"""
#
#