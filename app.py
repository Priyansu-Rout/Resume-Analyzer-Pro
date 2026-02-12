import streamlit as st
from io import BytesIO
import PyPDF2
import docx
import re
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration and Setup


OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt"]

# Helper Functions


def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None


def extract_text_from_txt(file):
    """Extract text from a TXT file."""
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        return None


def extract_resume_text(uploaded_file):
    """Main function to extract text based on file type."""
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    else:
        st.warning("Unsupported file type.")
        return None


def analyze_resume_with_ai(resume_text, target_job_role, job_desc=None):
    """Analyze resume using local Phi3 via Ollama API."""
    jd_prompt = f"Job Description:\n{job_desc}\n\n" if job_desc else ""
    prompt = f"""
    Please analyze the following resume and provide a detailed breakdown tailored for the job role '{target_job_role}'.

    {jd_prompt}
    Resume:
    {resume_text}

    Provide the following clearly labeled sections:
    
    PERSONAL INFO:
    Name: ...
    Email: ...
    Phone: ...
    LinkedIn: ...
    Portfolio: ...

    CURRENT SKILLS:
    - Skill 1
    - Skill 2
    ...

    MISSING SKILLS FOR JOB ROLE:
    - Skill 1
    - Skill 2
    ...

    EXPERIENCE SUMMARY:
    A few sentences summarizing professional background.

    EDUCATION:
    - Degree 1
    - Degree 2
    ...

    ATS SCORE ESTIMATE:
    Between 0 and 100

    FORMATTING ISSUES:
    - Issue 1
    - Issue 2
    ...

    RECOMMENDATIONS TAILORED TO '{target_job_role}':
    - Recommendation 1
    - Recommendation 2
    ...
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "max_tokens": 3000
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            raw_output = result.get("response", "").strip()
            return raw_output
        else:
            st.error(f"Phi3 API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def parse_analysis_output(output):
    """Manually parse the output into sections."""
    sections = {}
    current_section = ""

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(":") and line.isupper():
            current_section = line[:-1].lower().replace(" ", "_")
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)

    return sections


def download_suggestions(suggestions_text):
    """Create a downloadable text file."""
    b = BytesIO(suggestions_text.encode())
    return b


def estimate_ats_score(text):
    """
    Estimate ATS score using keyword presence and formatting heuristics.
    Returns an integer between 0 and 100.
    """
    if not text:
        return 30

    score = 50  # base score

    # Keywords that indicate good ATS compatibility
    positive_keywords = [
        "objective", "summary", "skills", "experience", "education",
        "contact", "phone", "email", "linkedin", "github", "portfolio"
    ]

    # Deduct points for bad formatting signs
    negative_patterns = [
        r"[★●•■▪▫–—]",  # unusual bullet characters
        r"\b\d+/\d+/\d+\b",  # dates like MM/DD/YYYY (not ideal for parsing)
        r"\b\d{10,}\b",  # long numbers possibly unstructured
    ]

    # Add score for keywords present
    for word in positive_keywords:
        if word.lower() in text.lower():
            score += 3

    # Subtract score for problematic patterns
    for pattern in negative_patterns:
        matches = re.findall(pattern, text)
        score -= len(matches) * 2

    # Normalize between 0 and 100
    score = max(0, min(100, score))
    return int(score)


def extract_keywords(text):
    """Basic keyword extraction using splitting and filtering."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'is', 'was', 'are'}
    filtered = [w for w in words if w not in stop_words]
    return set(filtered)


def match_resume_to_jd(resume_text, job_desc):
    """Match resume to job description."""
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_desc)

    matched = resume_keywords.intersection(jd_keywords)
    unmatched = jd_keywords.difference(matched)

    match_ratio = len(matched) / len(jd_keywords) if jd_keywords else 0

    return {
        "matched_keywords": sorted(list(matched)),
        "unmatched_keywords": sorted(list(unmatched)),
        "match_percentage": round(match_ratio * 100, 2)
    }


def visualize_skill_gap(current_skills, missing_skills, job_role):
    """Plot radar chart of skill gap."""
    all_skills = list(set(current_skills + missing_skills))
    current_values = [1 if skill in current_skills else 0 for skill in all_skills]
    missing_values = [1 if skill in missing_skills else 0 for skill in all_skills]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=current_values,
        theta=all_skills,
        fill='toself',
        name='Current Skills'
    ))
    fig.add_trace(go.Scatterpolar(
        r=missing_values,
        theta=all_skills,
        fill='none',
        name=f'Required for {job_role}'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"Skill Gap Visualization: {job_role}"
    )
    return fig


# -----------------------------
# Streamlit App Layout
# -----------------------------

st.set_page_config(page_title="Resume Analyzer Pro", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📤 Upload Resume")
    uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_FILE_TYPES)

    st.markdown("---")
    st.subheader("📄 Compare With Another Resume")
    compare_file = st.file_uploader("Upload second resume (optional)", type=SUPPORTED_FILE_TYPES)

    st.markdown("---")
    st.subheader("📝 Or Paste Resume Text")
    manual_input = st.text_area("Paste resume text here:", height=200)

    st.markdown("---")
    st.subheader("🎯 Target Job Role")
    target_job_role = st.text_input("Enter job title (e.g., Data Scientist):")

    st.markdown("---")
    st.subheader("📄 Paste Job Description")
    job_description = st.text_area("Paste job description here:", height=200)

    st.info("Supported formats: PDF, DOCX, TXT")
    st.caption(f"Using local model: `{MODEL_NAME}` via Ollama")

# Main Content Area
st.title("🔍 Resume Analyzer Pro")
st.markdown("Upload your resume or paste it below to get AI-powered insights.")

# Initialize session state
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None

# Handle file upload or manual input
resume_text = None
if uploaded_file:
    with st.spinner("Extracting text from file..."):
        resume_text = extract_resume_text(uploaded_file)
elif manual_input.strip():
    resume_text = manual_input

# Process resume if available
if resume_text and target_job_role:
    with st.spinner("Analyzing resume with local Phi3 AI..."):
        analysis_result = analyze_resume_with_ai(resume_text, target_job_role, job_description)
        if analysis_result:
            parsed_data = parse_analysis_output(analysis_result)
            st.session_state.resume_data = parsed_data
elif not target_job_role:
    st.warning("Please enter a target job role to tailor the analysis.")

# Tabs for displaying results
if st.session_state.resume_data:
    data = st.session_state.resume_data

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Summary", "🛠️ Skills", "💼 Experience", "🎓 Education", "💡 Recommendations"
    ])

    # Tab 1: Summary
    with tab1:
        st.subheader("👤 Personal Information")
        personal_lines = data.get("personal_info", [])
        for line in personal_lines:
            st.write(line)

        st.subheader("📈 ATS Compatibility Score")
        ats_lines = data.get("ats_score_estimate", [])
        score = 0
        if ats_lines:
            full_text = " ".join(ats_lines)
            score_match = re.search(r"\b\d{1,3}\b", full_text)
            if score_match:
                score = min(100, max(0, int(score_match.group(0))))
            else:
                score = estimate_ats_score(resume_text) if resume_text else 50
        else:
            score = estimate_ats_score(resume_text) if resume_text else 50

        st.progress(score / 100)
        st.write(f"**Estimated Score: {score}/100**")

        # Match with job description
        if job_description:
            match_result = match_resume_to_jd(resume_text, job_description)
            st.subheader("🎯 Match with Job Description")
            st.metric(label="Keyword Match %", value=f"{match_result['match_percentage']}%")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Matched Keywords:**")
                st.write(", ".join(match_result["matched_keywords"]))
            with col2:
                st.write("**Missing Keywords:**")
                st.write(", ".join(match_result["unmatched_keywords"]))

    # Tab 2: Skills
    with tab2:
        st.subheader("🔧 Current Skills")
        skills_list = [s.replace("- ", "") for s in data.get("current_skills", [])]
        if skills_list:
            df_skills = pd.DataFrame(skills_list, columns=["Skill"])
            st.table(df_skills)
        else:
            st.write("No skills extracted.")

        st.subheader(f"🎯 Recommended Skills for '{target_job_role}'")
        missing_skills = [s.replace("- ", "") for s in data.get("missing_skills_for_job_role", [])]
        if missing_skills:
            df_missing = pd.DataFrame(missing_skills, columns=["Recommended Skill"])
            st.table(df_missing)
        else:
            st.write("No additional skills recommended.")

        # Skill Gap Visualization
        if skills_list or missing_skills:
            st.subheader("📊 Skill Gap Visualization")
            fig = visualize_skill_gap(skills_list, missing_skills, target_job_role)
            st.plotly_chart(fig)

    # Tab 3: Experience
    with tab3:
        st.subheader("💼 Work Experience Summary")
        exp_summary = "\n".join(data.get("experience_summary", []))
        st.write(exp_summary if exp_summary else "No experience summary provided.")

    # Tab 4: Education
    with tab4:
        st.subheader("🎓 Education & Qualifications")
        edu_list = [e.replace("- ", "") for e in data.get("education", [])]
        if edu_list:
            df_edu = pd.DataFrame(edu_list, columns=["Qualification"])
            st.table(df_edu)
        else:
            st.write("No education details found.")

    # Tab 5: Recommendations
    with tab5:
        st.subheader("⚠️ Formatting Issues")
        issues = data.get("formatting_issues", [])
        if issues:
            for issue in issues:
                st.warning(issue.replace("- ", ""))
        else:
            st.success("No formatting issues detected.")

        st.subheader(f"💡 Recommendations for '{target_job_role}'")
        recs = data.get("recommendations_tailored_to_target_job_role", [])
        if recs:
            for i, rec in enumerate(recs, 1):
                st.markdown(f"{i}. {rec.replace('- ', '')}")
        else:
            st.info("No specific recommendations provided.")

        # Download suggestions
        all_suggestions = "\n".join(recs + issues)
        if all_suggestions:
            b = download_suggestions(all_suggestions)
            st.download_button(
                label="📥 Download Suggestions",
                data=b,
                file_name="resume_improvements.txt",
                mime="text/plain"
            )

    # Resume Comparison Section
    if compare_file:
        with st.expander("🔄 Compare With Second Resume"):
            comp_text = extract_resume_text(compare_file)
            if comp_text:
                st.subheader("📄 First Resume ATS Score")
                st.write(f"{score}/100")

                comp_score = estimate_ats_score(comp_text)
                st.subheader("📄 Second Resume ATS Score")
                st.write(f"{comp_score}/100")

                comp_keywords = extract_keywords(comp_text)
                main_keywords = extract_keywords(resume_text)

                shared = main_keywords.intersection(comp_keywords)
                unique_main = main_keywords.difference(comp_keywords)
                unique_comp = comp_keywords.difference(main_keywords)

                st.subheader("🔍 Keyword Comparison")
                st.write(f"- Shared Keywords ({len(shared)}): {', '.join(shared)}")
                st.write(f"- Unique to First Resume ({len(unique_main)}): {', '.join(unique_main)}")
                st.write(f"- Unique to Second Resume ({len(unique_comp)}): {', '.join(unique_comp)}")

                # Bar chart comparison
                df_compare = pd.DataFrame({
                    "Resume": ["First", "Second"],
                    "ATS Score": [score, comp_score],
                    "Unique Keywords": [len(unique_main), len(unique_comp)]
                })
                fig = px.bar(df_compare, x="Resume", y=["ATS Score", "Unique Keywords"], barmode="group")
                st.plotly_chart(fig)
            else:
                st.error("Could not process second resume.")

else:
    st.info("Please upload a resume and specify the job role you're applying for.")
