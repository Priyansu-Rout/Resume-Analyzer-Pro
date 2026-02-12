# Resume-Analyzer-Pro

AI-powered resume optimization tool that runs completely offline using local LLMs (Phi3 via Ollama). No API keys required.
## 📌 Overview
Resume Analyzer Pro is an intelligent, privacy-focused tool designed to help job seekers improve their resumes using AI — all processed locally without any external APIs or internet connection. Built with Python and Streamlit, it leverages the power of Ollama + Phi3 to offer real-time insights tailored to specific job roles.

---
## ✨ Features
* 🔐 Fully Offline & Private: No data leaves your machine.
* 📄 Multi-format Resume Upload: Supports PDF, DOCX, TXT.
* 🎯 Job Role Tailoring: Get feedback aligned with specific job titles.
* 📊 Visual Skill Gap Analysis: Radar chart showing current vs required skills.
* 🔄 Resume Comparison Tool: Compare two resumes side-by-side.
* 📈 ATS Score Estimation: Evaluate how well your resume passes filters.
* 📥 Downloadable Suggestions: Export improvement tips as a text file.
* 💬 Natural Language Feedback: Powered by Phi3 via Ollama.

  ---
  ## 🛠️ Tech Stack
|Component|	Technology Used|
|---------|----------------|
|Frontend/UI|	Streamlit|
|Text Extraction|	PyPDF2, python-docx|
|Local LLM|	Phi3 via Ollama|
|Visualization|	Plotly|
|Deployment|	Localhost / Streamlit Sharing|

## 🚀 Installation Guide
**Prerequisites**

* Python 3.8+
* Ollama installed and running
* Internet access only for initial setup (to install libraries)

**Steps**

1. Clone the repo:

```bash
git clone https://github.com/Priyansu-Rout/Resume-Analyzer-Pro.git
cd resume-analyzer-pro
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Pull the Phi3 model:
```bash 
ollama pull phi3
```
5. Open your browser at http://localhost:8501 to view the app.


## 🖱️ Usage Guide
1. **Upload Resume**: Choose a `PDF/DOCX/TXT` file or paste resume text manually.
2. **Specify Target Job Role**: Enter the position you're applying for.
3. **Paste Job Description (Optional)**: For better alignment and keyword matching.
4. **Analyze Resume**: Click “Analyze” to receive AI-powered insights.
5. Explore Tabs:
    * Summary
    * Skills & Skill Gap Visualization
    * Experience
    * Education
    * Recommendations
6. **Compare Resumes**: Upload a second resume for side-by-side analysis.
7. **Download Suggestions**: Save personalized improvement tips as a .txt file.

## 📸 Screenshots

![front page](<image/Screenshot 2026-02-10 203651.png>)

![compare resume](<image/Screenshot 2026-02-11 211944.png>)

![skill gap](<image/Screenshot 2026-02-11 212054.png>)

![skill gap visuaization](<image/Screenshot 2026-02-11 212110.png>)

![Recommendation](<image/Screenshot 2026-02-11 212151.png>)

## 📦 Directory Structure
```txt
resume-analyzer-pro/
├── image/                   # Screenshots and media
├── app.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
## 🙋‍♂️ Questions or Feedback?

[linkedin](https://www.linkedin.com/in/priyansu-rout-06a40834b/)
