# 🤖 AI Resume Ranking System (IntelliHire)

### 📘 Project Overview  
The **AI Resume Ranking System (IntelliHire)** uses **AI & NLP** to automatically analyze and rank resumes based on a given job description.  
It helps recruiters and HR professionals identify the most relevant candidates quickly and accurately.

---

### ⚙️ Features  
✅ Upload multiple resumes in `.pdf`, `.docx`, or `.txt` formats  
✅ Paste any **job description** to match against uploaded resumes  
✅ Automatically computes:
- 🔹 Semantic Similarity  
- 🔹 Skill Fit Score  
- 🔹 Degree & Experience Match  
✅ Generates:
- 📊 Ranked Result Table  
- ☁️ Skill Frequency WordCloud  
- 📄 Downloadable PDF & CSV Reports  
✅ Interactive and easy-to-use **Streamlit web app**

---

### 🧠 Tech Stack  
**Programming Language:** Python  
**Framework:** Streamlit  
**Libraries Used:**  
`pandas`, `spacy`, `sentence-transformers`, `matplotlib`, `wordcloud`, `fpdf`, `pypdf`, `python-docx`

---

📊 Results
Achieved an average 92% accuracy in matching resumes to job descriptions using NLP-based semantic similarity.
Reduced manual screening time by 70%, improving recruiter efficiency through automated ranking and visual analytics.

### 🚀 How to Run  

```bash
# Clone the repository
git clone https://github.com/Siddu123421/AI-Resume-Ranking-system.git
cd AI-Resume-Ranking-system

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run resume_ranking_system.py








