# ---------- Resume Ranking Streamlit App (Full Version) ----------
import streamlit as st
import pandas as pd, os, re
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import os

# Check if SpaCy model exists, otherwise download
try:
    from spacy.lang.en import English
nlp = English()
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ---------- Helper functions ----------
def clean_text(t):
    return re.sub(r"\s+", " ", t.replace("\x00", " ")).strip()

def read_pdf(path):
    reader = PdfReader(path)
    return " ".join([p.extract_text() or "" for p in reader.pages])

def read_docx(path):
    doc = Document(path)
    return " ".join(p.text for p in doc.paragraphs)

def read_resume(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return clean_text(read_pdf(path))
    if ext == ".docx":
        return clean_text(read_docx(path))
    if ext == ".txt":
        return clean_text(open(path, "r", errors="ignore").read())
    return ""

def extract_skills(text):
    skills = [
        "python","java","c++","sql","git","linux","docker","aws","azure","gcp",
        "pandas","numpy","scikit-learn","tensorflow","pytorch","keras","nlp",
        "opencv","power bi","tableau"
    ]
    found = [s for s in skills if re.search(rf"\b{s}\b", text.lower())]
    return list(set(found))

def degree_score(text):
    t = text.lower()
    if "phd" in t: return 1.0
    if any(x in t for x in ["m.tech","mtech","masters","m.sc","msc"]): return 0.8
    if any(x in t for x in ["b.tech","btech","b.e","be "]): return 0.6
    if "bsc" in t: return 0.4
    return 0.0

def extract_years(text):
    yrs = re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs|year)", text.lower())
    return int(max(yrs)) if yrs else 0

# ---------- Compute Scores ----------
def compute_score(job_desc, resume_text):
    job_vec = model.encode(job_desc, convert_to_tensor=True)
    res_vec = model.encode(resume_text, convert_to_tensor=True)
    sim = float(util.cos_sim(job_vec, res_vec).cpu().numpy()[0][0])
    sim = (sim + 1)/2

    job_skills = set(extract_skills(job_desc))
    res_skills = set(extract_skills(resume_text))
    skill_fit = len(job_skills & res_skills)/len(job_skills) if job_skills else 0
    deg = degree_score(resume_text)
    years = extract_years(resume_text)
    yrs_norm = min(1.0, years/8.0)

    final = 0.55*sim + 0.25*skill_fit + 0.12*yrs_norm + 0.08*deg
    return sim, skill_fit, yrs_norm, deg, final, res_skills

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume Ranking System", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Ranking System (Advanced)")

st.write("Upload a job description and resume files to get ranking, analysis, and skill insights.")

job_desc = st.text_area("Paste Job Description", height=200, placeholder="Enter or paste job description here...")

uploaded_files = st.file_uploader(
    "Upload Resume Files (.pdf, .docx, .txt)",
    accept_multiple_files=True
)

if st.button("Rank Resumes"):
    if not job_desc.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume file.")
    else:
        results = []
        all_skills = []
        for f in uploaded_files:
            with open(f.name, "wb") as out:
                out.write(f.read())
            text = read_resume(f.name)
            sim, skill_fit, yrs, deg, final, res_skills = compute_score(job_desc, text)
            all_skills.extend(res_skills)
            results.append({
                "File": f.name,
                "Semantic_Similarity": round(sim,4),
                "Skill_Fit": round(skill_fit,4),
                "Years_of_Experience": round(yrs*8,1),
                "Degree_Score": round(deg,3),
                "Final_Score": round(final,4),
                "Matched_Skills": ", ".join(sorted(res_skills))
            })

        df = pd.DataFrame(results).sort_values("Final_Score", ascending=False).reset_index(drop=True)
        st.success("Ranking Completed ✅")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", data=csv, file_name="ranked_resumes.csv", mime="text/csv")

        # ---------- WordCloud ----------
        if all_skills:
            st.subheader("🧠 Skill Frequency WordCloud")
            text_wc = " ".join(all_skills)
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_wc)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
