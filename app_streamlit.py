import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import streamlit as st
import matplotlib.pyplot as plt

from resume_extractor import extract_text_from_file
from resume_parser import parse_resume
from matching_engine import score_candidates
from db import get_session, Resume, JobDescription, MatchResult


st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("AI Resume Screening & Job Matching")


@st.cache_data(show_spinner=False)
def save_uploaded_file(uploaded_file) -> Path:
	data = uploaded_file.getvalue()
	path = Path("uploads") / uploaded_file.name
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(data)
	return path


def compute_and_persist_matches(jd_text: str, resume_paths: List[Path], use_sbert: bool) -> List[dict]:
	session = get_session()
	jd_struct = parse_resume(jd_text)
	jd = JobDescription(title=jd_struct.get("name") or "Job Description", content=jd_text, parsed_json=json.dumps(jd_struct))
	session.add(jd)
	session.commit()

	res_texts = []
	res_structs = []
	res_models = []
	for rp in resume_paths:
		text = extract_text_from_file(str(rp))
		res_texts.append(text)
		parsed = parse_resume(text)
		res_structs.append(parsed)
		res_model = Resume(filename=rp.name, content=text, parsed_json=json.dumps(parsed))
		session.add(res_model)
		session.commit()
		res_models.append(res_model)

	results = score_candidates(
		jd_text=jd_text,
		resume_texts=res_texts,
		jd_struct=jd_struct,
		resume_structs=res_structs,
		use_sbert=use_sbert,
	)

	display_rows = []
	for res_model, scores in zip(res_models, results):
		row = {
			"resume": res_model.filename,
			"tfidf_similarity": scores.tfidf_similarity,
			"sbert_similarity": scores.sbert_similarity,
			"skills_match": scores.skills_match,
			"experience_match": scores.experience_match,
			"education_match": scores.education_match,
			"overall_match": scores.overall_match,
		}
		display_rows.append(row)

		mr = MatchResult(
			jd_id=jd.id,
			resume_id=res_model.id,
			tfidf_similarity=scores.tfidf_similarity,
			sbert_similarity=scores.sbert_similarity,
			skills_match=scores.skills_match,
			experience_match=scores.experience_match,
			education_match=scores.education_match,
			overall_match=scores.overall_match,
		)
		session.add(mr)
		session.commit()

	def sort_key(r):
		primary = r["sbert_similarity"] if r["sbert_similarity"] is not None else r["tfidf_similarity"]
		secondary = r["overall_match"] if r["overall_match"] is not None else 0.0
		return (primary if primary is not None else -1.0, secondary)

	display_rows.sort(key=sort_key, reverse=True)
	return display_rows

def render_matcher_tab():
	with st.sidebar:
		st.header("Upload Inputs")
		jd_file = st.file_uploader("Upload Job Description (TXT/DOCX/PDF)", type=["txt", "docx", "pdf"], accept_multiple_files=False)
		resume_files = st.file_uploader("Upload Resumes (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
		use_sbert = st.toggle("Use Sentence Transformers (BERT)", value=False, key="matcher_sbert")
		start = st.button("Compute Matches", type="primary")

	if start:
		if not jd_file or not resume_files:
			st.warning("Please upload a job description and at least one resume.")
			st.stop()

		jd_path = save_uploaded_file(jd_file)
		jd_text = extract_text_from_file(str(jd_path))
		res_paths = [save_uploaded_file(f) for f in resume_files]

		with st.spinner("Scoring candidates..."):
			ranking = compute_and_persist_matches(jd_text, res_paths, use_sbert)

		st.subheader("Results")
		st.dataframe(ranking, use_container_width=True)

		# Missing skills suggestion: JD skills minus resume skills
		jd_struct = parse_resume(jd_text)
		jd_skills = set([s.lower() for s in jd_struct.get("skills", [])])

		st.subheader("Missing Skills by Candidate")
		for rp in res_paths:
			t = extract_text_from_file(str(rp))
			res_struct = parse_resume(t)
			res_skills = set([s.lower() for s in res_struct.get("skills", [])])
			missing = sorted(list(jd_skills - res_skills))
			st.write(f"{rp.name}: {', '.join(missing) if missing else 'No obvious missing skills'}")


def _aggregate_skill_counts() -> Dict[str, int]:
	session = get_session()
	rows = session.query(Resume).all()
	counts: Dict[str, int] = {}
	for r in rows:
		parsed = json.loads(r.parsed_json) if r.parsed_json else {}
		for s in parsed.get("skills", []) or []:
			key = str(s).lower()
			counts[key] = counts.get(key, 0) + 1
	return counts


def _compute_average_scores() -> Dict[str, float]:
	session = get_session()
	rows = session.query(MatchResult).all()
	if not rows:
		return {"tfidf": 0.0, "sbert": 0.0, "overall": 0.0}
	def avg(vals):
		vals = [v for v in vals if v is not None]
		return (sum(vals) / len(vals)) if vals else 0.0
	return {
		"tfidf": avg([r.tfidf_similarity for r in rows]),
		"sbert": avg([r.sbert_similarity for r in rows]),
		"overall": avg([r.overall_match for r in rows]),
	}


def render_dashboard_tab():
	st.subheader("Skill Distribution (Resumes)")
	counts = _aggregate_skill_counts()
	if not counts:
		st.info("No resumes found in database yet.")
	else:
		# Show top 15 as pie chart
		top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
		labels = [k for k, _ in top]
		values = [v for _, v in top]
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
		ax.axis('equal')
		st.pyplot(fig, use_container_width=False)

	st.subheader("Average Scores")
	avg_scores = _compute_average_scores()
	col1, col2, col3 = st.columns(3)
	col1.metric("Avg TF-IDF", f"{avg_scores['tfidf']:.3f}")
	col2.metric("Avg SBERT", f"{avg_scores['sbert']:.3f}")
	col3.metric("Avg Overall", f"{avg_scores['overall']:.3f}")


def _generate_feedback(jd_struct: dict, res_struct: dict) -> str:
	jd_sk = set([s.lower() for s in (jd_struct.get("skills") or [])])
	res_sk = set([s.lower() for s in (res_struct.get("skills") or [])])
	missing_sk = sorted(list(jd_sk - res_sk))
	if missing_sk:
		return f"Candidate is missing: {', '.join(missing_sk)}"
	return "Candidate covers the key listed skills."


def render_recommendations_tab():
	session = get_session()
	resumes = session.query(Resume).order_by(Resume.id.desc()).all()
	jds = session.query(JobDescription).order_by(JobDescription.id.desc()).all()
	if not resumes or not jds:
		st.info("Need stored resumes and job descriptions to generate recommendations.")
		return

	res_options = {f"{r.id} - {r.filename}": r for r in resumes}
	choice = st.selectbox("Select a Resume", list(res_options.keys()))
	use_sbert = st.toggle("Use SBERT", value=True, key="reco_sbert")
	if st.button("Recommend Jobs"):
		res = res_options[choice]
		res_text = res.content
		res_struct = json.loads(res.parsed_json) if res.parsed_json else parse_resume(res_text)
		jd_texts = [j.content for j in jds]
		jd_structs = [json.loads(j.parsed_json) if j.parsed_json else parse_resume(j.content) for j in jds]
		# Score resume against all JDs by swapping roles (query=resume)
		from matching_engine import compute_tfidf_similarity, compute_sbert_similarity
		tfidf = compute_tfidf_similarity(res_text, jd_texts)
		sbert = compute_sbert_similarity(res_text, jd_texts) if use_sbert else [None] * len(jds)
		rows = []
		for j, tf, sb, js in zip(jds, tfidf, sbert, jd_structs):
			feedback = _generate_feedback(js, res_struct)
			rows.append({
				"jd_id": j.id,
				"title": j.title,
				"tfidf_similarity": tf,
				"sbert_similarity": sb,
				"feedback": feedback,
			})
		def sort_key(r):
			primary = r["sbert_similarity"] if r["sbert_similarity"] is not None else r["tfidf_similarity"]
			return primary if primary is not None else -1.0
		rows.sort(key=sort_key, reverse=True)
		st.subheader("Recommended Jobs")
		st.dataframe(rows, use_container_width=True)


tab1, tab2, tab3 = st.tabs(["Matcher", "Dashboard", "Recommendations"])
with tab1:
	render_matcher_tab()
with tab2:
	render_dashboard_tab()
with tab3:
	render_recommendations_tab()


