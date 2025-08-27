from __future__ import annotations

import json
from pathlib import Path
from typing import List

from flask import Flask, jsonify, request
from flask_cors import CORS

from db import get_session, Resume, JobDescription, MatchResult
from resume_extractor import extract_text_from_file
from resume_parser import parse_resume
from matching_engine import score_candidates


app = Flask(__name__)
CORS(app)


@app.post("/upload/resume")
def upload_resume():
	if "file" not in request.files:
		return jsonify({"error": "No file uploaded"}), 400
	file = request.files["file"]
	if not file.filename:
		return jsonify({"error": "Empty filename"}), 400

	path = Path("uploads") / file.filename
	path.parent.mkdir(parents=True, exist_ok=True)
	file.save(str(path))

	text = extract_text_from_file(str(path))
	parsed = parse_resume(text)
	session = get_session()
	res = Resume(filename=file.filename, content=text, parsed_json=json.dumps(parsed))
	session.add(res)
	session.commit()
	return jsonify({"id": res.id, "filename": res.filename, "parsed": parsed})


@app.post("/upload/jd")
def upload_jd():
	if "file" not in request.files and not request.form.get("text"):
		return jsonify({"error": "Provide a file or 'text' field"}), 400

	session = get_session()
	if "file" in request.files:
		file = request.files["file"]
		path = Path("uploads") / file.filename
		path.parent.mkdir(parents=True, exist_ok=True)
		file.save(str(path))
		text = extract_text_from_file(str(path))
		title = Path(file.filename).stem
	else:
		text = request.form.get("text", "")
		title = request.form.get("title", "Job Description")

	parsed = parse_resume(text)
	jd = JobDescription(title=title, content=text, parsed_json=json.dumps(parsed))
	session.add(jd)
	session.commit()
	return jsonify({"id": jd.id, "title": jd.title, "parsed": parsed})


@app.get("/resumes")
def list_resumes():
	session = get_session()
	rows = session.query(Resume).order_by(Resume.id.desc()).all()
	return jsonify([
		{"id": r.id, "filename": r.filename}
		for r in rows
	])


@app.get("/jds")
def list_jds():
	session = get_session()
	rows = session.query(JobDescription).order_by(JobDescription.id.desc()).all()
	return jsonify([
		{"id": j.id, "title": j.title}
		for j in rows
	])


@app.post("/rank")
def rank_resumes():
	data = request.get_json(silent=True) or {}
	jd_id = data.get("jd_id")
	resume_ids: List[int] = data.get("resume_ids", [])
	use_sbert = bool(data.get("use_sbert", False))

	session = get_session()
	jd = session.get(JobDescription, jd_id)
	if not jd:
		return jsonify({"error": "JD not found"}), 404

	resumes = [session.get(Resume, rid) for rid in resume_ids]
	resumes = [r for r in resumes if r]
	if not resumes:
		return jsonify({"error": "No valid resumes found"}), 400

	res_texts = [r.content for r in resumes]
	res_structs = [json.loads(r.parsed_json) if r.parsed_json else parse_resume(r.content) for r in resumes]
	jd_struct = json.loads(jd.parsed_json) if jd.parsed_json else parse_resume(jd.content)

	results = score_candidates(
		jd_text=jd.content,
		resume_texts=res_texts,
		jd_struct=jd_struct,
		resume_structs=res_structs,
		use_sbert=use_sbert,
	)

	rows = []
	for r, s in zip(resumes, results):
		mr = MatchResult(
			jd_id=jd.id,
			resume_id=r.id,
			tfidf_similarity=s.tfidf_similarity,
			sbert_similarity=s.sbert_similarity,
			skills_match=s.skills_match,
			experience_match=s.experience_match,
			education_match=s.education_match,
			overall_match=s.overall_match,
		)
		session.add(mr)
		session.commit()
		rows.append({
			"resume_id": r.id,
			"filename": r.filename,
			"tfidf_similarity": s.tfidf_similarity,
			"sbert_similarity": s.sbert_similarity,
			"skills_match": s.skills_match,
			"experience_match": s.experience_match,
			"education_match": s.education_match,
			"overall_match": s.overall_match,
		})

	def sort_key(r):
		primary = r["sbert_similarity"] if r["sbert_similarity"] is not None else r["tfidf_similarity"]
		secondary = r["overall_match"] if r["overall_match"] is not None else 0.0
		return (primary if primary is not None else -1.0, secondary)

	rows.sort(key=sort_key, reverse=True)
	return jsonify(rows)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8000, debug=True)


