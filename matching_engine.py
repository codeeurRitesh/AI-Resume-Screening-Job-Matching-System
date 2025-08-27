from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchScores:
	tfidf_similarity: Optional[float]
	sbert_similarity: Optional[float]
	skills_match: Optional[float]
	experience_match: Optional[float]
	education_match: Optional[float]
	overall_match: Optional[float]


def compute_tfidf_similarity(query_text: str, candidate_texts: List[str]) -> List[float]:
	vectorizer = TfidfVectorizer(max_features=10000)
	matrix = vectorizer.fit_transform([query_text] + candidate_texts)
	q_vec = matrix[0:1]
	c_vecs = matrix[1:]
	sims = cosine_similarity(q_vec, c_vecs)[0].tolist()
	return sims


def compute_sbert_similarity(query_text: str, candidate_texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer(model_name)
	embs = model.encode([query_text] + candidate_texts, normalize_embeddings=True)
	q = embs[0:1]
	c = embs[1:]
	sims = (q @ c.T).ravel().tolist()
	return sims


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
	set_a, set_b = set(a), set(b)
	if not set_a and not set_b:
		return 1.0
	if not set_a or not set_b:
		return 0.0
	inter = len(set_a & set_b)
	union = len(set_a | set_b)
	return inter / union if union else 0.0


def average(values: List[Optional[float]]) -> Optional[float]:
	vals = [v for v in values if v is not None]
	return sum(vals) / len(vals) if vals else None


def compute_section_scores(jd: Dict[str, object], resume: Dict[str, object]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
	# Expect keys: skills (List[str]), experience (List[str]), education (List[str])
	jd_skills = set(map(str.lower, (jd.get("skills") or [])))  # type: ignore
	res_skills = set(map(str.lower, (resume.get("skills") or [])))  # type: ignore
	skills_score = jaccard_similarity(jd_skills, res_skills)

	jd_exp_lines = [str(x).lower() for x in (jd.get("experience") or [])]  # type: ignore
	res_exp_lines = [str(x).lower() for x in (resume.get("experience") or [])]  # type: ignore
	experience_score = jaccard_similarity(jd_exp_lines, res_exp_lines)

	jd_edu_lines = [str(x).lower() for x in (jd.get("education") or [])]  # type: ignore
	res_edu_lines = [str(x).lower() for x in (resume.get("education") or [])]  # type: ignore
	education_score = jaccard_similarity(jd_edu_lines, res_edu_lines)

	return skills_score, experience_score, education_score


def compute_overall_match(skills_match: Optional[float], experience_match: Optional[float], education_match: Optional[float]) -> Optional[float]:
	components = [skills_match, experience_match, education_match]
	vals = [v for v in components if v is not None]
	if not vals:
		return None
	return sum(vals) / len(vals)


def score_candidates(
	jd_text: str,
	resume_texts: List[str],
	jd_struct: Optional[Dict[str, object]] = None,
	resume_structs: Optional[List[Dict[str, object]]] = None,
	use_sbert: bool = False,
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[MatchScores]:
	"""
	Compute similarity scores between a JD and multiple resumes with TF-IDF and optional SBERT.
	Also compute section scores if structured dicts are provided.
	"""
	tfidf_sims = compute_tfidf_similarity(jd_text, resume_texts)
	sbert_sims: Optional[List[float]] = None
	if use_sbert:
		sbert_sims = compute_sbert_similarity(jd_text, resume_texts, model_name)

	section_scores: List[Tuple[Optional[float], Optional[float], Optional[float]]] = []
	if jd_struct is not None and resume_structs is not None:
		for res_struct in resume_structs:
			section_scores.append(compute_section_scores(jd_struct, res_struct))
	else:
		section_scores = [(None, None, None) for _ in resume_texts]

	results: List[MatchScores] = []
	for idx in range(len(resume_texts)):
		skills_match, exp_match, edu_match = section_scores[idx]
		overall = compute_overall_match(skills_match, exp_match, edu_match)
		results.append(
			MatchScores(
				tfidf_similarity=tfidf_sims[idx],
				sbert_similarity=(sbert_sims[idx] if sbert_sims is not None else None),
				skills_match=skills_match,
				experience_match=exp_match,
				education_match=edu_match,
				overall_match=overall,
			)
		)
	return results


