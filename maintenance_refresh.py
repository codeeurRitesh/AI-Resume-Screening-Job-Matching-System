"""
Recompute candidate scores for all stored JDs and resumes, and print aggregate averages.
Run nightly via GitHub Actions or a cron.
"""

import json
from typing import List

from db import get_session, Resume, JobDescription, MatchResult
from matching_engine import score_candidates
from resume_parser import parse_resume


def recompute_all(use_sbert: bool = False) -> None:
    session = get_session()
    jds = session.query(JobDescription).all()
    resumes = session.query(Resume).all()
    if not jds or not resumes:
        print("No JDs or resumes; nothing to recompute.")
        return

    # Clear previous match results to avoid growth
    session.query(MatchResult).delete()
    session.commit()

    jd_structs = [json.loads(j.parsed_json) if j.parsed_json else parse_resume(j.content) for j in jds]
    res_structs = [json.loads(r.parsed_json) if r.parsed_json else parse_resume(r.content) for r in resumes]
    res_texts = [r.content for r in resumes]

    for j, js in zip(jds, jd_structs):
        results = score_candidates(
            jd_text=j.content,
            resume_texts=res_texts,
            jd_struct=js,
            resume_structs=res_structs,
            use_sbert=use_sbert,
        )
        for r, s in zip(resumes, results):
            mr = MatchResult(
                jd_id=j.id,
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

    # Aggregates
    rows = session.query(MatchResult).all()
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else 0.0
    print("Averages:")
    print("TF-IDF:", avg([r.tfidf_similarity for r in rows]))
    print("SBERT:", avg([r.sbert_similarity for r in rows]))
    print("Overall:", avg([r.overall_match for r in rows]))


if __name__ == "__main__":
    # set to True if you want SBERT nightly (costs extra time)
    recompute_all(use_sbert=False)


