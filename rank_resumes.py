import argparse
import json
from pathlib import Path
from typing import List, Tuple

from resume_extractor import extract_text_from_file
from resume_parser import parse_resume
from matching_engine import score_candidates


def load_text_or_file(path_or_text: str) -> str:
	p = Path(path_or_text)
	if p.exists():
		return p.read_text(encoding="utf-8", errors="ignore")
	return path_or_text


def main() -> None:
	parser = argparse.ArgumentParser(description="Rank resumes for a given job description")
	parser.add_argument("--jd", required=True, type=str, help="Path to JD file or raw text")
	parser.add_argument("--resumes", required=True, nargs='+', help="Paths to resume files (pdf/docx/txt)")
	parser.add_argument("--use-sbert", action="store_true", help="Use sentence-transformers embeddings")
	parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model name")
	parser.add_argument("--output", type=str, help="Optional path to write ranking JSON")
	args = parser.parse_args()

	jd_text = load_text_or_file(args.jd)
	res_texts: List[str] = []
	res_structs: List[dict] = []
	for rp in args.resumes:
		text = extract_text_from_file(rp)
		res_texts.append(text)
		res_structs.append(parse_resume(text))

	jd_struct = parse_resume(jd_text)
	results = score_candidates(
		jd_text=jd_text,
		resume_texts=res_texts,
		jd_struct=jd_struct,
		resume_structs=res_structs,
		use_sbert=args.use_sbert,
		model_name=args.model,
	)

	# Build display scores and ranking
	display = []
	for path, scores in zip(args.resumes, results):
		row = {
			"resume": path,
			"tfidf_similarity": scores.tfidf_similarity,
			"sbert_similarity": scores.sbert_similarity,
			"skills_match": scores.skills_match,
			"experience_match": scores.experience_match,
			"education_match": scores.education_match,
			"overall_match": scores.overall_match,
		}
		display.append(row)

	# Default rank by SBERT if available else TF-IDF
	def sort_key(r):
		primary = r["sbert_similarity"] if r["sbert_similarity"] is not None else r["tfidf_similarity"]
		secondary = r["overall_match"] if r["overall_match"] is not None else 0.0
		return (primary if primary is not None else -1.0, secondary)

	display.sort(key=sort_key, reverse=True)
	print(json.dumps(display, indent=2, ensure_ascii=False))

	if args.output:
		out = Path(args.output)
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(json.dumps(display, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Ranking written to {out}")


if __name__ == "__main__":
	main()


