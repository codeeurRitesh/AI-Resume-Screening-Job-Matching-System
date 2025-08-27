import json
from pathlib import Path
from typing import List

from db import get_session, Resume, JobDescription
from resume_extractor import extract_text_from_file
from resume_parser import parse_resume


def seed_resumes(paths: List[str]) -> None:
	session = get_session()
	for p in paths:
		path = Path(p)
		if not path.exists():
			print(f"Skip missing resume: {path}")
			continue
		text = extract_text_from_file(str(path))
		parsed = parse_resume(text)
		session.add(Resume(filename=path.name, content=text, parsed_json=json.dumps(parsed)))
	session.commit()


def seed_jds(paths: List[str]) -> None:
	session = get_session()
	for p in paths:
		path = Path(p)
		if not path.exists():
			print(f"Skip missing JD: {path}")
			continue
		text = path.read_text(encoding="utf-8", errors="ignore")
		parsed = parse_resume(text)
		session.add(JobDescription(title=path.stem, content=text, parsed_json=json.dumps(parsed)))
	session.commit()


def main() -> None:
	# Place demo files under demo_resumes/ and demo_jds/
	resumes = sorted([str(p) for p in Path("demo_resumes").glob("**/*") if p.suffix.lower() in {".pdf", ".docx", ".txt"}])
	jds = sorted([str(p) for p in Path("demo_jds").glob("**/*.txt")])
	print(f"Found {len(resumes)} resumes, {len(jds)} JDs")
	seed_resumes(resumes)
	seed_jds(jds)
	print("Seeding completed.")


if __name__ == "__main__":
	main()


