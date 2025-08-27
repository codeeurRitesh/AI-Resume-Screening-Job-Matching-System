import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# Simple, extensible skills dictionary. Extend as needed.
SKILL_TERMS = {
	"python", "java", "c++", "sql", "mysql", "sqlite", "postgresql", "mongodb",
	"pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
	"nlp", "spacy", "nltk", "bert", "transformers", "flask", "django",
	"git", "docker", "kubernetes", "aws", "azure", "gcp",
}

DEGREE_TERMS = [
	"b\.?e\.?", "b\.?tech\.?", "bachelor", "bsc", "b\.sc\.",
	"m\.?e\.?", "m\.?tech\.?", "master", "msc", "m\.sc\.", "mba",
	"phd", "ph\.d\.", "doctorate", "bca", "mca",
]

EXPERIENCE_HEADERS = [
	"experience", "professional experience", "work experience", "employment history",
]

EDUCATION_HEADERS = [
	"education", "academic background", "qualifications", "academic qualifications",
]


@dataclass
class Resume:
	name: Optional[str]
	skills: List[str]
	education: List[str]
	experience: List[str]


def _normalize(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def _guess_name(lines: List[str]) -> Optional[str]:
	# Heuristic: first non-empty line with 2-4 words, mostly title-case
	for line in lines[:20]:
		cand = line.strip()
		if not cand:
			continue
		# Skip if line looks like contact or header noise
		if re.search(r"@|https?://|\b(phone|email|address|linkedin|github)\b", cand, re.I):
			continue
		parts = [p for p in re.split(r"\s+", cand) if p]
		if 2 <= len(parts) <= 4:
			title_like = sum(1 for p in parts if re.match(r"^[A-Z][a-z'\-]+$", p))
			if title_like >= max(2, len(parts) - 1):
				return cand
	return None


def _extract_section(lines: List[str], headers: List[str]) -> List[str]:
	indices: List[int] = []
	for i, line in enumerate(lines):
		low = line.lower().strip()
		if any(re.match(rf"^\s*{h}\b", low) for h in headers):
			indices.append(i)
	if not indices:
		return []
	start = indices[0] + 1
	# End at next all-caps header or blank gap
	end = len(lines)
	for j in range(start + 1, len(lines)):
		if re.match(r"^[A-Z][A-Z\s\-&/]{3,}$", lines[j].strip()):
			end = j
			break
	return [_normalize(l) for l in lines[start:end] if _normalize(l)]


def _extract_skills(text: str) -> List[str]:
	text_low = text.lower()
	found = sorted({s for s in SKILL_TERMS if re.search(rf"\b{re.escape(s)}\b", text_low)})
	return found


def _extract_education(lines: List[str]) -> List[str]:
	edu_lines = _extract_section(lines, EDUCATION_HEADERS)
	if not edu_lines:
		# fallback: scan for degrees anywhere
		pattern = re.compile(r"|".join(DEGREE_TERMS), re.I)
		return [l for l in (_normalize(x) for x in lines) if pattern.search(l)]
	return edu_lines


def _extract_experience(lines: List[str]) -> List[str]:
	exp_lines = _extract_section(lines, EXPERIENCE_HEADERS)
	if exp_lines:
		return exp_lines
	# fallback: scan for job-like bullets containing years or verbs
	job_like = []
	for l in lines:
		if re.search(r"\b(\d{4})\b", l) or re.search(r"\b(designed|built|led|managed|developed|implemented)\b", l, re.I):
			job_like.append(_normalize(l))
	return job_like[:20]


def parse_resume(raw_text: str) -> Dict[str, object]:
	lines = [l for l in (x.strip() for x in raw_text.splitlines()) if l]
	name = _guess_name(lines)
	skills = _extract_skills(raw_text)
	education = _extract_education(lines)
	experience = _extract_experience(lines)
	resume = Resume(name=name, skills=skills, education=education, experience=experience)
	return asdict(resume)


