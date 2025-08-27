import argparse
import re
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

try:
	from pypdf import PdfReader
except ImportError:  # fallback name
	from PyPDF2 import PdfReader  # type: ignore


def read_pdf_text(pdf_path: Path) -> str:
	"""
	Extract raw text from a PDF using pypdf.
	"""
	reader = PdfReader(str(pdf_path))
	if getattr(reader, "is_encrypted", False):
		try:
			reader.decrypt("")
		except Exception:
			raise RuntimeError(f"PDF appears to be encrypted and cannot be decrypted: {pdf_path}")

	pages_text: List[str] = []
	for page in reader.pages:
		page_text = page.extract_text() or ""
		pages_text.append(page_text)
	return "\n".join(pages_text)


def ensure_nltk_resources() -> None:
	"""
	Ensure required NLTK resources are available for tokenization and stopwords.
	"""
	import nltk
	try:
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		nltk.download("punkt", quiet=True)
	try:
		nltk.data.find("corpora/stopwords")
	except LookupError:
		nltk.download("stopwords", quiet=True)


def clean_and_tokenize(text: str) -> List[str]:
	"""
	Lowercase, remove URLs/emails/non-letters, normalize whitespace, tokenize, remove stopwords.
	Returns tokens.
	"""
	ensure_nltk_resources()
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize

	text = text.lower()
	# Remove URLs and emails
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b", " ", text)
	# Keep letters and basic separators only
	text = re.sub(r"[^a-z\s]", " ", text)
	# Normalize whitespace
	text = re.sub(r"\s+", " ", text).strip()

	stop_words = set(stopwords.words("english"))
	# Add common resume stopwords
	stop_words.update({"resume", "curriculum", "vitae", "cv", "page"})

	tokens = word_tokenize(text)
	filtered = [tok for tok in tokens if tok not in stop_words and len(tok) > 1]
	return filtered


def tokens_to_text(tokens: List[str]) -> str:
	return " ".join(tokens)


def compute_tfidf(docs: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, List[List[Tuple[int, float]]]]:
	"""
	Compute TF-IDF for a list of documents. Returns the vectorizer and a sparse-like list
	of lists [(feature_index, value)].
	"""
	vectorizer = TfidfVectorizer(max_features=max_features)
	matrix = vectorizer.fit_transform(docs)
	# Convert each row to (col_index, value) pairs for light display
	rows: List[List[Tuple[int, float]]]= []
	for i in range(matrix.shape[0]):
		row = matrix.getrow(i)
		indices = row.indices.tolist()
		values = row.data.tolist()
		rows.append(list(zip(indices, values)))
	return vectorizer, rows


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract and clean text from a PDF; optional TF-IDF.")
	parser.add_argument("--input", type=str, required=True, help="Path to input PDF file")
	parser.add_argument("--output", type=str, required=False, help="Optional path to write cleaned text")
	parser.add_argument("--show-tfidf", action="store_true", help="Compute and display top TF-IDF terms")
	parser.add_argument("--top-k", type=int, default=20, help="Top K TF-IDF terms to display when enabled")
	args = parser.parse_args()

	pdf_path = Path(args.input)
	if not pdf_path.exists():
		raise FileNotFoundError(f"Input file not found: {pdf_path}")

	raw_text = read_pdf_text(pdf_path)
	tokens = clean_and_tokenize(raw_text)
	clean_text = tokens_to_text(tokens)

	if args.output:
		out_path = Path(args.output)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(clean_text, encoding="utf-8")
		print(f"Clean text written to: {out_path}")
	else:
		print(clean_text[:1000] + ("..." if len(clean_text) > 1000 else ""))

	if args.show_tfidf:
		vectorizer, rows = compute_tfidf([clean_text])
		feature_names = vectorizer.get_feature_names_out()
		pairs = rows[0]
		# Sort by tf-idf score descending
		top = sorted(pairs, key=lambda p: p[1], reverse=True)[: args.top_k]
		print("\nTop TF-IDF terms:")
		for idx, val in top:
			print(f"{feature_names[idx]}\t{val:.4f}")


if __name__ == "__main__":
	main()


