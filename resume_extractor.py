import os
from pathlib import Path
from typing import Optional


def _extract_pdf_with_pypdf(pdf_path: Path) -> Optional[str]:
	try:
		from pypdf import PdfReader
	except Exception:
		try:
			from PyPDF2 import PdfReader  # type: ignore
		except Exception:
			return None

	try:
		reader = PdfReader(str(pdf_path))
		if getattr(reader, "is_encrypted", False):
			try:
				reader.decrypt("")
			except Exception:
				return None
		texts = []
		for page in reader.pages:
			texts.append(page.extract_text() or "")
		return "\n".join(texts)
	except Exception:
		return None


def _extract_pdf_with_pdfminer(pdf_path: Path) -> Optional[str]:
	try:
		from pdfminer.high_level import extract_text  # type: ignore
	except Exception:
		return None
	try:
		return extract_text(str(pdf_path))
	except Exception:
		return None


def _extract_docx_with_docx2txt(docx_path: Path) -> Optional[str]:
	try:
		import docx2txt  # type: ignore
	except Exception:
		return None
	try:
		return docx2txt.process(str(docx_path)) or ""
	except Exception:
		return None


def extract_text_from_file(file_path: str) -> str:
	"""
	Extract text from PDF/DOCX/TXT resumes using available backends.
	Order for PDF: pypdf -> pdfminer; For DOCX: docx2txt. For TXT: read.
	"""
	path = Path(file_path)
	if not path.exists():
		raise FileNotFoundError(f"File not found: {file_path}")

	ext = path.suffix.lower()
	if ext == ".pdf":
		text = _extract_pdf_with_pypdf(path)
		if not text or text.strip() == "":
			text = _extract_pdf_with_pdfminer(path)
			if not text:
				raise RuntimeError("Failed to extract text from PDF using available backends")
		return text
	elif ext == ".docx":
		text = _extract_docx_with_docx2txt(path)
		if text is None:
			raise RuntimeError("Failed to extract text from DOCX with docx2txt")
		return text
	elif ext in {".txt", ".md"}:
		return path.read_text(encoding="utf-8", errors="ignore")
	else:
		raise ValueError(f"Unsupported file type: {ext}")


