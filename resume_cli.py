import argparse
import json
from pathlib import Path

from resume_extractor import extract_text_from_file
from resume_parser import parse_resume


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract and parse resumes (PDF/DOCX/TXT)")
	parser.add_argument("--input", required=True, type=str, help="Path to resume file")
	parser.add_argument("--dump-text", action="store_true", help="Print extracted raw text")
	parser.add_argument("--output-json", type=str, help="Optional path to write parsed JSON")
	args = parser.parse_args()

	text = extract_text_from_file(args.input)
	if args.dump_text:
		print(text[:2000] + ("..." if len(text) > 2000 else ""))

	parsed = parse_resume(text)
	print(json.dumps(parsed, indent=2, ensure_ascii=False))

	if args.output_json:
		out = Path(args.output_json)
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Parsed JSON written to {out}")


if __name__ == "__main__":
	main()


