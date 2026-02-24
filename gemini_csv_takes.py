import argparse
import csv
import json
import os
import sys
from pathlib import Path
from urllib import error, parse, request

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_MAX_ROWS = 80
DEFAULT_TEMPERATURE = 0.3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask Gemini for analysis/takes on a CSV file."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file.")
    parser.add_argument(
        "--api-key",
        default='AIzaSyDYb1By96FxZp5HUFljRg3JKf8Pt9PbeEI',
        help="Gemini API key. Defaults to GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"How many rows to include in the prompt sample (default: {DEFAULT_MAX_ROWS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write Gemini's response.",
    )
    return parser.parse_args()


def _maybe_update_numeric(stats_by_col, column, value):
    raw = (value or "").strip()
    if not raw:
        return
    try:
        number = float(raw)
    except ValueError:
        return

    stats = stats_by_col[column]
    stats["count"] += 1
    stats["sum"] += number
    stats["min"] = number if stats["min"] is None else min(stats["min"], number)
    stats["max"] = number if stats["max"] is None else max(stats["max"], number)


def load_csv_summary(csv_path, max_rows):
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows_preview = []
        total_rows = 0
        numeric_stats = {
            name: {"count": 0, "sum": 0.0, "min": None, "max": None}
            for name in fieldnames
        }

        for row in reader:
            total_rows += 1
            if len(rows_preview) < max_rows:
                rows_preview.append(row)
            for name in fieldnames:
                _maybe_update_numeric(numeric_stats, name, row.get(name))

    return fieldnames, total_rows, rows_preview, numeric_stats


def build_prompt(csv_path, fieldnames, total_rows, rows_preview, numeric_stats):
    numeric_lines = []
    for name, stats in numeric_stats.items():
        if stats["count"] == 0:
            continue
        mean = stats["sum"] / stats["count"]
        numeric_lines.append(
            f"- {name}: count={stats['count']}, min={stats['min']:.6g}, "
            f"max={stats['max']:.6g}, mean={mean:.6g}"
        )

    if not numeric_lines:
        numeric_section = "No numeric columns detected."
    else:
        numeric_section = "\n".join(numeric_lines)

    sampled_json = json.dumps(rows_preview, indent=2, ensure_ascii=False)
    truncated_note = ""
    if total_rows > len(rows_preview):
        truncated_note = (
            f"\nOnly the first {len(rows_preview)} rows are shown out of {total_rows} total rows."
        )

    return (
        "You are a sharp data analyst.\n"
        "Heres a dataset of human movements and behaviors in a tourists plaza.\n"
        "This dataset was collected using a computer simulation system\n\n"
        "There are 4 different places of interest in the plaza:\n"
        "1) Burj Khalifa, refered to as Burj\n"
        "2) Pisa tower, refered to as Pisa\n"
        "3) Japanese house refered to as center\n"
        "4) Eiffel tower refered to as Eiffel\n"
        "This Plaza has this Layout. Burj khalifa, Pisa Tower and Eiffel Tower make a triangle with the Japanese as the center\n\n"
        "Output the following sections, when refering to a place of interest refer to it with its original name:\n"
        "1) Key patterns and trends\n"
        "2) Surprising or non-obvious insights\n"
        "3) Actionable next steps\n"
        "Please cite concrete evidence from the sampled rows when possible.\n\n"
        f"CSV file: {csv_path.name}\n"
        f"Columns: {', '.join(fieldnames) if fieldnames else '(none)'}\n"
        f"Total rows: {total_rows}\n"
        f"Numeric profile:\n{numeric_section}\n"
        f"{truncated_note}\n\n"
        "Sample rows (JSON):\n"
        f"{sampled_json}\n"
    )


def call_gemini(api_key, model, prompt, temperature):
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{parse.quote(model)}:generateContent?key={parse.quote(api_key)}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error ({exc.code}): {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error while calling Gemini: {exc}") from exc


def extract_text(response_json):
    candidates = response_json.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    joined = "\n".join(t for t in texts if t).strip()
    return joined or None


def main():
    args = parse_args()

    if not args.api_key:
        print(
            "Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    csv_path = args.csv_path.resolve()
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1
    if csv_path.suffix.lower() != ".csv":
        print(f"Expected a .csv file, got: {csv_path}", file=sys.stderr)
        return 1
    if args.max_rows <= 0:
        print("--max-rows must be > 0", file=sys.stderr)
        return 1

    try:
        fieldnames, total_rows, rows_preview, numeric_stats = load_csv_summary(
            csv_path, args.max_rows
        )
    except Exception as exc:
        print(f"Failed reading CSV: {exc}", file=sys.stderr)
        return 1

    prompt = build_prompt(
        csv_path, fieldnames, total_rows, rows_preview, numeric_stats
    )

    try:
        response_json = call_gemini(
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    text = extract_text(response_json)
    if not text:
        print(
            "Gemini returned no text. Raw response:\n"
            + json.dumps(response_json, indent=2),
            file=sys.stderr,
        )
        return 1

    print(text)

    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved response to {args.output.resolve()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
