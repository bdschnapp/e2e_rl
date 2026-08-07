#!/usr/bin/env python3
"""Turn `% REVIEW:` comments and \\pending{} markers in the LaTeX source into real
PDF annotations, so they can be read in a PDF viewer instead of in the source.

Each marker becomes a highlight over the sentence it applies to, carrying the
marker text as its comment, so it appears in the viewer's comment sidebar.
The page layout is untouched: annotations are added to a copy of the built PDF.

Usage:
    python3 scripts/annotate_review_markers.py \
        --pdf build/main_merged_test.pdf \
        --out build/main_merged_test_annotated.pdf \
        --tex chapters/04_learning_and_results.tex chapters/sections/scalable_simulation.tex

Re-run after every rebuild; the annotated copy is derived, never edited by hand.
"""
from __future__ import annotations

import argparse
import re
import sys

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight
from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject

# marker kind -> (sidebar author, highlight colour)
KINDS = {
    "CUT-CANDIDATE": ("REVIEW / cut candidate", "ff9999"),
    "PENDING": ("PENDING DATA", "ff6666"),
    "REVIEW": ("REVIEW", "ffd24d"),
}


def strip_latex(text: str) -> str:
    """Reduce a LaTeX line to the plain words that will appear in the PDF."""
    text = re.sub(r"\\(cref|Cref|ref|eqref|cite)\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}~\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def candidate_phrases(anchor: str, min_words: int = 5, max_words: int = 8) -> list[str]:
    """Word runs to look for in the PDF, longest first, so the match is specific.

    Short runs are tried last because a phrase spanning a line break in the
    rendered PDF will not match literally.
    """
    words = [w for w in strip_latex(anchor).split() if w]
    out: list[str] = []
    for n in range(max_words, min_words - 1, -1):
        for i in range(0, max(1, len(words) - n + 1)):
            run = words[i : i + n]
            if len(run) < n:
                continue
            # skip runs that are mostly punctuation or digits
            if sum(c.isalpha() for c in "".join(run)) < 12:
                continue
            out.append(" ".join(run))
    return out


def extract_markers(tex_paths: list[str]) -> list[dict]:
    """Collect (kind, note, anchor) triples from the LaTeX sources."""
    markers: list[dict] = []
    for path in tex_paths:
        lines = open(path, encoding="utf-8").read().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # ---- % REVIEW ... (may continue over several comment lines)
            if re.match(r"\s*%\s*REVIEW", line):
                note_lines = [re.sub(r"^\s*%\s*", "", line)]
                j = i + 1
                while j < len(lines) and re.match(r"\s*%", lines[j]) and not re.match(r"\s*%\s*REVIEW", lines[j]):
                    note_lines.append(re.sub(r"^\s*%\s*", "", lines[j]))
                    j += 1
                note = " ".join(note_lines).strip()
                kind = "CUT-CANDIDATE" if "CUT-CANDIDATE" in note else "REVIEW"

                # anchor: the next prose line; fall back to the previous one
                anchor = ""
                k = j
                while k < len(lines) and k < j + 6:
                    cand = lines[k].strip()
                    if cand and not cand.startswith("%") and len(strip_latex(cand).split()) >= 6:
                        anchor = cand
                        break
                    k += 1
                if not anchor:
                    for k in range(i - 1, max(-1, i - 8), -1):
                        cand = lines[k].strip()
                        if cand and not cand.startswith("%") and len(strip_latex(cand).split()) >= 6:
                            anchor = cand
                            break
                markers.append({"kind": kind, "note": note, "anchor": anchor, "src": f"{path}:{i+1}"})
                i = j
                continue

            # ---- \pending{...} markers, which flag missing data
            for m in re.finditer(r"\\pending\{([^}]*)\}", line):
                markers.append(
                    {
                        "kind": "PENDING",
                        "note": f"PENDING DATA: {m.group(1)}",
                        "anchor": line.strip(),
                        "src": f"{path}:{i+1}",
                    }
                )
            i += 1
    return markers


def body_pages(pdf_path: str) -> set[int]:
    """Indices of body pages, i.e. those with an arabic page label.

    Front matter carries roman labels, and its table of contents and lists of
    figures and tables repeat every caption verbatim, so a caption-anchored
    marker would otherwise match there first.
    """
    try:
        labels = PdfReader(pdf_path).page_labels
    except Exception:
        return set()
    return {i for i, lab in enumerate(labels) if str(lab).isdigit()}


def locate(pdf_path: str, markers: list[dict]) -> None:
    """Fill in page index and bounding box for each marker, in place."""
    body = body_pages(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        pages = [(n, p, p.extract_text() or "") for n, p in enumerate(pdf.pages)
                 if not body or n in body]
        for mk in markers:
            mk["page"] = None
            if not mk["anchor"]:
                continue
            for phrase in candidate_phrases(mk["anchor"]):
                flat = re.sub(r"\s+", " ", phrase)
                for n, page, text in pages:
                    if flat.lower() not in re.sub(r"\s+", " ", text).lower():
                        continue
                    try:
                        hits = page.search(re.escape(phrase), regex=True, case=False)
                    except Exception:
                        hits = []
                    if hits:
                        h = hits[0]
                        mk["page"] = n
                        mk["bbox"] = (h["x0"], h["top"], h["x1"], h["bottom"])
                        mk["height"] = float(page.height)
                        break
                if mk["page"] is not None:
                    break


def annotate(pdf_path: str, out_path: str, markers: list[dict]) -> tuple[int, int]:
    writer = PdfWriter(clone_from=pdf_path)

    placed = 0
    for mk in markers:
        if mk.get("page") is None:
            continue
        x0, top, x1, bottom = mk["bbox"]
        h = mk["height"]
        y0, y1 = h - bottom, h - top          # PDF coords are bottom-up
        author, colour = KINDS[mk["kind"]]

        annot = Highlight(
            rect=(x0, y0, x1, y1),
            quad_points=ArrayObject([FloatObject(v) for v in (x0, y1, x1, y1, x0, y0, x1, y0)]),
            highlight_color=colour,
        )
        annot[NameObject("/Contents")] = TextStringObject(mk["note"])
        annot[NameObject("/T")] = TextStringObject(author)
        writer.add_annotation(page_number=mk["page"], annotation=annot)
        placed += 1

    with open(out_path, "wb") as fh:
        writer.write(fh)
    return placed, len(markers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, help="built PDF to annotate (not modified)")
    ap.add_argument("--out", required=True, help="annotated copy to write")
    ap.add_argument("--tex", required=True, nargs="+", help="LaTeX sources to scan for markers")
    args = ap.parse_args()

    markers = extract_markers(args.tex)
    if not markers:
        print("no % REVIEW or \\pending markers found")
        return 1
    locate(args.pdf, markers)
    labels = list(PdfReader(args.pdf).page_labels)
    placed, total = annotate(args.pdf, args.out, markers)

    print(f"{placed}/{total} markers placed -> {args.out}\n")
    by_page = sorted((m for m in markers if m.get("page") is not None), key=lambda m: m["page"])
    for m in by_page:
        printed = labels[m["page"]] if m["page"] < len(labels) else "?"
        print(f"  pdf p{m['page']+1:>3} (printed {printed:>4})  [{m['kind']:<13}] {m['note'][:74]}")
    missed = [m for m in markers if m.get("page") is None]
    if missed:
        print("\n  NOT PLACED (anchor text not found in the PDF):")
        for m in missed:
            print(f"    {m['src']}  {m['note'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
