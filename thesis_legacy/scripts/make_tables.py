"""Generate LaTeX tables from experiment result CSV files."""

import argparse
import csv
from pathlib import Path


def csv_to_latex(csv_path: Path, output_path: Path, caption: str = "", label: str = ""):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Empty CSV, skipping.")
        return

    headers = rows[0]
    col_fmt = "l" + "c" * (len(headers) - 1)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows[1:]:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--caption", default="")
    parser.add_argument("--label", default="tab:generated")
    args = parser.parse_args()
    csv_to_latex(args.csv, args.output, args.caption, args.label)
