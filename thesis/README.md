# Thesis

End-to-End Reinforcement Learning for Autonomous Tractor-Trailer Control

## Build

```bash
make        # compile PDF → build/main.pdf
make view   # compile and open PDF
make watch  # auto-recompile on file changes
make clean  # remove build artifacts
```

Requires: `latexmk`, `pdflatex`, and the packages listed in `main.tex`.

## Structure

```
thesis/
├── main.tex                    # Root document
├── Makefile / latexmkrc        # Build system
├── chapters/                   # Chapter source files
├── figures/                    # Images, diagrams (by topic)
├── tables/                     # Standalone table .tex files
├── equations/                  # Standalone equation .tex files
├── bibliography/               # .bib files
├── appendices/                 # Appendix source files
├── scripts/                    # Python helper scripts
├── data/                       # Experiment data and logs
└── build/                      # Compiled output (git-ignored)
```
