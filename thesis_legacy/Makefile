MAIN = main
BUILDDIR = build
LATEXMK = latexmk

MERGED = main_merged_test
REVIEW_TEX = chapters/04_learning_and_results.tex chapters/sections/scalable_simulation.tex

.PHONY: all clean view merged merged-view

all:
	$(LATEXMK) -r latexmkrc $(MAIN).tex

view: all
	xdg-open $(BUILDDIR)/$(MAIN).pdf

# Build the merged-chapter draft and turn every "% REVIEW" comment and \pending
# marker in its sources into a PDF annotation, readable in a viewer's comment
# sidebar. The annotated copy is derived: rebuild it, never edit it by hand.
merged:
	$(LATEXMK) -r latexmkrc $(MERGED).tex
	python3 scripts/annotate_review_markers.py \
		--pdf $(BUILDDIR)/$(MERGED).pdf \
		--out $(BUILDDIR)/$(MERGED)_annotated.pdf \
		--tex $(REVIEW_TEX)

merged-view: merged
	xdg-open $(BUILDDIR)/$(MERGED)_annotated.pdf

clean:
	$(LATEXMK) -C -r latexmkrc $(MAIN).tex
	rm -rf $(BUILDDIR)

# Watch mode: rebuild on file changes
watch:
	$(LATEXMK) -pvc -r latexmkrc $(MAIN).tex
