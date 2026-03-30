MAIN = main
BUILDDIR = build
LATEXMK = latexmk

.PHONY: all clean view

all:
	$(LATEXMK) -r latexmkrc $(MAIN).tex

view: all
	xdg-open $(BUILDDIR)/$(MAIN).pdf

clean:
	$(LATEXMK) -C -r latexmkrc $(MAIN).tex
	rm -rf $(BUILDDIR)

# Watch mode: rebuild on file changes
watch:
	$(LATEXMK) -pvc -r latexmkrc $(MAIN).tex
