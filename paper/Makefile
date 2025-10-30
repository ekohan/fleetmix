# Makefile for LaTeX paper compilation
# Usage: make         # compile the paper
#        make watch   # auto-recompile on changes
#        make clean   # remove auxiliary files
#        make distclean # remove all generated files including PDF

# Main document
MAIN = main
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
BIB = mybib.bib

# LaTeX compiler settings
LATEX = pdflatex
BIBTEX = bibtex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Auxiliary files to clean
AUX_FILES = *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot \
            *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb \
            *.bcf *.run.xml *.acn *.acr *.alg *.glg *.glo *.gls \
            *.ist *.spl *-blx.bib

.PHONY: all clean distclean watch help

# Default target
all: $(PDF)

# Compile the paper (full build with bibliography)
$(PDF): $(TEX) $(BIB)
	@echo "==> First pass..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "==> Running BibTeX..."
	$(BIBTEX) $(MAIN)
	@echo "==> Second pass..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "==> Final pass..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "==> PDF compiled successfully: $(PDF)"

# Quick compile (single pass, no bibliography)
quick: $(TEX)
	@echo "==> Quick compile (single pass)..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "==> PDF compiled: $(PDF)"

# Watch mode - recompile on changes (requires latexmk)
watch:
	@echo "==> Watching for changes... (Ctrl+C to stop)"
	@command -v latexmk >/dev/null 2>&1 || { echo "Error: latexmk not found. Install with: brew install latexmk"; exit 1; }
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

# Clean auxiliary files
clean:
	@echo "==> Cleaning auxiliary files..."
	rm -f $(AUX_FILES)

# Clean everything including PDF
distclean: clean
	@echo "==> Removing PDF..."
	rm -f $(PDF)

# Open PDF (macOS)
open: $(PDF)
	@echo "==> Opening PDF..."
	open $(PDF)

# View PDF (Linux with evince)
view: $(PDF)
	@echo "==> Opening PDF..."
	evince $(PDF) &

# Help
help:
	@echo "Available targets:"
	@echo "  make         - Compile the paper (full build with bibliography)"
	@echo "  make quick   - Quick compile (single pass, no bibliography)"
	@echo "  make watch   - Auto-recompile on changes (requires latexmk)"
	@echo "  make clean   - Remove auxiliary files"
	@echo "  make distclean - Remove all generated files including PDF"
	@echo "  make open    - Compile and open PDF (macOS)"
	@echo "  make view    - Compile and view PDF (Linux)"
	@echo "  make help    - Show this help message"

