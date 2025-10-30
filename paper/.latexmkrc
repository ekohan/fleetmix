# latexmk configuration for elsarticle paper

# Use pdflatex
$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Compiler settings
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# BibTeX settings
$bibtex_use = 2;  # Run bibtex when needed

# Clean extra extensions
$clean_ext = 'bbl nav snm vrb spl synctex.gz';

# Preview settings (for continuous preview)
$preview_mode = 1;
$pdf_previewer = 'open %O %S';  # macOS default (use 'evince' for Linux)

# Generate PDF in current directory
$out_dir = '.';

