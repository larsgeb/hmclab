# Minimal makefile for Sphinx documentation
#
SHELL := '/bin/bash'
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../docs/build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile



# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	rm -rf source/notebooks 
	rm -rf ../docs/* 
	mkdir source/notebooks
	mkdir source/notebooks/tutorials
	mkdir source/notebooks/examples
	# mkdir source/notebooks/inverse_problems
	cp -r ../notebooks/tutorials/*.ipynb source/notebooks/tutorials
	cp -r ../notebooks/examples/*.ipynb source/notebooks/examples
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	git rev-parse --short HEAD | xargs -I {} sed -i 's/COMMIT_HASH/{}/g' ../docs/build/html/index.html 
