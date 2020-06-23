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
	git rev-parse --short HEAD | xargs -I {} sed 's/COMMIT_HASH/{}/g' templates/index.rst > source/index.rst
	rm source/examples -rf
	rm ../docs/* -rf
	mkdir source/examples
	cp ../examples/notebooks/*.ipynb source/examples/
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	echo "This file will be overwritten by ../templates/index.rst" > source/index.rst