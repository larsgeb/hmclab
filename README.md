---

# HMCLab

__Andrea Zunino, Lars Gebraad, Andreas Fichtner__

**HMC Lab** is a numerical laboratory for research in Bayesian seismology, written in Python and Julia. Jump to [Docker one-command setup](#docker-one-command-setup).

- **Website:** https://hmclab.science
- **Python documentation:** https://python.hmclab.science
- **Source code:** https://github.com/larsgeb/hmclab

This software provides all the ingredients to set up probabilistic (and deterministic) inverse
problems, appraise them, and analyse them. This includes a plethora of prior
distributions, different physical modelling modules and various MCMC (and
other) algorithms. 

In particular it provides prior distributions, physics and appraisal algorithms.

---

# A note on version 2 of HMCLab

The project has gotten a little away from me, so this second version is to refocus and improve. Major things I want to make sure of:

- Make the project maintanable
- Make the project easier to install
- Make the project easier to understand
- Focus on the most-used functionality
- Bring forward to Python 3.11
- Make running in parallel and resuming from crashes easier