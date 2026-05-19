# Agent Instructions

## Environment Safety

Do not modify the user's base conda environment.

- Treat `/home/lq53/miniconda3` and the `base` conda env as read-only.
- Do not run `pip install`, `conda install`, `mamba install`, `uv pip install`,
  or any equivalent install command against `base`.
- Before running install or package-management commands, confirm the active
  Python/env with `which python`, `python -V`, and `echo $CONDA_DEFAULT_ENV`.
- If installability needs to be tested, create and use an isolated disposable
  test environment outside `base`, preferably under `/tmp` or another explicit
  scratch location.
- If a test environment would require network access or package installation,
  ask first and name the exact environment path.

For this repository, prefer static checks and import/path inspection unless the
user explicitly asks to run tests in a named non-base environment.
