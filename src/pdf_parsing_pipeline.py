"""Compatibility wrapper for the pipeline module.

The canonical implementation lives at repo root: `pdf_parsing_pipeline.py`.
Some scripts/tests import it as `src.pdf_parsing_pipeline`, so we re-export
everything from there.
"""

from pdf_parsing_pipeline import *  # noqa: F401,F403

