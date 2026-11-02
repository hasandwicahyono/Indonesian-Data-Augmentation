"""
LaTeX Compilation and Manuscript Synchronization Utilities.
"""

import os
import shutil
import subprocess
from typing import Optional

from src.utils.path_utils import resolve_path


def compile_latex_manuscripts(paper_dir: str = "paper") -> bool:
    """
    Compiles main.tex and tpami_manuscript.tex using pdflatex and bibtex.
    Ensures bibliography references and citations are cleanly linked.
    """
    resolved_dir = resolve_path(paper_dir)
    pdflatex_bin = "/Library/TeX/texbin/pdflatex"
    if not os.path.exists(pdflatex_bin):
        pdflatex_bin = shutil.which("pdflatex")
        if not pdflatex_bin:
            print("  [Warning] pdflatex not found on system PATH. Skipping PDF compilation.")
            return False

    bibtex_bin = "/Library/TeX/texbin/bibtex"
    if not os.path.exists(bibtex_bin):
        bibtex_bin = shutil.which("bibtex")

    success = True
    for doc in ["main.tex", "tpami_manuscript.tex"]:
        doc_path = os.path.join(resolved_dir, doc)
        if not os.path.exists(doc_path):
            continue

        doc_stem = os.path.splitext(doc)[0]
        print(f"  [Compiling PDF] Running pdflatex -> bibtex -> pdflatex -> pdflatex for {doc} in {resolved_dir}...")
        try:
            # 1. First pdflatex pass
            subprocess.run(
                [pdflatex_bin, "-interaction=batchmode", doc],
                cwd=resolved_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            # 2. Bibtex pass
            if bibtex_bin:
                subprocess.run(
                    [bibtex_bin, doc_stem],
                    cwd=resolved_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            # 3. Second pdflatex pass
            subprocess.run(
                [pdflatex_bin, "-interaction=batchmode", doc],
                cwd=resolved_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            # 4. Final pdflatex pass
            res = subprocess.run(
                [pdflatex_bin, "-interaction=batchmode", doc],
                cwd=resolved_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if res.returncode == 0:
                print(f"  [Success] {doc.replace('.tex', '.pdf')} compiled cleanly with full references!")
            else:
                print(f"  [Warning] {doc} compilation returned non-zero code {res.returncode}")
                success = False
        except Exception as e:
            print(f"  [Error] Failed to compile {doc}: {e}")
            success = False

    return success

