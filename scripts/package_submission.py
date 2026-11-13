import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "Adaptive_RAG_Assistant_Submission.zip"

EXCLUDE_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
EXCLUDE_EXTS = {".pyc", ".pyo"}

print(f"Packaging project from {PROJECT_ROOT} into {OUTPUT_ZIP}...")

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in EXCLUDE_EXTS:
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            zipf.write(full_path, os.path.join("adaptive_rag_assistant", rel_path))

print(f"SUCCESS! Created submission archive at:\n{OUTPUT_ZIP}")
