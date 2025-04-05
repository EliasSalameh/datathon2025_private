import zipfile
import os
from pathlib import Path
import tempfile
import shutil

# Paths
top_level_zip_dir = Path.cwd()
final_output_dir = Path("clients")
final_output_dir.mkdir(parents=True, exist_ok=True)


def extract_inner_zip_preserve_structure(inner_zip_path, output_base_dir):
    client_name = inner_zip_path.stem  # e.g., client_001
    output_client_dir = output_base_dir / client_name
    output_client_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
        for file_name in inner_zip.namelist():
            if file_name.endswith('.json'):
                inner_zip.extract(file_name, output_client_dir)


# Process each top-level zip
for top_zip in top_level_zip_dir.glob("*.zip"):
    print(f"Processing: {top_zip.name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(top_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdir_path)

        # Find all inner client zips inside extracted folder
        for inner_zip_path in tmpdir_path.rglob("*.zip"):
            print(f"  ↳ Extracting client zip: {inner_zip_path.name}")
            extract_inner_zip_preserve_structure(inner_zip_path, final_output_dir)

print("✅ All JSON files extracted with folder structure preserved.")
