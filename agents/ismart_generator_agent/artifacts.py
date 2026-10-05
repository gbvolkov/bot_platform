"""Export generated packages without changing generation or CLI output files."""
import base64
import io
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def export_artifacts(state):
    artifacts = []
    for result in state.get("results", []):
        directory = Path(result["output_dir"]).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Generated package is missing: {directory}")
        content = io.BytesIO()
        with ZipFile(content, "w", ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.resolve().is_relative_to(directory):
                    archive.write(path, path.relative_to(directory).as_posix())
        artifacts.append({"filename": directory.name + ".zip", "content_type": "application/zip",
                          "data": base64.b64encode(content.getvalue()).decode("ascii")})
    return artifacts
