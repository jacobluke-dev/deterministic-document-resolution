from pathlib import Path

def project_path(rel: str) -> str:
    # keep simple; you can make smarter later
    return str(Path.cwd() / rel)

def load_file(path: str, ext: str):
    p = Path(path)
    if ext == "json":
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8")
