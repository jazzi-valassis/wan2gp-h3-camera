import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SLUG = "wan2gp-h3-camera"
FILES = ("__init__.py", "plugin.py", "plugin_info.json", "camera_plan.py", "editor.py", "image_anchors.py", "timing.py",
         "README.md", "VALIDATION.md", "CHANGELOG.md", "LICENSE", "NOTICE.md", ".gitignore")
DIRECTORIES = ("web", "tests", "scripts")


def main():
    parser = argparse.ArgumentParser(description="Build a portable H3 Camera plugin ZIP.")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "plugin_info.json").read_text(encoding="utf-8"))
    sources = [ROOT / name for name in FILES]
    for folder in DIRECTORIES:
        sources.extend(path for path in (ROOT / folder).rglob("*")
                       if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                       and path.name != ".DS_Store")
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.is_file()]
    if missing:
        parser.error("Missing release files: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"{SLUG}-{metadata['version']}.zip"
    manifest = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(sources):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(f"{SLUG}/{relative}", date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            target.writestr(info, data)
            manifest[relative] = hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    archive.with_suffix(".files.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive.resolve()), "files": len(manifest),
                      "bytes": archive.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
