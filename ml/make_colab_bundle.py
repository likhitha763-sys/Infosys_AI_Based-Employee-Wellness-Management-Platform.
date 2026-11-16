"""Bundle the cleaned data and label code into one zip for Google Colab."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "moodmentor_colab_bundle.zip"

FILES = [
    "ml/__init__.py",
    "ml/config.py",
    "ml/data_prep/__init__.py",
    "ml/data_prep/labels.py",
    "ml/train_goemotions.py",
    "ml/score_external.py"
]
DATA_DIRS = [
    "ml/data/processed/goemotions",
    "ml/data/processed/semeval2018_voc_binary",
    "ml/data/processed/isear",
]


def main() -> None:
    paths = [ROOT / f for f in FILES]
    for d in DATA_DIRS:
        folder = ROOT / d
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        paths += sorted(p for p in folder.rglob("*") if p.is_file())

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.relative_to(ROOT).as_posix())
            print("added:", p.relative_to(ROOT).as_posix())
    print(f"\nWrote {OUT.name}: {OUT.stat().st_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()