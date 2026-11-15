"""Download the ISEAR CSV files from a Hugging Face dataset mirror."""
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "TahaRasouli/ISEAR"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ml" / "data" / "raw" / "isear"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    print("Files in the repo:")
    for f in files:
        print("  ", f)

    csv_files = [f for f in files if f.lower().endswith(".csv")]
    if not csv_files:
        raise SystemExit("No CSV files found in the repo.")

    for f in csv_files:
        path = hf_hub_download(
            repo_id=REPO_ID, filename=f, repo_type="dataset", local_dir=OUT_DIR
        )
        print("Downloaded:", path)


if __name__ == "__main__":
    main()