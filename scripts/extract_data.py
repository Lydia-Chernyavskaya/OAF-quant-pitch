"""Extract OptionsDX SPX .7z archives into ~/data/spx_eod/."""
import py7zr
import glob
from pathlib import Path

DOWNLOAD_DIR = Path.home() / "Downloads"
TARGET_DIR = Path.home() / "data" / "spx_eod"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

archives = list(DOWNLOAD_DIR.glob("spx*.7z"))
if not archives:
    raise FileNotFoundError(f"No spx*.7z files found in {DOWNLOAD_DIR}")

for archive in archives:
    print(f"Extracting {archive.name}...")
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(TARGET_DIR)
    print(f"  Done → {TARGET_DIR}")

print(f"\nAll done. {len(list(TARGET_DIR.iterdir()))} files in {TARGET_DIR}")