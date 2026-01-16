#!/usr/bin/env python3
import csv
import shutil
import sys
from pathlib import Path


def main() -> int:
    root = Path("test_data")
    target = root / "YTL_testsets"
    nested = root / "test_data" / "YTL_testsets"

    if not target.exists() and nested.exists():
        shutil.move(str(nested), str(target))
        try:
            (root / "test_data").rmdir()
        except OSError:
            pass

    if not target.exists():
        print("ERROR: Expected test_data/YTL_testsets after extraction.")
        return 1

    tsv_files = [
        target / "fleurs_test.tsv",
        target / "malay_conversational_meta.tsv",
        target / "malay_scripted_meta.tsv",
    ]

    missing_files: list[str] = []
    for tsv in tsv_files:
        if not tsv.exists():
            missing_files.append(str(tsv))
            continue

        rows: list[dict[str, str]] = []
        fixed = False
        with tsv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            for row in reader:
                rel_path = row.get("path", "")
                audio_path = tsv.parent / rel_path
                if not audio_path.exists() and rel_path.startswith("YTL_testsets/"):
                    rel_path = rel_path.replace("YTL_testsets/", "", 1)
                    row["path"] = rel_path
                    audio_path = tsv.parent / rel_path
                    fixed = True
                if not audio_path.exists():
                    missing_files.append(str(audio_path))
                rows.append(row)

        if fixed and fieldnames:
            with tsv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)

    if missing_files:
        print("ERROR: Missing test data files:")
        for path in missing_files[:25]:
            print(f" - {path}")
        if len(missing_files) > 25:
            print(f" ... and {len(missing_files) - 25} more")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
