#!/usr/bin/env python3

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "vllm" / "results" / "raw"
ARCHIVES_DIR = RAW_DIR / "archives"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def expected_sha256(checksum_path: Path) -> str:
    text = checksum_path.read_text().strip()

    if not text:
        raise RuntimeError(
            f"Empty checksum file: {checksum_path}"
        )

    value = text.split()[0]

    if len(value) != 64:
        raise RuntimeError(
            f"Invalid SHA256 in {checksum_path}: {value}"
        )

    return value.lower()


def archive_run_name(archive: Path) -> str:
    name = archive.name

    if not name.endswith(".tar.gz"):
        raise RuntimeError(
            f"Expected .tar.gz archive, got: {archive}"
        )

    return name[:-7]


def validate_tar_paths(tf: tarfile.TarFile):
    for member in tf.getmembers():
        path = Path(member.name)

        if path.is_absolute():
            raise RuntimeError(
                f"Unsafe absolute archive path: {member.name}"
            )

        if ".." in path.parts:
            raise RuntimeError(
                f"Unsafe parent traversal in archive: {member.name}"
            )


def top_level_names(tf: tarfile.TarFile):
    names = set()

    for member in tf.getmembers():
        parts = Path(member.name).parts

        if parts:
            names.add(parts[0])

    return sorted(names)


def count_files(root: Path, pattern: str) -> int:
    return sum(
        1
        for p in root.rglob(pattern)
        if p.is_file()
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify, archive, extract, and inventory "
            "a downloaded benchmark run."
        )
    )

    parser.add_argument(
        "archive",
        type=Path,
        help="Downloaded .tar.gz file",
    )

    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()

    if not archive.is_file():
        raise RuntimeError(
            f"Archive not found: {archive}"
        )

    checksum = Path(
        str(archive) + ".sha256"
    )

    if not checksum.is_file():
        raise RuntimeError(
            f"Checksum not found: {checksum}"
        )

    print("=== VERIFY CHECKSUM ===")

    expected = expected_sha256(checksum)
    actual = sha256_file(archive)

    print(f"expected={expected}")
    print(f"actual  ={actual}")

    if actual != expected:
        raise RuntimeError(
            "SHA256 mismatch. Refusing to ingest archive."
        )

    print("checksum=OK")

    run_name = archive_run_name(archive)

    ARCHIVES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_archive = ARCHIVES_DIR / archive.name
    stored_checksum = ARCHIVES_DIR / checksum.name

    print()
    print("=== ARCHIVE STORAGE ===")

    if archive != stored_archive:
        if stored_archive.exists():
            raise RuntimeError(
                f"Archive already exists: {stored_archive}"
            )

        shutil.move(
            str(archive),
            str(stored_archive),
        )

        shutil.move(
            str(checksum),
            str(stored_checksum),
        )

        print(f"moved={stored_archive}")
    else:
        print(f"already_stored={stored_archive}")

    print()
    print("=== INSPECT ARCHIVE ===")

    with tarfile.open(
        stored_archive,
        "r:gz",
    ) as tf:
        validate_tar_paths(tf)

        roots = top_level_names(tf)

        print(
            "top_level="
            + ",".join(roots)
        )

        if len(roots) != 1:
            raise RuntimeError(
                "Expected exactly one top-level directory "
                f"in archive, found: {roots}"
            )

        archived_root = roots[0]

        if archived_root != run_name:
            raise RuntimeError(
                "Archive/run-name mismatch: "
                f"archive implies '{run_name}', "
                f"content contains '{archived_root}'"
            )

        experiment_dir = RAW_DIR / archived_root

        print()
        print("=== EXTRACT ===")

        if experiment_dir.exists():
            print(
                f"already_extracted={experiment_dir}"
            )
        else:
            tf.extractall(
                path=RAW_DIR
            )

            print(
                f"extracted={experiment_dir}"
            )

    if not experiment_dir.is_dir():
        raise RuntimeError(
            f"Experiment directory missing: {experiment_dir}"
        )

    benchmark_count = count_files(
        experiment_dir,
        "benchmark.json",
    )

    gpu_summary_count = count_files(
        experiment_dir,
        "gpu-benchmark-summary.json",
    )

    nsys_count = count_files(
        experiment_dir,
        "*.nsys-rep",
    )

    nsys_stats_count = count_files(
        experiment_dir,
        "*_stats.txt",
    )

    point_meta_count = count_files(
        experiment_dir,
        "point-meta.txt",
    )

    print()
    print("=== ARTIFACT INVENTORY ===")
    print(
        f"benchmark_json_count={benchmark_count}"
    )
    print(
        f"gpu_summary_count={gpu_summary_count}"
    )
    print(
        f"nsys_report_count={nsys_count}"
    )
    print(
        f"nsys_stats_count={nsys_stats_count}"
    )
    print(
        f"point_meta_count={point_meta_count}"
    )

    print()
    print("=== READY ===")
    print(
        f"EXP_DIR={experiment_dir.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()