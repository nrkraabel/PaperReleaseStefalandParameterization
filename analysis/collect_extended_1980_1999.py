"""
Collect the lightweight spatial-testing artifacts (predictions, targets,
metrics) for the Extended1980_1999_Camels531 experiments out of scratch and
into HBVPaperResults, mirroring the <model>/<split>/[seed*/]spatial_aggregated_<split>
layout the plotting scripts (e.g. embedding_vs_lstmhbv_cdf*.py) expect.

Only copies aggregated_predictions.npy, aggregated_targets.npy, metrics.json,
and metrics_agg.json -- not model checkpoints, per-holdout dirs, or logs --
since that's all the CDF/comparison figures read.

Run again any time new seeds/experiments finish in scratch; already-copied
files are skipped unless --overwrite is passed.
"""
import argparse
import shutil
from pathlib import Path

SRC_ROOT = Path("${oc.env:DMG_OUTPUT_ROOT}/Extended1980_1999_Camels531")
DST_ROOT = Path("${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Extended1980_1999_Camels531")

FILES_TO_COPY = [
    "aggregated_predictions.npy",
    "aggregated_targets.npy",
    "metrics.json",
    "metrics_agg.json",
]


def find_aggregated_dirs(src_root: Path):
    """Yield every spatial_aggregated_{PUB,PUR} dir under src_root, whether it
    sits directly under <model>/<split>/ or under a <model>/<split>/seed*/.

    When a <model>/<split>/ has seed* subdirs, its top-level
    spatial_aggregated_* is a duplicate of seed111111 (same run, written to
    both places) -- skip it there to avoid double-counting that seed.
    """
    for split_dir in sorted(src_root.glob("*/*")):
        if not split_dir.is_dir():
            continue
        seed_dirs = sorted(split_dir.glob("seed*"))
        if seed_dirs:
            for seed_dir in seed_dirs:
                yield from seed_dir.glob("spatial_aggregated_*")
        else:
            yield from split_dir.glob("spatial_aggregated_*")


def collect(overwrite: bool = False, dry_run: bool = False):
    copied, skipped, empty, incomplete = [], [], [], []

    for agg_dir in sorted(find_aggregated_dirs(SRC_ROOT)):
        rel = agg_dir.relative_to(SRC_ROOT)
        dst_dir = DST_ROOT / rel

        present = [f for f in FILES_TO_COPY if (agg_dir / f).exists() and (agg_dir / f).stat().st_size > 0]
        if not present:
            empty.append(rel)
            continue
        if len(present) < len(FILES_TO_COPY):
            incomplete.append((rel, present))

        dst_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
        for fname in present:
            src_f = agg_dir / fname
            dst_f = dst_dir / fname
            if dst_f.exists() and not overwrite:
                skipped.append(dst_f.relative_to(DST_ROOT))
                continue
            if dry_run:
                copied.append(dst_f.relative_to(DST_ROOT))
                continue
            shutil.copy2(src_f, dst_f)
            copied.append(dst_f.relative_to(DST_ROOT))

    print(f"{'Would copy' if dry_run else 'Copied'}: {len(copied)} files")
    print(f"Skipped (already present): {len(skipped)} files")
    if empty:
        print(f"\nEmpty/missing aggregated dirs skipped ({len(empty)}):")
        for r in empty:
            print(f"  {r}")
    if incomplete:
        print(f"\nPartial aggregated dirs (missing some files) ({len(incomplete)}):")
        for r, present in incomplete:
            missing = sorted(set(FILES_TO_COPY) - set(present))
            print(f"  {r}  missing={missing}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true", help="Re-copy files that already exist at the destination.")
    p.add_argument("--dry-run", action="store_true", help="List what would be copied without writing anything.")
    args = p.parse_args()
    collect(overwrite=args.overwrite, dry_run=args.dry_run)
