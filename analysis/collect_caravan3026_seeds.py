"""
Collect the multi-seed spatial-testing artifacts for the Caravan3026
experiments out of scratch and into HBVPaperResults, mirroring the
<model>[/<subtype>]/<split>/seed*/spatial_aggregated_<split> layout the
plotting scripts (caravan3026_cdf_3seed_panels.py) expect.

This copies the raw prediction/target arrays, NOT the stored metrics JSON.
Earlier versions did the reverse, on the argument that metrics.json exactly
reproduced a from-scratch recomputation off the .npy arrays. That argument
held only as long as nothing re-ran: when Embedding/daily and
Embedding/monthly were re-tested, the new aggregated_predictions.npy landed
next to a metrics.json still describing the *previous* predictions, and the
stale JSON silently kept feeding the figures. Checked 2026-08-24: for those
runs the stored metrics.json matches the superseded predictions exactly
(NSE max|diff| = 0) and disagrees with the current ones by up to ~1e2 NSE.

So the arrays are the source of truth and the plotting script recomputes
per-basin NSE/KGE from them. That costs ~3.5 GB here and ~40 s per figure
run, which buys the guarantee that a rerun test can never be misread as its
predecessor. It also makes the collected tree useful for hydrographs, FDCs
and any other per-timestep analysis, which metrics.json never supported.

Only DATA_FILES are ever copied -- an explicit allowlist, not a glob -- so
unrelated files sitting in an aggregated dir are never picked up.

The Caravan3026 tree nests two ways -- LSTMHBV/<split>/seed*/ but
Embedding/<daily|monthly|annual>/<split>/seed*/ -- so aggregated dirs are
found by walking for seed* rather than by a fixed glob depth.

Run again any time new seeds finish in scratch. Files already at the
destination are re-copied when the source is newer (that is the whole point
of the rewrite above); pass --no-refresh for the old skip-if-present
behaviour, or --overwrite to re-copy unconditionally.

    python collect_caravan3026_seeds.py --dry-run     # see what would move
    python collect_caravan3026_seeds.py               # copy arrays
    python collect_caravan3026_seeds.py --prune-json  # + drop stale JSONs
"""
import argparse
import shutil
from pathlib import Path

SRC_ROOT = Path("${oc.env:DMG_OUTPUT_ROOT}/Caravan3026")
DST_ROOT = Path("${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Caravan3026")

# Explicit allowlist. Nothing else in an aggregated dir is copied, whatever
# it is called.
DATA_FILES = ["aggregated_predictions.npy", "aggregated_targets.npy"]

# Written by dmg at test time, superseded by recomputation from the arrays.
# --prune-json deletes these from the destination so a future reader cannot
# pick up a copy that has drifted from the predictions beside it.
STALE_JSON = ["metrics.json", "metrics_agg.json"]


def find_aggregated_dirs(src_root: Path):
    """Yield every spatial_aggregated_* dir that represents a distinct run.

    A split dir with seed* subdirs also carries a split-level
    spatial_aggregated_* that duplicates (or predates) seed111111 -- skip it
    so that run is not counted twice. Split dirs without seeds contribute
    their own.
    """
    for seed_dir in sorted(src_root.glob("**/seed*")):
        if seed_dir.is_dir():
            yield from sorted(seed_dir.glob("spatial_aggregated_*"))

    for agg_dir in sorted(src_root.glob("**/spatial_aggregated_*")):
        parent = agg_dir.parent
        if parent.name.startswith("seed"):
            continue
        if any(parent.glob("seed*")):
            continue
        yield agg_dir


def needs_copy(src_f: Path, dst_f: Path, overwrite: bool, refresh: bool):
    """Whether to copy, and a one-word reason for the log."""
    if not dst_f.exists():
        return True, "new"
    if overwrite:
        return True, "overwrite"
    # Source rewritten since we last collected it -- this is the stale-data
    # case the module docstring is about, so refresh by default.
    if refresh and src_f.stat().st_mtime > dst_f.stat().st_mtime + 1:
        return True, "newer"
    if refresh and src_f.stat().st_size != dst_f.stat().st_size:
        return True, "size"
    return False, "present"


def prune_stale_json(dry_run: bool):
    """Delete metrics{,_agg}.json copies from the destination tree.

    Scoped to spatial_aggregated_* dirs -- the runs this script manages, and
    the ones whose JSON can disagree with the arrays sitting beside it. JSON
    under spatial_holdout_*/ and elsewhere is left alone.

    They are still in scratch beside the run that produced them; what is
    removed here is only the collected copy, which nothing reads any more:
    both caravan3026_cdf_*.py compute per-basin metrics from the arrays.
    """
    removed = []
    for name in STALE_JSON:
        for f in sorted(DST_ROOT.glob("**/spatial_aggregated_*/" + name)):
            removed.append(f.relative_to(DST_ROOT))
            if not dry_run:
                f.unlink()
    print("\n{} {} stale JSON file(s) from {}".format(
        "Would remove" if dry_run else "Removed", len(removed), DST_ROOT))
    return removed


def collect(overwrite: bool = False, refresh: bool = True,
            dry_run: bool = False, prune_json: bool = False):
    copied, skipped, empty, incomplete = [], [], [], []
    reasons = {}
    n_bytes = 0

    for agg_dir in sorted(set(find_aggregated_dirs(SRC_ROOT))):
        rel = agg_dir.relative_to(SRC_ROOT)
        dst_dir = DST_ROOT / rel

        present = [f for f in DATA_FILES
                   if (agg_dir / f).exists() and (agg_dir / f).stat().st_size > 0]
        if not present:
            empty.append(rel)
            continue
        if len(present) < len(DATA_FILES):
            incomplete.append((rel, present))

        for fname in present:
            src_f, dst_f = agg_dir / fname, dst_dir / fname
            do_copy, why = needs_copy(src_f, dst_f, overwrite, refresh)
            if not do_copy:
                skipped.append(dst_f.relative_to(DST_ROOT))
                continue
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst_f)
            n_bytes += src_f.stat().st_size
            copied.append(dst_f.relative_to(DST_ROOT))
            reasons[dst_f.relative_to(DST_ROOT)] = why

    print("{}: {} files ({:.2f} GB)".format(
        "Would copy" if dry_run else "Copied", len(copied), n_bytes / 1e9))
    print("Skipped (up to date): {} files".format(len(skipped)))

    refreshed = [r for r in copied if reasons.get(r) in ("newer", "size")]
    if refreshed:
        print("\nRefreshed because scratch is newer than the collected copy "
              "({}):".format(len(refreshed)))
        for r in refreshed:
            print("  {}".format(r))

    if empty:
        print("\nEmpty/missing aggregated dirs skipped ({}):".format(len(empty)))
        for r in empty:
            print("  {}".format(r))
    if incomplete:
        print("\nPartial aggregated dirs (missing some arrays) ({}):".format(
            len(incomplete)))
        for r, present in incomplete:
            missing = sorted(set(DATA_FILES) - set(present))
            print("  {}  missing={}".format(r, missing))

    if prune_json:
        prune_stale_json(dry_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-copy every file, even if the destination is current.")
    p.add_argument("--no-refresh", dest="refresh", action="store_false",
                   help="Skip any file already present, even if scratch is "
                        "newer. Restores the pre-2026-08-24 behaviour that let "
                        "re-tested runs go uncollected.")
    p.add_argument("--prune-json", action="store_true",
                   help="Also delete collected metrics{,_agg}.json, which the "
                        "plotting scripts no longer read and which go stale "
                        "when a run is re-tested.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be copied/removed without writing.")
    args = p.parse_args()
    collect(overwrite=args.overwrite, refresh=args.refresh,
            dry_run=args.dry_run, prune_json=args.prune_json)
