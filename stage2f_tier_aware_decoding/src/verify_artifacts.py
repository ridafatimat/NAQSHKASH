"""Lightweight hash/schema verification; never evaluates a candidate."""
import argparse
from runtime import *


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',action='store_true',help='Also verify the saved DEVELOPMENT baseline coverage/lineage')
    args=ap.parse_args()
    report=verify_integrity()
    if args.baseline:
        from evaluate_tier_aware import load_saved
        items,lineage=load_saved(BASELINE)
        report.update(baseline_rows=len(items),baseline_sha256=lineage['observations_sha256'],candidate_evaluation=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
