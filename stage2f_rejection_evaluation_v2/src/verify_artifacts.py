"""Hash/metadata preflight. Optional selected-development image hash checks only."""
import argparse
from integration import *
from dataset_loader import DevelopmentDataset,SPLIT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--split',choices=(SPLIT,),default=SPLIT)
    p.add_argument('--check-dataset',action='store_true',help='Verify ONLY selected development image bytes and annotations; no inference')
    a=p.parse_args();report=verify_integrity()
    if a.check_dataset:
        data=DevelopmentDataset(a.dataset,a.split,verify_files=True)
        report.update(development_images=len(data.rows),development_parents=len(data.groups),split=SPLIT,
                      protected_or_validation_images_opened=0,dataset_manifest_sha256=data.manifest_sha256)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
