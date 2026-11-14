"""Verify optional newly downloaded Final dataset copies against audited inputs."""
import json
from pathlib import Path
from common import HERE,sha256
report=json.loads((HERE/'reports/dataset_audit.json').read_text())
checks=[]
for name,dataset in [('talim_10000_scanned.zip','general'),('talim_edge_cases_positive_scanned_v1.zip','positive')]:
 path=Path(r'C:\FYP Dataset Generation\Final dataset')/name
 row=dict(path=str(path),audited_path=report['datasets'][dataset]['scanned'],expected_sha256=report['datasets'][dataset]['archive_sha256'])
 if path.is_file():
  actual=sha256(path);row.update(status='identical' if actual==row['expected_sha256'] else 'different_or_incomplete',actual_sha256=actual)
 else:row['status']='not_present'
 checks.append(row)
(HERE/'reports/download_copy_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
