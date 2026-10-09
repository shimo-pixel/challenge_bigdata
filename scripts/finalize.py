"""Validate all 400 predictions, package them, then update the report."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text2query.pipeline import validate_submission
from text2query.io import dump_json
root=Path(__file__).resolve().parents[1]
source=root/'result_v2.json'
if not source.exists(): raise SystemExit('result_v2.json is not ready; resume prediction first')
validation=validate_submission(root/'.t2q',source)
if not validation['ok']: raise SystemExit(str(validation))
destination=root/'submission_v2'/'result.json'
destination.parent.mkdir(parents=True,exist_ok=True)
fd,temporary=tempfile.mkstemp(dir=destination.parent,prefix='.result-')
os.close(fd)
try:
    shutil.copyfile(source,temporary)
    os.replace(temporary,destination)
finally:
    if os.path.exists(temporary): os.unlink(temporary)
dump_json(root/'reports'/'submission_validation.json',validation)
subprocess.run([sys.executable,str(root/'scripts'/'summarize.py')],check=True,cwd=root)
print('Ready to submit:',destination)
