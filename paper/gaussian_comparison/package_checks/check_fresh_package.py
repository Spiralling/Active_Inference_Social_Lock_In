#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    extraction=Path(tempfile.mkdtemp(prefix='iwai-gaussian-portability-')).resolve()
    start=time.monotonic();commands=[]
    report={'status':'started','extraction_directory':str(extraction),'python':sys.version,'platform':platform.platform(),'archive_name':a.archive.name}
    try:
        with zipfile.ZipFile(a.archive) as z:
            for member in z.namelist():
                target=(extraction/member).resolve()
                if not target.is_relative_to(extraction):raise AssertionError('Unsafe ZIP member')
            assert z.testzip() is None
            z.extractall(extraction)
        roots=list(extraction.glob('*/MANIFEST.json'))
        assert len(roots)==1
        root=roots[0].parent;study=root/'paper/gaussian_comparison';fig=study/'figures';old=root/'paper/communication_experiments/figures'
        def run(label,args):
            logfile=a.out/(label+'.log');then=time.monotonic()
            print('BEGIN',label,flush=True)
            with logfile.open('w') as log:
                result=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,text=True)
            commands.append({'label':label,'command':args,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-then,'log_sha256':sha(logfile)})
            print('END',label,result.returncode,round(time.monotonic()-then,2),flush=True)
            if result.returncode:raise RuntimeError(label+' failed; '+str(logfile))
        run('payload_manifest',[sys.executable,str(root/'VERIFY_PACKAGE.py')])
        before={str(p.relative_to(root)):sha(p) for p in fig.glob('fig_matched_camp_*') if p.suffix in ('.pdf','.svg','.png')}
        assert len(before)==6
        pack_before=sha(fig/'figure_pack.pdf');curves_before=sha(fig/'curve_statistics.csv')
        oldmanifest=json.loads((old/'figure_manifest.json').read_text())
        old_before={str((old/(stem+'.'+ext)).relative_to(root)):sha(old/(stem+'.'+ext)) for stem in oldmanifest['figures'] for ext in ('pdf','svg','png')}
        assert len(old_before)==21
        run('independent_mathematics',[sys.executable,str(study/'independent_tests.py')])
        run('analysis_boundaries',[sys.executable,str(study/'test_analysis.py')])
        run('validate_shipped_figures',[sys.executable,str(fig/'validate_figures.py')])
        run('regenerate_new_figures_and_pack',[sys.executable,str(fig/'plot_figures.py')])
        run('validate_regenerated_figures',[sys.executable,str(fig/'validate_figures.py')])
        equality={name:sha(root/name)==digest for name,digest in before.items()}
        assert all(equality.values()),equality
        old_equality={name:sha(root/name)==digest for name,digest in old_before.items()}
        assert all(old_equality.values())
        assert sha(fig/'figure_pack.pdf')==pack_before
        assert sha(fig/'curve_statistics.csv')==curves_before
        report.update(status='passed',payload_root=str(root),manifest_sha256=sha(root/'MANIFEST.json'),
            mathematical_tests=json.loads((study/'INDEPENDENT_TESTS.json').read_text()),
            analysis_boundary_tests=json.loads((study/'ANALYSIS_BOUNDARY_TESTS.json').read_text()) if (study/'ANALYSIS_BOUNDARY_TESTS.json').exists() else 'See analysis_boundaries.log for complete unittest result',
            figure_validation=json.loads((fig/'figure_validation.json').read_text()),
            new_asset_sha256=before,new_asset_byte_equality=equality,
            old_asset_sha256=old_before,old_asset_unchanged=old_equality,
            regenerated_nine_page_pack_byte_identical=True,pack_sha256=pack_before,
            plotted_curve_statistics_byte_identical=True,curve_statistics_sha256=curves_before,
            scope='Fresh ZIP extraction, complete payload SHA manifest, all11 independent mathematical tests, all6 analysis boundary tests, figure validator before and after actual regeneration; all6 new PDF/SVG/PNG assets and9-page pack byte-identical; all21 old assets preserved without rerender.',
            hash_binding='Archive final SHA intentionally omitted from this payload report. Parent writes an external final archive validation binding after inserting this report.')
    except Exception as e:
        report.update(status='failed',error=repr(e))
        raise
    finally:
        report.update(commands=commands,elapsed_seconds=time.monotonic()-start)
        (a.out/'PORTABLE_PAYLOAD_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',json.dumps({'seconds':report['elapsed_seconds'],'new_assets':6,'old_assets':21,'pack_pages':9}),flush=True)

if __name__=='__main__':main()
