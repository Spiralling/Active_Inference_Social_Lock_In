"""Recreate exactly the sixteen unique figures in the two supplied archives."""
from pathlib import Path
import json
import shutil
from . import style


def render_all(data_dir, output_dir):
    """Render preserved layouts from portable, hash-checked per-seed data.

    Filenames match reference/figures and the manifest's paper-specific aliases.
    This entry point never runs simulations or reads repository history.
    """
    from . import camera, abstract, communication, gaussian
    from .data import CameraData, PlotData
    data_dir=Path(data_dir).resolve();out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    style.use_style()
    camera.HERE=out;communication.HERE=out;gaussian.HERE=out
    source=Path(__file__).resolve().parent
    for stem in ['fig01_general_representation','fig_model']:
        shutil.copyfile(source/(stem+'.tex'),out/(stem+'.tex'))
    records={};data=CameraData(data_dir/'camera')
    protocol=json.loads((data_dir/'camera/protocol.json').read_text())
    for stem in ['fig01_general_representation','fig_model']:
        records[stem]=camera.diagram(stem,'Schematic; no simulated result.')
    records['fig02_precision_pool']=camera.precision_figure(data['world'])
    for stem,fn in [('fig03_camp_distance',lambda:camera.camps(data)),
                    ('fig04_cycle',lambda:camera.cycle(data,protocol)),
                    ('fig06_topology',lambda:camera.topology_figure(data,data['graphs'],protocol)),
                    ('fig_camp_choices',lambda:camera.camp_choices(data)),
                    ('fig_adaptation',lambda:camera.adaptation_figure(data,protocol)),
                    ('fig_identifiability',lambda:camera.structural_figure(data))]:
        fig,entry=fn();entry['outputs']=style.save(fig,out/stem)
        camera.check_pdf(out/(stem+'.pdf'));records[stem]=entry
        print('Rendered',stem,flush=True)
    files,entry=abstract.figure1(out)
    for extension in ['pdf','svg','png']:
        (out/('fig01_model.'+extension)).rename(out/('fig_graphical_abstract.'+extension))
    records['fig_graphical_abstract']=entry
    commdata=PlotData(data_dir/'communication')
    commprotocol=json.loads((data_dir/'communication/protocol.json').read_text())
    communication.network(commdata,records)
    communication.camps(commdata,records)
    communication.diagnostics(commdata,records,commprotocol)
    communication.branches(commdata,records)
    gaussdata=PlotData(data_dir/'gaussian',gaussian=True)
    gaussian.comparison(gaussdata,records);gaussian.diagnostics(gaussdata,records)
    # Keep output provenance portable: filenames and hashes, no machine paths.
    manifest={'figure_count':len(records),'figures':{name:{'pdf':name+'.pdf','sha256':camera.sha256(out/(name+'.pdf'))} for name in records},
              'font':'LaTeX Computer Modern / CM-Super T1','style':'vendored orx; original layouts preserved'}
    (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
