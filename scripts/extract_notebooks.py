import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

paths = [
 r"C:\GitHub\Paradigm_Shift_Act_Inf\notebooks\12_resource_coupling_gating.ipynb",
 r"C:\GitHub\Paradigm_Shift_Act_Inf\notebooks\13_bistability_probe.ipynb",
 r"C:\GitHub\Paradigm_Shift_Act_Inf\notebooks\14_jacobian_scan.ipynb",
]
for p in paths:
    print("\n\n########################", p)
    try:
        nb=json.load(open(p,encoding="utf-8"))
        for i,c in enumerate(nb.get("cells",[])):
            src="".join(c.get("source",[]))
            if c["cell_type"]=="markdown":
                print(f"\n--- MD[{i}] ---\n{src}")
            else:
                outs=[]
                for o in c.get("outputs",[]):
                    if o.get("output_type")=="stream": outs.append("".join(o.get("text",[])))
                    elif "data" in o and "text/plain" in o["data"]: outs.append("".join(o["data"]["text/plain"]))
                if outs:
                    srctrunc = src[:800] if len(src) > 800 else src
                    print(f"\n--- CODE[{i}] ---\n{srctrunc}")
                    outstr = chr(10).join(outs)
                    print(f"--- OUT[{i}] ---\n{outstr}")
    except FileNotFoundError:
        print(f"FILE NOT FOUND: {p}")
