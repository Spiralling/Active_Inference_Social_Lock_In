import json
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

outf = io.open(r'C:\GitHub\Paradigm_Shift_Act_Inf\audit_output.txt', 'w', encoding='utf-8')

paths = [
    r'C:\GitHub\Paradigm_Shift_Act_Inf\notebooks\09_paradigm_competition.ipynb',
    r'C:\GitHub\Paradigm_Shift_Act_Inf\notebooks\10_fiedler_vanguard.ipynb',
    r'C:\GitHub\Paradigm_Shift_Act_Inf\notebooks\11_endogenous_oscillation.ipynb',
]

for p in paths:
    outf.write('\n\n######################## ' + p + '\n')
    try:
        nb = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        outf.write(f"Error loading {p}: {e}\n")
        continue

    for i, c in enumerate(nb.get('cells', [])):
        src = "".join(c.get('source', []))

        if c['cell_type'] == 'markdown':
            outf.write(f'\n--- MD[{i}] ---\n{src[:1500]}\n')
        else:
            # Extract code cell outputs
            outs = []
            for o in c.get('outputs', []):
                if o.get('output_type') == 'stream':
                    outs.append("".join(o.get('text', [])))
                elif 'data' in o and 'text/plain' in o['data']:
                    outs.append("".join(o['data']['text/plain']))

            if outs:
                outf.write(f'\n--- CODE[{i}] (first 400 chars) ---\n{src[:400]}\n')
                outf.write(f'--- OUTPUT[{i}] (first 2000 chars) ---\n')
                for out in outs:
                    outf.write(out[:2000] + '\n')

outf.close()
print("Output written to audit_output.txt")
