"""Generate TikZ fragments for the trust-model figure from the real graph library.
Writes three fragments: node-link graph, adjacency heatmap, trust (update) heatmap.
The matrices are the actual Watts-Strogatz adjacency and its row-stochastic
trust matrix T = rownorm(A + I) used by src/pomdp/step.build_trust."""
import math
import numpy as np
import networkx as nx

N, k, p, seed = 12, 4, 0.25, 7
g = nx.watts_strogatz_graph(n=N, k=k, p=p, seed=seed)
A = nx.to_numpy_array(g, nodelist=list(range(N)), dtype=float)
np.fill_diagonal(A, 0.0)
# Trust = row-normalised adjacency over NEIGHBOURS only (no self term):
# own-belief inertia is supplied by the conservatism lambda in the update,
# so a self-weight here would double-count it.
T = A / A.sum(1, keepdims=True)
Tmax = float(T.max())

R = 2.45
pos = {i: (R * math.cos(2 * math.pi * i / N + math.pi / 2),
           R * math.sin(2 * math.pi * i / N + math.pi / 2)) for i in range(N)}

# --- node-link graph (nodes first so edges can reference them) ---
graph = []
for i in range(N):
    graph.append(rf"\node[anode] (n{i}) at ({pos[i][0]:.2f},{pos[i][1]:.2f}) {{{i}}};")
for i in range(N):
    for j in range(i + 1, N):
        if A[i, j] > 0:
            graph.append(rf"\draw[chan] (n{i}) -- (n{j});")

s = 0.34  # cell size in cm
def cell(j, i):
    x0, y0 = j * s, -i * s
    return f"({x0:.3f},{y0:.3f}) rectangle ({x0 + s:.3f},{y0 + s:.3f})"

adj = []
for i in range(N):
    for j in range(N):
        if A[i, j] > 0:
            adj.append(rf"\fill[adjc] {cell(j, i)};")
adj.append(rf"\draw[mframe] (0,{s:.3f}) rectangle ({N*s:.3f},{-(N-1)*s:.3f});")

trust = []
for i in range(N):
    for j in range(N):
        w = T[i, j]
        if w > 0:
            pct = max(10, int(round(w / Tmax * 100)))
            trust.append(rf"\fill[trustc!{pct}] {cell(j, i)};")
trust.append(rf"\draw[mframe] (0,{s:.3f}) rectangle ({N*s:.3f},{-(N-1)*s:.3f});")

for name, lines in [("graph", graph), ("adj", adj), ("T", trust)]:
    with open(f"paper/_trustfig_{name}.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

print(f"wrote 3 fragments; N={N} k={k} p={p} seed={seed} Tmax={Tmax:.3f} gridside={N*s:.2f}cm")
print("degrees:", [int(A[i].sum()) for i in range(N)])
