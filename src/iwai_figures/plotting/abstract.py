"""Original Matplotlib graphical abstract, preserved exactly."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from .style import use_style, save as orx_save, BLUE, ORANGE, GREY
INK = "#222222"
def arrow(ax, a, b, color=GREY, width=.9, both=False, **kwargs):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<->" if both else "-|>",
                                mutation_scale=7, linewidth=width, color=color,
                                shrinkA=0, shrinkB=0, **kwargs))


def node(ax, xy, label, color="white", radius=.08, textcolor=INK):
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor=INK, linewidth=.7))
    ax.text(*xy, label, ha="center", va="center", fontsize=8, color=textcolor)


def mini_dag(ax, xy, kind, scale=1., labels=("A", "B", "C"), color=BLUE):
    x, y = xy
    pts = [(x, y), (x + .42 * scale, y), (x + .84 * scale, y)]
    radius = .07 * scale
    if kind == "direct":
        arrow(ax, (x + radius, y + .035 * scale), (x + .84 * scale - radius, y + .035 * scale),
              color=color, connectionstyle="arc3,rad=-.55")
    else:
        for a, b in zip(pts[:-1], pts[1:]):
            arrow(ax, (a[0] + radius, a[1]), (b[0] - radius, b[1]), color=color)
    for p, label in zip(pts, labels):
        node(ax, p, label, radius=radius)


def save(fig, out, name):
    return orx_save(fig, out / name)


def figure1(out):
    fig = plt.figure(figsize=(4.8, 1.8))
    ax = fig.add_axes([.02, .08, .96, .88]); ax.set(xlim=(0, 4.8), ylim=(0, 1.55)); ax.axis("off")
    ax.text(.48, 1.44, "Observations", ha="center")
    ax.text(2.13, 1.44, "Agent $i$", ha="center")
    ax.text(4.03, 1.44, "Social graph", ha="center")
    for y, label in ((.96, "$y_{i,1}$"), (.52, "$y_{i,2}$")):
        ax.add_patch(Rectangle((.18, y-.12), .61, .24, facecolor="white", edgecolor=GREY, lw=.8))
        ax.text(.485, y, label, ha="center", va="center")
        arrow(ax, (.79,y), (1.31,.75), color=GREY)
    ax.add_patch(Rectangle((1.34,.08), 1.60, 1.23, fill=False, edgecolor="#999999", lw=.7))
    ax.text(1.47,1.08,"$m_1$", ha="center", color=BLUE)
    ax.text(1.47,.66,"$m_2$", ha="center", color=BLUE)
    mini_dag(ax,(1.74,1.02),"direct",scale=1.05,color=BLUE)
    mini_dag(ax,(1.74,.65),"chain",scale=1.05,color=BLUE)
    ax.text(2.14,.25,r"$q_i(m_1),\;q_i(m_2)$",ha="center")
    arrow(ax,(2.97,.77),(3.51,.77),color=INK,both=True)
    ax.text(3.25,1.01,"Share",ha="center",fontsize=8)
    pts=[(3.66,.77),(4.08,1.13),(4.17,.54),(4.58,.94),(4.6,.23)]
    for i,j in [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]:
        ax.plot([pts[i][0],pts[j][0]],[pts[i][1],pts[j][1]],color="#999999",lw=.8,zorder=0)
    for i,p in enumerate(pts):
        node(ax,p,"$i$" if i==0 else "",color="#DCEEF6" if i==0 else "white",radius=.10)
    return save(fig,out,"fig01_model"), {
        "caption": "An agent maintains a Gaussian belief for each supplied candidate model and a probability over candidate identities. Observations update the candidates' predictive scores and parameters. Along the fixed social graph, agents pool model scores and Gaussian information within matching candidate identities. The two candidate graphs are conceptual illustrations; no numerical model weights are depicted. The headline experiments do not generate or transmit new candidate structures.",
        "role": "Main-text architecture; replaces the original prototype proposal diagram."}
