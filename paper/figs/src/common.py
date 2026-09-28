"""Shared style, paths and loaders for the paper's data figures (Joule, final widths 85 / 174 mm).

Every number drawn comes from a frozen artifact under experiments/phase1/out (named in each figure script); nothing is
recomputed except presentation-level aggregation of those artifacts.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "phase1" / "out"
FIGDIR = ROOT / "paper" / "figs"
NE = pathlib.Path(r"D:\_caches\naturalearth")

MM = 1 / 25.4
FULL = 174 * MM
SINGLE = 85 * MM

# Okabe-Ito colour-blind-safe palette; meaning is fixed across figures
C = {
    "hist": "#0072B2",      # historical climate (blue)
    "c2040": "#D55E00",     # SSP2-4.5 circa 2040 (vermillion)
    "stress": "#E69F00",    # uniform +1.2 K stress (orange)
    "grey": "#7F7F7F",
    "lightgrey": "#D9D9D9",
    "black": "#000000",
    "sky": "#56B4E9",
    "green": "#009E73",
    "purple": "#CC79A7",
}


def style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.unicode_minus": True,
    })


def panel_label(ax, s, dx=-0.13, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig)


def jload(rel):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))


def sites():
    return pd.read_csv(ROOT / "data" / "sites" / "hub_sites.csv")


def cds_hstar():
    """h* (hours of full IT heat) per hub cell and climate on the CDS data of record (DD-017)."""
    rec = {}
    for f in (OUT / "m2" / "cds_rebuild").glob("cell_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rec[d["cell"]] = {cl: d[cl]["h_star"] for cl in ("historical", "stress_plus1.2K", "ssp245_median")}
    return rec


def site_hstar(climate):
    s = sites()
    rec = cds_hstar()
    s = s.copy()
    s["h_star"] = s.cell_id.map(lambda c: rec[c][climate]).astype(float)
    return s


# ----------------------------------------------------------------------------- map projection (Albers, CONUS)
def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0, lat0, p1, p2 = map(np.radians, (lon0, lat0, p1, p2))
    n = 0.5 * (np.sin(p1) + np.sin(p2))
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(c - 2 * n * np.sin(lat0)) / n
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    th = n * (lon - lon0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def _shapes(path, keep=None):
    import shapefile  # pyshp

    r = shapefile.Reader(str(path), encoding="utf-8")
    fields = [f[0] for f in r.fields[1:]]
    out = []
    for sr in r.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        if keep is not None and not keep(rec):
            continue
        pts = np.asarray(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        out.append([pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)])
    return out


def clip_poly(p, box):
    """Sutherland-Hodgman clip of a closed polygon to box = (x0, x1, y0, y1). Clipping here, not by the renderer,
    keeps invisible polygon edges out of the PDF (figure_lint reads every drawn segment)."""
    x0, x1, y0, y1 = box
    out = [tuple(v) for v in p]
    for inside, inter in (
            (lambda v: v[0] >= x0, lambda a, b: (x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]))),
            (lambda v: v[0] <= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
            (lambda v: v[1] >= y0, lambda a, b: (a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0)),
            (lambda v: v[1] <= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1))):
        if not out:
            break
        inp, out = out, []
        prev = inp[-1]
        for cur in inp:
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
            prev = cur
    return np.asarray(out) if len(out) >= 3 else None


def draw_conus(ax, box=(-0.40, 0.40, -0.235, 0.215), lw=0.3, fc="#F2F2F2", ec="#BDBDBD"):
    """US lower-48 states (Natural Earth 1:50m admin-1, public domain) in Albers equal-area, clipped to box."""
    from matplotlib.patches import Polygon

    states = _shapes(NE / "ne_50m_admin_1_states_provinces_lakes" / "ne_50m_admin_1_states_provinces_lakes.shp",
                     keep=lambda r: r.get("adm0_a3") == "USA" and r.get("name") not in ("Alaska", "Hawaii"))
    for parts in states:
        for p in parts:
            if len(p) < 3:
                continue
            x, y = albers(p[:, 0], p[:, 1])
            q = clip_poly(np.c_[x, y], box)
            if q is not None:
                ax.add_patch(Polygon(q, closed=True, fc=fc, ec=ec, lw=lw, zorder=0))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.axis("off")


def draw_world(ax, box=(-25.0, 155.0, -45.0, 72.0), lw=0.25, fc="#F2F2F2", ec="#BDBDBD"):
    """Countries (Natural Earth 1:110m, public domain), plate carree, clipped to box."""
    from matplotlib.patches import Polygon

    for parts in _shapes(NE / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp",
                         keep=lambda r: r.get("ADM0_A3") != "ATA"):
        for p in parts:
            if len(p) < 3:
                continue
            q = clip_poly(p, box)
            if q is not None:
                ax.add_patch(Polygon(q, closed=True, fc=fc, ec=ec, lw=lw, zorder=0))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.axis("off")


def msize(mw, k=0.045):
    """Marker area (pt^2) proportional to IT capacity."""
    return np.asarray(mw, float) * k
