"""Live demo: peptide-HLA class I stability, with an honest refusal to be trusted.

Run:  .venv/bin/python app.py        -> http://127.0.0.1:7860

Everything slow happens at import (warm()), so the first click is instant.
The model behind this is swapped in at predict.py's SWAP HERE marker.
"""

import io
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image

import ood
import predict

AA = predict.AA
PORT = 7860

# Projected from the back of a room: three colours, nothing else carries meaning.
COLOUR = {"in": "#22c55e", "borderline": "#f59e0b", "out": "#ef4444"}
INK = "#f8fafc"
PAPER = "#0b1120"

EXAMPLES = [
    ("SLYNTVATL", "HLA-A*02:01"),   # HIV gag, the textbook A2 epitope
    ("GILGFVFTL", "HLA-A*02:01"),   # influenza M1
    ("KRWIILGLNK", "HLA-B*27:05"),
]

CSS = f"""
.gradio-container {{ background:{PAPER} !important; max-width:1200px !important; }}
#head {{ color:{INK}; font:600 15px/1.4 system-ui; letter-spacing:.14em;
        text-transform:uppercase; opacity:.55; margin:0 0 2px; }}
#big {{ text-align:center; padding:4px 0 2px; }}
#big .n {{ font:800 128px/1 system-ui; color:{INK}; letter-spacing:-.04em; }}
#big .u {{ font:600 40px/1 system-ui; color:{INK}; opacity:.6; margin-left:10px; }}
#big .band {{ font:500 26px/1.3 system-ui; color:{INK}; opacity:.72; margin-top:6px; }}
#big .ctx {{ font:500 19px/1.3 system-ui; color:{INK}; opacity:.45; margin-top:8px; }}
#flag {{ border-radius:14px; padding:20px 24px; margin:14px 0 4px; }}
#flag .h {{ font:800 40px/1.1 system-ui; letter-spacing:-.01em; }}
#flag .d {{ font:500 21px/1.4 system-ui; margin-top:10px; opacity:.92; }}
#err {{ font:700 26px/1.3 system-ui; color:{COLOUR['out']}; text-align:center;
        padding:40px 0; }}
label, .gr-input, input, select {{ font-size:19px !important; }}
#go {{ font:700 22px/1 system-ui !important; }}
footer {{ display:none !important; }}
"""


# --------------------------------------------------------------------------
# dropdown: measured alleles first, then the ones nobody has ever measured
# --------------------------------------------------------------------------

def allele_choices():
    st = ood.state()
    out = [(a, a) for a in st["measured"]]
    out += [(f"⚠ {a}  — never measured", a) for a in st["unmeasured"]]
    return out


def _fmt(h):
    return f"{h:.2f}" if h < 10 else f"{h:.1f}" if h < 100 else f"{h:.0f}"


def present_number(p, allele, kind):
    """How much of the number do we show when we have just said not to trust it?

    Policy: out of distribution, the number is dimmed and struck through. It
    stays legible -- a judge can still see what the model said, and suppressing
    it entirely would look like the demo had failed -- but it can no longer be
    read as a claim. The flag below it then supplies the reason.

    `kind` is 'in' | 'borderline' | 'out'. Returns (number, unit, css).
    """
    if kind == "out":
        return (_fmt(p.thalf), "hours",
                "opacity:.35;text-decoration:line-through;"
                f"text-decoration-color:{COLOUR['out']};"
                "text-decoration-thickness:7px")
    if kind == "borderline":
        return _fmt(p.thalf), "hours", "opacity:.72"
    return _fmt(p.thalf), "hours", ""


def big_html(p, allele):
    st = ood.state()
    if ood.is_measured(allele):
        ctx = (f"{allele} &middot; {st['support'][allele]:,} measurements in training "
               f"&middot; {st['censored'][allele]:.0%} below assay resolution")
    else:
        ctx = f"{allele} &middot; <b>zero</b> measurements exist"
    n, unit, extra = present_number(p, allele, ood.verdict(allele)[0])
    return (f"<div id='big'><span class='n' style='{extra}'>{n}</span>"
            f"<span class='u'>{unit}</span>"
            f"<div class='band'>seed ensemble {_fmt(p.lo)} &ndash; {_fmt(p.hi)} h</div>"
            f"<div class='ctx'>{ctx}</div></div>")


def flag_html(allele):
    kind, head, detail = ood.verdict(allele)
    c = COLOUR[kind]
    bg = "#2a1116" if kind == "out" else "#2a2011" if kind == "borderline" else "#0d2318"
    return (f"<div id='flag' style='background:{bg};border:3px solid {c}'>"
            f"<div class='h' style='color:{c}'>{head}</div>"
            f"<div class='d' style='color:{INK}'>{detail}</div></div>")


# --------------------------------------------------------------------------
# single prediction
# --------------------------------------------------------------------------

def run(pep, allele):
    ok, msg = predict.valid_peptide(pep)
    if not ok:
        return (f"<div id='err'>{msg}</div>", "", gr.update(visible=False))
    pep = pep.strip().upper()
    p = predict.predict_one(pep, allele)
    return (big_html(p, allele), flag_html(allele), gr.update(visible=True))


# --------------------------------------------------------------------------
# single-point mutant scan: all 19 substitutions at each of the 9 positions
# --------------------------------------------------------------------------

def scan(pep, allele):
    ok, _ = predict.valid_peptide(pep)
    if not ok:
        return None, ""
    pep = pep.strip().upper()
    variants = [pep[:i] + a + pep[i + 1:] for i in range(9) for a in AA]
    preds = predict.predict_batch(variants, allele)
    M = np.array([p.thalf for p in preds]).reshape(9, 20)
    wt = predict.predict_one(pep, allele).thalf

    # best single mutation, excluding the wild-type residue at each position
    best = None
    for i in range(9):
        for j, a in enumerate(AA):
            if a == pep[i]:
                continue
            if best is None or M[i, j] > best[0]:
                best = (M[i, j], i, a)
    gain, bi, ba = best
    head = (f"<div style='font:700 30px/1.35 system-ui;color:{INK};text-align:center;"
            f"padding:6px 0 2px'>Best single mutation: "
            f"<span style='color:{COLOUR['in']}'>P{bi + 1} {pep[bi]}→{ba}</span>"
            f" &nbsp;{_fmt(wt)} &rarr; {_fmt(gain)} h"
            f" <span style='opacity:.6'>({gain / max(wt, 1e-9):.1f}×)</span></div>")

    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    im = ax.imshow(np.log10(np.clip(M, 1e-3, None)), aspect="auto", cmap="magma")
    ax.set_xticks(range(20), list(AA), color=INK, fontsize=15)
    ax.set_yticks(range(9), [f"P{i + 1}  {pep[i]}" for i in range(9)],
                  color=INK, fontsize=15)
    for i in range(9):  # ring the wild-type residue
        ax.add_patch(plt.Rectangle((AA.index(pep[i]) - .5, i - .5), 1, 1,
                                   fill=False, ec=INK, lw=2.5))
    ax.add_patch(plt.Rectangle((AA.index(ba) - .5, bi - .5), 1, 1,
                               fill=False, ec=COLOUR["in"], lw=4))
    ax.set_xlabel("substituted residue", color=INK, fontsize=17, labelpad=10)
    cb = fig.colorbar(im, ax=ax, pad=.015)
    cb.set_label("predicted half-life  log₁₀ h", color=INK, fontsize=15)
    cb.ax.tick_params(colors=INK, labelsize=13)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf), head


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

def build():
    st = ood.state()
    note = "" if st["basis"] == "groove" else (
        "  ·  allele distances on the provisional NAME basis "
        "— sequences still downloading")
    stub = "  ·  PLACEHOLDER MODEL" if predict.IS_STUB else ""

    with gr.Blocks(title="Peptide-HLA stability") as app:
        gr.HTML(f"<div id='head'>Peptide&ndash;HLA class I stability &nbsp;&middot;&nbsp; "
                f"predicted half-life{stub}{note}</div>")
        with gr.Row():
            pep = gr.Textbox(label="9-mer peptide", value=EXAMPLES[0][0],
                             max_lines=1, scale=3)
            allele = gr.Dropdown(allele_choices(), label="HLA allele",
                                 value=EXAMPLES[0][1], scale=3)
            go = gr.Button("Predict", variant="primary", elem_id="go", scale=1)

        big = gr.HTML()
        flag = gr.HTML()

        with gr.Accordion("Mutant scan — which substitution stabilises it?",
                          open=False, visible=False) as acc:
            scan_head = gr.HTML()
            heat = gr.Image(label="", show_label=False, height=560)
            gr.Button("Run scan (171 predictions)", variant="secondary").click(
                scan, [pep, allele], [heat, scan_head])

        ex = [[p, a] for p, a in EXAMPLES]
        if st["unmeasured"]:
            # the money example: a real patient allele nobody ever measured
            ex.append(["SLYNTVATL", st["unmeasured"][0]])
        gr.Examples(ex, [pep, allele])

        for ev in (go.click, pep.submit, allele.change):
            ev(run, [pep, allele], [big, flag, acc])
        app.load(run, [pep, allele], [big, flag, acc])
    return app


def free_port(start=PORT, tries=20):
    """First free port at or after `start`.

    A stale instance holding 7860 should not be the thing that ends the demo,
    so we step past it and print where we actually landed.
    """
    for p in range(start, start + tries):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise OSError(f"no free port in {start}-{start + tries - 1}")


if __name__ == "__main__":
    predict.warm()
    port = free_port()
    if port != PORT:
        print(f"!! port {PORT} busy (stale instance?) -- using {port}", flush=True)
    print(f"\n   ->  http://127.0.0.1:{port}\n", flush=True)
    build().launch(server_name="127.0.0.1", server_port=port,
                   css=CSS, theme=gr.themes.Base(), quiet=True)
