"""Build the static documentation site in docs/ (served by GitHub Pages from /docs).

    python scripts/build_docs.py

The API reference is generated from the live signatures and docstrings of the
public objects in `ssmix2d.__all__`; the other pages are written here.
"""
import html
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ssmix2d  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
REPO = "https://github.com/AngshulMajumdar/State-Space-Mixer-2D"
SPACE = "https://huggingface.co/spaces/Angshul/ssmix2d-audio-embedder"
ONE_D = "https://github.com/AngshulMajumdar/State-Space-Mixer"

PAGES = [("index.html", "Overview"), ("article.html", "Article"), ("method.html", "Method"), ("api.html", "API"),
         ("benchmarks.html", "Benchmarks")]

CSS = """
:root { --ink:#1b1f24; --muted:#5b6470; --line:#e3e6ea; --bg:#ffffff; --soft:#f6f7f9; --accent:#0b6e69; --code:#f3f5f7; }
* { box-sizing: border-box; }
html { font-size: 16px; }
body { margin:0; color:var(--ink); background:var(--bg);
       font: 1rem/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
header { border-bottom:1px solid var(--line); background:var(--bg); position:sticky; top:0; z-index:10; }
.bar { max-width:920px; margin:0 auto; padding:0.9rem 1.25rem; display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap; }
.brand { font-weight:700; letter-spacing:0.01em; color:var(--ink); text-decoration:none; font-size:1.1rem; }
.brand span { color:var(--accent); }
nav { display:flex; flex:1; flex-wrap:wrap; align-items:center; }
nav a { color:var(--muted); text-decoration:none; margin-right:1.1rem; font-size:0.95rem; }
nav a.on, nav a:hover { color:var(--accent); }
nav a.gh { margin-left:auto; }
main { max-width:920px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1 { font-size:2rem; line-height:1.25; margin:0.5rem 0 0.75rem; }
h2 { font-size:1.35rem; margin:2.4rem 0 0.6rem; padding-top:0.4rem; border-top:1px solid var(--line); }
h3 { font-size:1.08rem; margin:1.6rem 0 0.4rem; }
p, li { color:var(--ink); }
.lead { font-size:1.12rem; color:var(--muted); max-width:44rem; }
a { color:var(--accent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:0.88rem; }
code { background:var(--code); padding:0.1rem 0.3rem; border-radius:4px; }
pre { background:var(--code); padding:0.9rem 1rem; border-radius:8px; overflow-x:auto; line-height:1.5; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; margin:0.8rem 0 1.2rem; font-size:0.95rem; }
th, td { text-align:left; padding:0.45rem 0.6rem; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; background:var(--soft); }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
tr.ours td { font-weight:600; background:#eef7f6; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:0.9rem; margin:1.4rem 0; }
.card { border:1px solid var(--line); border-radius:10px; padding:0.9rem 1rem; background:var(--soft); }
.card b { display:block; margin-bottom:0.2rem; }
.card .big { font-size:1.6rem; font-weight:700; color:var(--accent); }
.note { color:var(--muted); font-size:0.92rem; }
.api-item { border:1px solid var(--line); border-radius:10px; padding:0.2rem 1.1rem 0.8rem; margin:1.2rem 0; }
.api-item h3 { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:1rem; }
.sig { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:0.85rem; color:var(--muted);
       white-space:pre-wrap; word-break:break-word; margin:0 0 0.6rem; }
.doc { white-space:pre-wrap; background:none; padding:0; margin:0.2rem 0 0.6rem; font-size:0.86rem; }
.method { margin-left:0.6rem; padding-left:0.9rem; border-left:3px solid var(--line); }
.toc { columns:2; font-size:0.95rem; }
.paper .authors { color:var(--muted); margin:-0.2rem 0 1.2rem; }
.abstract { background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:1rem 1.2rem; margin:1rem 0 1.5rem; }
.caption { color:var(--muted); font-size:0.9rem; margin-top:-0.6rem; }
ol.refs li { font-size:0.92rem; margin-bottom:0.35rem; }
footer { border-top:1px solid var(--line); color:var(--muted); font-size:0.85rem; }
footer .bar { justify-content:space-between; }
@media (max-width:640px) { h1 { font-size:1.6rem; } .toc { columns:1; } nav a.gh { margin-left:0; } }
"""

MATHJAX = ('<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]]}};</script>\n'
           '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>')


def page(fname, title, body, math=False):
    nav = "".join(f'<a href="{f}"{" class=on" if f == fname else ""}>{t}</a>' for f, t in PAGES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · ssmix2d</title>
<meta name="description" content="ssmix2d: training-free time-frequency transformers on Daubechies wavelets.">
<link rel="stylesheet" href="style.css">
{MATHJAX if math else ""}
</head>
<body>
<header><div class="bar"><a class="brand" href="index.html">ss<span>mix</span>2d</a>
<nav>{nav}<a class="gh" href="{REPO}">GitHub</a></nav></div></header>
<main>
{body}
</main>
<footer><div class="bar"><span>ssmix2d {ssmix2d.__version__} · Angshul Majumdar</span><a href="{REPO}">{REPO.replace("https://", "")}</a></div></footer>
</body>
</html>
"""


# --------------------------------------------------------------------- API page
def sig(obj, name):
    try:
        try:
            s = str(inspect.signature(obj, eval_str=True))
        except NameError:
            s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        s = "(...)"
    return f"{name}{s}"


def api_body():
    items, toc = [], []
    for name in ssmix2d.__all__:
        obj = getattr(ssmix2d, name)
        toc.append(f'<li><a href="#{name}"><code>{name}</code></a></li>')
        kind = "class" if inspect.isclass(obj) else "function"
        doc = inspect.getdoc(obj) or ""
        block = [f'<div class="api-item" id="{name}"><h3>{kind} {name}</h3>',
                 f'<p class="sig">{html.escape(sig(obj, "ssmix2d." + name))}</p>',
                 f'<pre class="doc">{html.escape(doc)}</pre>']
        if inspect.isclass(obj):
            for mname, f in vars(obj).items():
                if mname.startswith("_") or not callable(f) or mname == "extra_repr":
                    continue
                mdoc = inspect.getdoc(f) or ""
                block.append(f'<div class="method"><p class="sig"><b>{html.escape(mname)}</b>'
                             f'{html.escape(sig(f, "")[0:])}</p><pre class="doc">{html.escape(mdoc)}</pre></div>')
        block.append("</div>")
        items.append("\n".join(block))
    return (f"<h1>API reference</h1>\n<p class=lead>Everything below is importable from <code>ssmix2d</code>. "
            f"All modules are parameter-free <code>torch.nn.Module</code>s that run on CPU or CUDA. "
            f"<code>CrossScaleMixer2D</code> takes <code>(batch, time, frequency)</code>; "
            f"<code>ScaleSpaceMixer</code> takes <code>(batch, tokens, channels)</code>.</p>\n<ul class=toc>{''.join(toc)}</ul>\n"
            + "\n".join(items))


# -------------------------------------------------------------------- content
R = {}  # benchmark numbers, filled from benchmarks/results.json (see main)

NAMES = {
    "ssmix2d": ("ssmix2d AudioEmbedder2D (time × frequency)", "none"),
    "ssmix": ("ssmix AudioEmbedder (1-D, time only)", "none"),
    "handcrafted_all": ("All eight hand-crafted sets, concatenated", "none"),
    "compare": ("openSMILE ComParE 2016 (6,373 functionals)", "none"),
    "egemaps": ("openSMILE eGeMAPS v02 (88 functionals)", "none"),
    "librosa": ("librosa ESC bundle (MFCC, Δ, chroma, contrast, tonnetz, spectral)", "none"),
    "scattering": ("Wavelet scattering, orders 0–2 (kymatio, J=9, Q=8)", "none"),
    "gfcc": ("Gammatone cepstral coefficients + Δ", "none"),
    "mfcc": ("MFCC + Δ, mean/std", "none"),
    "logmel": ("Log-mel, mean/std", "none"),
    "cqt": ("Constant-Q spectrum, mean/std", "none"),
    "random_transformer": ("Random-init Transformer (4 layers, d=256)", "none"),
    "vggish": ("VGGish", "YouTube-8M"),
    "wav2vec2": ("wav2vec 2.0 base (all layers, mean-pooled)", "LibriSpeech 960 h"),
    "passt": ("PaSST-S", "AudioSet"),
    "efficientat_mn10": ("EfficientAT MobileNet mn10", "AudioSet"),
    "panns": ("PANNs CNN14", "AudioSet"),
}
OURS = "ssmix2d"


def esc(n):
    return R["esc50"][n]


def wins(a, b):
    return sum(x > y for x, y in zip(esc(a)["folds"], esc(b)["folds"]))


def ties(a, b):
    return sum(x == y for x, y in zip(esc(a)["folds"], esc(b)["folds"]))


def won(n):
    w, t = wins(OURS, n), ties(OURS, n)
    return f"{w}/5" + (f" ({t} tied)" if t else "")


def esc_rows():
    order = sorted(R["esc50"], key=lambda n: -esc(n)["mean"])
    return "".join(f'<tr{" class=ours" if n == OURS else ""}><td>{NAMES[n][0]}</td><td>{NAMES[n][1]}</td>'
                   f'<td class=num>{esc(n)["mean"]} ± {esc(n)["std"]}</td></tr>' for n in order)


def fold_rows(others):
    rows = [(OURS, " class=ours")] + [(o, "") for o in others]
    return "".join(f'<tr{c}><td>{NAMES[n][0].split(" (")[0]}</td>' + "".join(f"<td class=num>{v}</td>" for v in esc(n)["folds"])
                   + (f"<td class=num>{won(n)}</td>" if n != OURS else "<td class=num>—</td>") + "</tr>"
                   for n, c in rows)


def fsdd_rows():
    order = sorted(R["fsdd"], key=lambda n: -R["fsdd"][n])
    return "".join(f'<tr{" class=ours" if n == OURS else ""}><td>{NAMES[n][0]}</td><td>{NAMES[n][1]}</td>'
                   f'<td class=num>{R["fsdd"][n]}</td></tr>' for n in order)


def speaker_rows():
    S = R["fsdd_speakers"]
    return "".join(f'<tr{" class=ours" if n == OURS else ""}><td>{NAMES[n][0]}</td><td class=num>{S[n]["mean"]}</td></tr>'
                   for n in (OURS, "ssmix"))


def speed_rows():
    return "".join(f"<tr><td class=num>{T:,}</td><td class=num>{T / 100:g} s</td><td class=num>{a}</td>"
                   f"<td class=num>{b:,}</td><td class=num>{c:,}</td><td class=num>{c / b:.1f}×</td></tr>"
                   for T, a, b, c in R["speed"]["rows"])


def gain_speakers():
    S = R["fsdd_speakers"]
    return round(S[OURS]["mean"] - S["ssmix"]["mean"], 1)


def index_body():
    E = R["esc50"]
    return f"""
<h1>Training-free time–frequency transformers on Daubechies wavelets</h1>
<p class=lead>ssmix2d mixes audio jointly across time and frequency with exact multiscale operators: a separable
2-D Daubechies tight frame over (time, mel), and a cross-scale attention mixer in which every coefficient attends to
its siblings, parents and children along both axes. Nothing is trained.</p>

<div class=cards>
<div class=card><b>ESC-50, linear probe</b><span class=big>{E[OURS]['mean']}%</span><br><span class=note>no training · ahead of the 1-D ssmix, the hand-crafted sets, ComParE, VGGish and wav2vec 2.0</span></div>
<div class=card><b>Unseen speakers</b><span class=big>+{gain_speakers()}</span><br><span class=note>points over the 1-D ssmix on FSDD, leave one speaker out</span></div>
<div class=card><b>Zero parameters</b><span class=big>0</span><br><span class=note>no weights, no data, no fine-tuning; exact reconstruction (error below 10<sup>−14</sup>)</span></div>
</div>

<h2>Install</h2>
<pre><code>pip install git+{REPO}</code></pre>
<p>Try it in the browser: <a href="{SPACE}">live demo on Hugging Face Spaces</a>. The 1-D predecessor is
<a href="{ONE_D}">State-Space-Mixer</a>.</p>
<p>Requires Python ≥ 3.9, PyTorch ≥ 2.0, torchaudio, NumPy and mpmath.</p>

<h2>Embed audio in three lines</h2>
<pre><code>from ssmix2d import AudioEmbedder2D

emb = AudioEmbedder2D()                                # no weights to download
E = emb.embed(waves, sample_rate=44100, length=5.0)    # list of 1-D arrays -> (N, D) numpy</code></pre>
<p>Inputs are resampled to 32 kHz. With <code>length</code> every clip is padded or cut to the same
duration, so all embeddings share one dimension. Use the embeddings with any linear model, k-NN or clustering.</p>

<h2>Use the 2-D mixer as a layer</h2>
<pre><code>import torch
from ssmix2d import CrossScaleMixer2D

X = torch.randn(8, 500, 128)                           # (batch, time, frequency)
mixer = CrossScaleMixer2D(order=4, freq_order=2, window=2, tau=0.1)
Y = mixer(X)                                           # same shape, cross-scale attention on both axes

W, plan = mixer.coefficients(X)                        # (8, S, R, nt, nf): every time scale x every frequency scale
assert torch.allclose(mixer.synthesize(W, plan), X, atol=1e-4)   # exact round trip</code></pre>

<h2>What is inside</h2>
<table>
<tr><th>Object</th><th>Role</th></tr>
<tr><td><a href="api.html#AudioEmbedder2D"><code>AudioEmbedder2D</code></a></td><td>Training-free audio transformer: log-mel → 2-D cross-scale mixer blocks → pooled scale-space readout.</td></tr>
<tr><td><a href="api.html#CrossScaleMixer2D"><code>CrossScaleMixer2D</code></a></td><td>Cross-scale attention on the separable 2-D Daubechies tight frame; shift-equivariant in time.</td></tr>
<tr><td><a href="api.html#ScaleSpaceMixer"><code>ScaleSpaceMixer</code></a></td><td>FFT-batched 1-D Daubechies tight frame for any length: the building block of each axis.</td></tr>
<tr><td><a href="api.html#daubechies"><code>daubechies</code></a></td><td>Daubechies filters of any order, built from scratch by spectral factorisation.</td></tr>
</table>
<p>The full write-up is the <a href="article.html">article</a>; the construction is on <a href="method.html">Method</a>
and the measurements on <a href="benchmarks.html">Benchmarks</a>.</p>
"""


def method_body():
    return r"""
<h1>Method</h1>
<p class=lead>Three exact building blocks: Daubechies filters of controllable support on each axis, a separable
tight frame over time and frequency, and a parameter-free cross-scale attention that moves along both axes.</p>

<h2>1. Daubechies filters of any order</h2>
<p>Order \(N\) gives \(N\) vanishing moments and support \(2N-1\). The lowpass filter is obtained from
\(|H(\omega)|^2 = 2\cos^{2N}(\omega/2)\,P(\sin^2(\omega/2))\) with
\(P(y)=\sum_{k\lt N}\binom{N-1+k}{k}y^k\), by spectral factorisation at high precision. The highpass is
\(g[n] = (-1)^n h[L-1-n]\). The time axis and the frequency axis each get their own order.</p>

<h2>2. A separable tight frame over time and frequency</h2>
<p>Along one axis, the undecimated, delay-aligned Daubechies frame has band responses \(\Psi_s(\omega)\),
\(s=0,\dots,J\), with \(\sum_s|\Psi_s(\omega)|^2=1\) (the sum telescopes because \(|\tilde H|^2+|\tilde G|^2=1\)).
For a field \(X(t,f)\) the 2-D bands are the products</p>
<p>\[ \Psi_{s,r}(\omega_t,\omega_f)=\Psi_s(\omega_t)\,\Phi_r(\omega_f),\qquad
\sum_{s,r}|\Psi_{s,r}|^2=\Big(\sum_s|\Psi_s|^2\Big)\Big(\sum_r|\Phi_r|^2\Big)=1 , \]</p>
<p>so the 2-D frame is Parseval-tight and its adjoint is the exact inverse. Every time scale is paired with every
frequency scale, so a fine time scale can meet a coarse frequency scale (a transient with a broad spectrum) and
vice versa (a slowly varying, narrow-band partial). Both axes are computed with batched FFTs on the DFT grid, at a
cost that does not depend on the filter supports.</p>

<h2>3. Cross-scale attention along both axes</h2>
<p>Tokens are coefficient vectors over \(g\) consecutive frequency bins: \(w_{s,r}(t,b)\in\mathbb R^g\) for band
\((s,r)\), instant \(t\) and bin group \(b\). Token \((s,r,t,b)\) attends to</p>
<p>\[ \mathcal N=\{(s',r,\,t+o\,2^{\ell(s')},\,b) : s'\in\{s-1,s,s+1\}\}\ \cup\
\{(s,r',\,t,\,b\text{ shifted by }o\,2^{\ell(r')}\text{ bins}) : r'\in\{r-1,r,r+1\}\},\quad |o|\le k , \]</p>
<p>its siblings, parents and children along time and along frequency (\(\ell\) is the dyadic level of a band). With
per-token cosine weights</p>
<p>\[ a_n=\operatorname{softmax}_{n\in\mathcal N}\frac{\langle\hat w_{s,r}(t,b),\hat w_n\rangle}{\tau},\qquad
w'_{s,r}(t,b)=\sum_{n\in\mathcal N}a_n\,w_n , \]</p>
<p>followed by the exact 2-D synthesis. Every key is normalised over its own bins after shifting, so frequency
shifts that are not multiples of \(g\) are exact. The operator has no parameters. It is shift-equivariant in time
(and in frequency for shifts by multiples of \(g\)) and 1-homogeneous, and it tends to the identity as \(\tau\to0\).
With \(k=2\) a token has at most \(1+14+14=29\) keys. Memory is \(O(SR\,TF)\) and work \(O(29\,SR\,TF)\):
linear in duration.</p>

<h2>4. Audio embedder</h2>
<p><code>AudioEmbedder2D</code> runs at 32 kHz with 128 mel bands (25 ms window, 10 ms hop) and per-clip
normalisation. Two 2-D cross-scale mixer blocks follow: db4 along time, db2 along mel, 8-bin tokens, \(k=2\),
\(\tau=0.1\). For each block, the readout pools over time:</p>
<ul>
<li>the magnitude of every time × frequency band, per mel bin, before and after mixing;</li>
<li>the mean attention weight of every link and bin group: how strongly each band couples to its neighbours
in time scale and in frequency scale;</li>
<li>the state statistics.</li>
</ul>
<p>The mean log-mel envelope is appended once, at the end. All settings are inherited unchanged from the 1-D
ssmix, whose configuration was chosen on development splits only.</p>
"""


def benchmarks_body():
    S = R["speed"]
    r = S["rows"][-1]
    return f"""
<h1>Benchmarks</h1>
<p class=lead>Frozen embeddings and one linear probe for every model. ssmix2d is never trained; the probe is the only
fitted component.</p>

<h2>Protocol</h2>
<ul>
<li><b>Probe:</b> StandardScaler, then multinomial logistic regression. C is chosen from
{{10<sup>−3</sup>, 10<sup>−2</sup>, 10<sup>−1</sup>, 1}} by 3-fold cross-validation inside each training split.
The probe is identical for every model.</li>
<li><b>ESC-50:</b> 2,000 five-second clips in 50 environmental classes, official 5-fold cross-validation.</li>
<li><b>FSDD:</b> Free Spoken Digit Dataset, 3,000 recordings of the digits 0–9: the official split (takes 0–4 form
the test set), and leave-one-speaker-out over its six speakers (train on five, test on the unseen one).</li>
<li>ssmix2d inherits every setting of the 1-D ssmix; nothing was tuned for the 2-D model.</li>
<li>Baselines are those of the <a href="{ONE_D}">State-Space-Mixer</a> suite, run on the same prepared audio with the
same probe. Pretrained models use the authors' released weights.</li>
</ul>
<p>Reproduce with the scripts in <a href="{REPO}/tree/main/benchmarks"><code>benchmarks/</code></a>.</p>

<h2>ESC-50 (accuracy %, 5-fold mean ± std)</h2>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy</th></tr>{esc_rows()}</table>
<h3>Per fold</h3>
<table><tr><th>Model</th><th class=num>Fold 1</th><th class=num>Fold 2</th><th class=num>Fold 3</th><th class=num>Fold 4</th>
<th class=num>Fold 5</th><th class=num>Folds won by ssmix2d</th></tr>{fold_rows(["wav2vec2", "compare", "vggish"])}</table>

<h2>FSDD (accuracy %, official split)</h2>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy</th></tr>{fsdd_rows()}</table>

<h2>Unseen speakers (FSDD, leave one speaker out, mean accuracy %)</h2>
<table><tr><th>Model</th><th class=num>Accuracy</th></tr>{speaker_rows()}</table>
<p>Mixing along frequency adds {gain_speakers()} points on speakers the probe has never heard: pitch and formant
differences between speakers are shifts along the mel axis, which the 2-D mixer couples across frequency scales.</p>

<h2>Speed</h2>
<p>CPU with 2 threads, batch 1, 128 mel bins, fp32; times in milliseconds. Softmax attention is parameter-free
self-attention over the same tokens (every frame × 8-bin group).</p>
<table><tr><th class=num>Frames</th><th class=num>Audio</th><th class=num>2-D frame round trip</th><th class=num>2-D cross-scale mixer</th>
<th class=num>Softmax attention</th><th class=num>Speed-up</th></tr>{speed_rows()}</table>
<ul>
<li>The mixer's cost grows linearly with duration, attention's quadratically: at {r[0]:,} frames
({r[0] / 100:g} s of audio) the mixer is {r[3] / r[2]:.1f}× faster.</li>
<li>The frame's cost is nearly flat in support: {S['db1']} ms with db1, {S['db4']} ms with db4 and {S['db16']} ms with
db16 along time, at 1,024 frames.</li>
<li>Embedding all 2,000 ESC-50 clips (10,000 s of audio) takes {S['esc50_seconds']:,} s on the same 2-thread CPU.</li>
</ul>
<p>Run <code>python benchmarks/speed.py --device cuda</code> for GPU timings.</p>
"""


ARTICLE = r"""
<article class=paper>
<h1>ssmix2d: Training-Free Time–Frequency Transformers for Audio on Daubechies Wavelets</h1>
<p class=authors>Angshul Majumdar · Indraprastha Institute of Information Technology Delhi</p>

<div class=abstract><b>Abstract.</b>
Sound is structured along two axes at once: events unfold in time, and their identity lives in how energy is
arranged across frequency. We present ssmix2d, an audio transformer with no learnable parameters whose token mixer
works on both axes jointly. The representation is a separable Daubechies frame over time and mel frequency, which
pairs every time scale with every frequency scale; it is Parseval-tight, so reconstruction is exact. On this frame,
every coefficient attends to its dyadic siblings, parents and children along time and along frequency through
parameter-free cosine attention. With a linear probe on frozen embeddings, ssmix2d reaches @@ESC@@% on ESC-50,
ahead of its time-only predecessor (@@ESC_1D@@%), the concatenation of eight hand-crafted feature sets
(@@ESC_ALL@@%), openSMILE ComParE (@@ESC_CMP@@%), VGGish (@@ESC_VGG@@%) and wav2vec 2.0 (@@ESC_W2V@@%). On spoken
digits it reaches @@FSDD@@%, and on speakers never seen by the probe it improves on the time-only model by
@@SPK_GAIN@@ points. Nothing is learned, which makes the model a natural starting point for low-resource audio.</div>

<h2>1. Introduction</h2>
<p>A transformer is a token mixer plus a readout. In audio, the mixer is usually learned from large corpora, and
that is exactly what low-resource settings lack. The 1-D ssmix [1] showed that the mixer can instead be an exact
operator: attention over a Daubechies wavelet frame along time, with no parameters. But it mixed only along time.
Each mel channel was a passive coordinate of the token, and frequency structure entered only through the readout.</p>
<p>Frequency is not a passive coordinate. A change of speaker moves formants and pitch along the mel axis; a
transient spreads energy across many bands at once; a harmonic sound is a regular pattern across frequency at
several scales. A mixer that can relate a coefficient to its frequency neighbours, at the right scale, sees these
structures directly. Because Daubechies wavelets are separable, the extension costs nothing in exactness: the
tensor product of two tight frames is a tight frame.</p>
<p><b>Contributions.</b></p>
<ol>
<li>A separable 2-D Daubechies tight frame over time and mel frequency with independent supports on each axis,
computed with batched FFTs and inverted exactly.</li>
<li>A parameter-free 2-D cross-scale attention: every coefficient attends to its siblings, parents and children
along both axes. It is shift-equivariant in time, 1-homogeneous, exactly the identity in its zero-temperature limit,
and linear in duration.</li>
<li>A training-free audio embedder that, with every setting inherited from the 1-D model, improves on it for
environmental sound, spoken digits and unseen speakers, and is ahead of the established hand-crafted and
pretrained baselines under an identical linear probe.</li>
</ol>

<h2>2. Related work</h2>
<p>Multiresolution spectrotemporal analysis [2] models auditory cortex as a bank of filters tuned jointly to
temporal rate and spectral scale. Joint time–frequency scattering [3] builds an invariant representation from
wavelets along both axes of a scalogram. Both are fixed cascades; ssmix2d instead mixes coefficients through
data-dependent attention across scales of both axes. Separable 2-D wavelet frames go back to Mallat's
multiresolution analysis [4]. FNet [5] and WaveMix [6] replace attention with Fourier and wavelet operators but keep
learned layers. The standard training-free baselines are hand-crafted descriptors, from MFCCs to openSMILE eGeMAPS
[7] and ComParE [8]; the standard learned ones are pretrained embedders such as VGGish [9], wav2vec 2.0 [10],
PANNs [11], PaSST [12] and EfficientAT [13].</p>

<h2>3. Method</h2>
<h3>3.1 Daubechies filters</h3>
<p>Order \(N\) gives \(N\) vanishing moments and support \(2N-1\) [14]. The lowpass filter comes from
\(|H(\omega)|^2 = 2\cos^{2N}(\omega/2)\,P(\sin^2(\omega/2))\), \(P(y)=\sum_{k\lt N}\binom{N-1+k}{k}y^k\), by
spectral factorisation with high-precision roots; the highpass is \(g[n]=(-1)^n h[L-1-n]\). The time and frequency
axes have independent orders.</p>

<h3>3.2 The separable tight frame</h3>
<p>Along one axis, the undecimated transform has band responses</p>
<p>\[ D_j(\omega)=\tilde G(2^{j-1}\omega)\prod_{k\lt j-1}\tilde H(2^k\omega),\qquad
A_J(\omega)=\prod_{k\lt J}\tilde H(2^k\omega),\qquad \tilde H=H/\sqrt2,\ \tilde G=G/\sqrt2 , \]</p>
<p>and \(\sum_s|\Psi_s(\omega)|^2=1\) by telescoping. Each band is shifted by its integer group delay (a pure phase),
so an index refers to the same position at every scale. For a field \(X(t,f)\), with \(\Psi_s\) along time
(\(J_t\) levels) and \(\Phi_r\) along frequency (\(J_f\) levels), the bands are</p>
<p>\[ W_{s,r}=\mathcal F^{-1}\big[\Psi_s(\omega_t)\,\Phi_r(\omega_f)\,\hat X\big],\qquad
\sum_{s=0}^{J_t}\sum_{r=0}^{J_f}|\Psi_s(\omega_t)\,\Phi_r(\omega_f)|^2=1 . \]</p>
<p><b>Proposition 1.</b> The analysis operator is an isometry, \(\sum_{s,r}\|W_{s,r}\|^2=\|X\|^2\), and its adjoint
\(X=\sum_{s,r}\mathcal F^{-1}[\overline{\Psi_s\Phi_r}\,\hat W_{s,r}]\) is its exact inverse. <i>Proof.</i> By
Parseval, \(\sum_{s,r}\|W_{s,r}\|^2=\sum_{\omega}|\hat X|^2\sum_{s,r}|\Psi_s\Phi_r|^2=\|X\|^2\); applying the adjoint
multiplies \(\hat X\) by the same sum, which is 1. ∎ Boundaries are handled by reflection padding to an
FFT-friendly length on each axis, and the result is cropped after synthesis.</p>

<h3>3.3 Cross-scale attention along both axes</h3>
<p>Tokens are coefficient vectors over \(g\) consecutive frequency bins. Token \((s,r,t,b)\) attends to the
neighbourhood made of (i) bands \(s'\in\{s-1,s,s+1\}\) of the same frequency scale at time offsets
\(o\,2^{\ell(s')}\), and (ii) bands \(r'\in\{r-1,r,r+1\}\) of the same time scale at frequency offsets
\(o\,2^{\ell(r')}\) bins, with \(|o|\le k\). These are its siblings, parents and children along each axis, on the
dyadic grid of the decimated tree but without decimation. With per-token cosine scores at temperature \(\tau\),</p>
<p>\[ a_n=\operatorname{softmax}_{n\in\mathcal N}\frac{\langle\hat w_{s,r}(t,b),\hat w_n\rangle}{\tau},\qquad
w'_{s,r}(t,b)=\sum_{n\in\mathcal N}a_n\,w_n , \]</p>
<p>followed by the exact synthesis of Proposition 1.</p>
<p><b>Proposition 2.</b> The mixer has no parameters and (a) is 1-homogeneous, \(M(cX)=cM(X)\) for \(c\gt0\);
(b) commutes with circular shifts in time, and with circular shifts in frequency by multiples of \(g\);
(c) tends to the identity as \(\tau\to0\) at every token with no other key parallel to it; (d) is a convex combination per token, so
\(\|w'\|\le\max_{n}\|w_n\|\). <i>Proof.</i> (a) Cosine scores are scale-invariant and values are linear.
(b) The frame is shift-equivariant, and the neighbourhood is defined by relative offsets. (c) Each token has cosine
1 with itself and strictly less with every non-parallel key, so the softmax concentrates on itself. (d) The weights
are non-negative and sum to one. ∎</p>
<p>With \(k=2\) a token has at most 29 keys. Keys are never materialised as a separate tensor; key norms after a
frequency shift come from a sliding-window sum of squares, so arbitrary bin shifts are exact. Memory is
\(O(SR\,TF)\) and work \(O(29\,SR\,TF)\), linear in duration.</p>

<h3>3.4 The embedder</h3>
<p>Audio at 32 kHz becomes a 128-band log-mel sequence (25 ms windows, 10 ms hop), normalised per clip. Two 2-D
blocks follow (db4 along time, db2 along mel, 8-bin tokens, \(k=2\), \(\tau=0.1\)). For each block the readout pools
over time: the magnitude of every time × frequency band per mel bin, before and after mixing; the mean attention
weight of every link and bin group; and the state statistics. The mean log-mel envelope is appended. Every setting
is inherited from the 1-D ssmix, whose configuration was chosen on development splits only; nothing was tuned for
the 2-D model.</p>

<h2>4. Experiments</h2>
<h3>4.1 Protocol</h3>
<p>Every model produces a frozen clip embedding, and one probe is used throughout: standardisation followed by
multinomial logistic regression, with C chosen from \(\{10^{-3},10^{-2},10^{-1},1\}\) by 3-fold cross-validation
inside each training split. ESC-50 [15] uses its official 5-fold cross-validation. FSDD [16] uses its official split
and, to measure generalisation to new voices, leave-one-speaker-out over its six speakers. Baselines are those of
the State-Space-Mixer suite, run on the same prepared audio with the same probe.</p>

<h3>4.2 Environmental sound (ESC-50)</h3>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy (%)</th></tr>@@ESC_ROWS@@</table>
<p class=caption>Table 1. ESC-50, 5-fold mean ± std. Hand-crafted sets, ssmix and ssmix2d use no training data.</p>
<table><tr><th>Model</th><th class=num>Fold 1</th><th class=num>Fold 2</th><th class=num>Fold 3</th><th class=num>Fold 4</th>
<th class=num>Fold 5</th><th class=num>Folds won by ssmix2d</th></tr>@@FOLD_ROWS@@</table>
<p class=caption>Table 2. Per-fold accuracy.</p>
<p>Adding the frequency axis to the mixer raises ESC-50 from @@ESC_1D@@% to @@ESC@@% with no new setting. ssmix2d is
ahead of wav2vec 2.0 and openSMILE ComParE on every fold, and of the 7,519-dimensional concatenation of all eight
hand-crafted sets on average.</p>

<h3>4.3 Spoken digits and unseen speakers (FSDD)</h3>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy (%)</th></tr>@@FSDD_ROWS@@</table>
<p class=caption>Table 3. FSDD, official split.</p>
<table><tr><th>Model</th><th class=num>Accuracy (%)</th></tr>@@SPK_ROWS@@</table>
<p class=caption>Table 4. FSDD, leave one speaker out (mean over the six held-out speakers).</p>
<p>On the official split ssmix2d reaches @@FSDD@@%, ahead of the AudioSet-trained PaSST and EfficientAT and of
wav2vec 2.0. On unseen speakers it improves on the time-only model by @@SPK_GAIN@@ points. This is the effect the
design targets: a new voice moves formants and pitch along the mel axis, and the 2-D mixer couples each coefficient
to its neighbours across frequency scales.</p>

<h3>4.4 Exactness and cost</h3>
<p>The 2-D frame reconstructs with error below \(10^{-14}\) in float64, and the implementation matches a direct evaluation of the
attention to \(10^{-12}\). On a 2-thread CPU, the mixer processes @@T_LAST@@ frames (@@SEC_LAST@@ s of audio) in
@@MIX_LAST@@ ms, @@SPEEDUP@@× faster than parameter-free softmax attention over the same tokens, and its cost grows
linearly with duration.</p>

<h2>5. Conclusion</h2>
<p>A transformer's token mixer need not be learned, and it need not be one-dimensional. A separable Daubechies frame
gives an exact time–frequency scale space, and attention defined on its tree along both axes gives an audio embedder
with zero parameters that improves on its time-only predecessor on every benchmark: environmental sound, spoken
digits and, most of all, unseen speakers. It needs no data at all, which makes it a natural starting point for low-resource corpora.</p>

<h2>Reproducibility</h2>
<p>Code, tests and benchmark scripts: <a href="@@REPO@@">@@REPO_SHORT@@</a>. Live demo:
<a href="@@SPACE@@">Hugging Face Space</a>. Every number on this page is read from
<code>benchmarks/results.json</code>.</p>

<h2>References</h2>
<ol class=refs>
<li>A. Majumdar. ssmix: Training-free scale-space transformers for audio on Daubechies wavelets. GitHub: AngshulMajumdar/State-Space-Mixer, 2026.</li>
<li>T. Chi, P. Ru and S. A. Shamma. Multiresolution spectrotemporal analysis of complex sounds. <i>Journal of the Acoustical Society of America</i>, 118(2), 2005.</li>
<li>J. Andén, V. Lostanlen and S. Mallat. Joint time–frequency scattering. <i>IEEE Transactions on Signal Processing</i>, 67(14), 2019.</li>
<li>S. Mallat. A theory for multiresolution signal decomposition: the wavelet representation. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 11(7), 1989.</li>
<li>J. Lee-Thorp, J. Ainslie, I. Eckstein and S. Ontañón. FNet: Mixing tokens with Fourier transforms. <i>NAACL</i>, 2022.</li>
<li>P. Jeevan and A. Sethi. WaveMix: Resource-efficient token mixing for images. arXiv:2203.03689, 2022.</li>
<li>F. Eyben et al. The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing. <i>IEEE Transactions on Affective Computing</i>, 7(2), 2016.</li>
<li>B. Schuller et al. The INTERSPEECH 2016 computational paralinguistics challenge. <i>Interspeech</i>, 2016.</li>
<li>S. Hershey et al. CNN architectures for large-scale audio classification. <i>ICASSP</i>, 2017.</li>
<li>A. Baevski, H. Zhou, A. Mohamed and M. Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. <i>NeurIPS</i>, 2020.</li>
<li>Q. Kong et al. PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. <i>IEEE/ACM TASLP</i>, 28, 2020.</li>
<li>K. Koutini, J. Schlüter, H. Eghbal-zadeh and G. Widmer. Efficient training of audio transformers with patchout. <i>Interspeech</i>, 2022.</li>
<li>F. Schmid, K. Koutini and G. Widmer. Efficient large-scale audio tagging via transformer-to-CNN knowledge distillation. <i>ICASSP</i>, 2023.</li>
<li>I. Daubechies. Orthonormal bases of compactly supported wavelets. <i>Communications on Pure and Applied Mathematics</i>, 41(7), 1988.</li>
<li>K. J. Piczak. ESC: Dataset for environmental sound classification. <i>ACM Multimedia</i>, 2015.</li>
<li>Z. Jackson et al. Free Spoken Digit Dataset. GitHub: Jakobovski/free-spoken-digit-dataset, 2018.</li>
</ol>
</article>
"""


def article_body():
    E = R["esc50"]
    r = R["speed"]["rows"][-1]
    rep = {
        "@@ESC@@": E[OURS]["mean"], "@@ESC_1D@@": E["ssmix"]["mean"], "@@ESC_VGG@@": E["vggish"]["mean"],
        "@@ESC_W2V@@": E["wav2vec2"]["mean"], "@@ESC_CMP@@": E["compare"]["mean"], "@@ESC_ALL@@": E["handcrafted_all"]["mean"],
        "@@FSDD@@": R["fsdd"][OURS], "@@SPK_GAIN@@": gain_speakers(),
        "@@ESC_ROWS@@": esc_rows(), "@@FOLD_ROWS@@": fold_rows(["wav2vec2", "compare", "vggish"]),
        "@@FSDD_ROWS@@": fsdd_rows(), "@@SPK_ROWS@@": speaker_rows(),
        "@@T_LAST@@": f"{r[0]:,}", "@@SEC_LAST@@": f"{r[0] / 100:g}", "@@MIX_LAST@@": f"{r[2]:,}",
        "@@SPEEDUP@@": f"{r[3] / r[2]:.1f}",
        "@@REPO@@": REPO, "@@REPO_SHORT@@": REPO.replace("https://", ""), "@@SPACE@@": SPACE,
    }
    out = ARTICLE
    for k, v in rep.items():
        out = out.replace(k, str(v))
    assert "@@" not in out
    return out


def main():
    import json
    with open(os.path.join(ROOT, "benchmarks", "results.json")) as f:
        R.update(json.load(f))
    # claims made in the text, checked against the numbers
    E = R["esc50"]
    assert E[OURS]["mean"] == max(v["mean"] for v in E.values())
    assert R["fsdd"][OURS] == max(R["fsdd"].values())
    assert wins(OURS, "wav2vec2") == 5 and wins(OURS, "compare") == 5
    assert E[OURS]["mean"] > E["handcrafted_all"]["mean"] and gain_speakers() > 0
    assert all(c > b for _, _, b, c in R["speed"]["rows"])
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "style.css"), "w") as f:
        f.write(CSS.lstrip())
    bodies = {"index.html": (index_body(), False), "article.html": (article_body(), True),
              "method.html": (method_body(), True), "api.html": (api_body(), False),
              "benchmarks.html": (benchmarks_body(), False)}
    for fname, title in PAGES:
        body, math = bodies[fname]
        with open(os.path.join(DOCS, fname), "w") as f:
            f.write(page(fname, title, body, math))
    print("wrote", ", ".join(["style.css"] + [p for p, _ in PAGES]))


if __name__ == "__main__":
    main()
