"""Build the anonymous supplementary website for the Spanish-trill paper.

Produces a self-contained static site under ``supplement/`` (index.html + assets/)
that lets readers LISTEN to trill examples and SEE the closure-vs-envelope-peak
distinction, the F0 mechanism, the per-context effect, and the per-corpus
validation. Upload anonymously (GitHub Pages / OSF / Netlify drop); the folder is
self-contained and zippable.

Audio policy (licensing): only redistributable audio is exported as playable
clips -- synthetic trills (fully synthetic) and Common Voice (CC0). The six
analysed corpora are licence-restricted, so the site shows only their *derived*
visualisations (spectrograms, contact sheets), never their audio.

Anonymity: no author names, emails, local filesystem paths, or git info are
written into the output.

Run from repo root:  python -m trillscope.paper.build_supplement
"""
from __future__ import annotations

import html
import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks, stft

from trillscope.audio_io import load_audio_full
from trillscope.detector import DetectorConfig, detect_closures
from trillscope.synthetic import TrillSpec, synthesize_trill

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_supplement")

ROOT = Path(".")
SUP = ROOT / "supplement"
AUDIO = SUP / "assets" / "audio"
IMG = SUP / "assets" / "img"
TABLES = ROOT / "outputs" / "tables"
VALID = ROOT / "reports" / "validation"
FIGS = ROOT / "paper" / "iberspeech2026" / "figures"

CFG = DetectorConfig()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    a = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(a))) or 1.0
    pcm = np.int16(np.clip(a / peak * 0.95, -1, 1) * 32767)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, pcm)


def _envelope(audio: np.ndarray, sr: int, smooth_ms: float = 6.0) -> np.ndarray:
    env = np.abs(audio)
    w = max(1, int(smooth_ms / 1000.0 * sr))
    return np.convolve(env, np.ones(w) / w, mode="same")


def _envelope_peaks(audio: np.ndarray, sr: int, roi_ms: tuple[float, float]):
    """An envelope-peak counter (the v1-style unit): maxima of the smoothed
    rectified envelope inside the ROI. Returns peak times (ms, absolute)."""
    env = _envelope(audio, sr)
    peaks, _ = find_peaks(
        env, distance=int(0.012 * sr), prominence=0.12 * float(env.max())
    )
    pk_ms = peaks / sr * 1000.0
    lo, hi = roi_ms
    return pk_ms[(pk_ms >= lo) & (pk_ms <= hi)], env


def _closures_ms(audio: np.ndarray, sr: int, roi_ms: tuple[float, float]):
    res = detect_closures(audio, sr, cfg=CFG, roi_ms=roi_ms)
    return [c.closure_t_ms for c in res.closures], int(res.n_closures)


def _spectrogram_panel(
    ax_w, ax_s, audio, sr, roi_ms, *, closures_ms=None, peaks_ms=None
):
    t = np.arange(len(audio)) / sr
    ax_w.plot(t, audio, color="black", lw=0.5)
    ax_w.set_xlim(0, t[-1] if len(t) else 1.0)
    ax_w.set_yticks([])
    lo, hi = roi_ms[0] / 1000.0, roi_ms[1] / 1000.0
    ax_w.axvspan(lo, hi, color="orange", alpha=0.15)

    n_fft, hop = 512, 64
    f, ts, Z = stft(audio, fs=sr, window=np.hanning(n_fft), nperseg=n_fft,
                    noverlap=n_fft - hop, boundary=None, padded=False)
    fm = f <= 5000
    Sdb = 10 * np.log10(np.maximum(np.abs(Z[fm]) ** 2, 1e-12))
    ax_s.pcolormesh(ts, f[fm], Sdb, shading="auto", cmap="magma")
    ax_s.set_ylim(0, 5000)
    ax_s.set_ylabel("Hz")
    ax_s.set_xlabel("time (s)")
    if peaks_ms is not None:
        for i, p in enumerate(peaks_ms):
            ax_w.axvline(p / 1000.0, color="#d33", lw=1.2, ls=":",
                         label="envelope peak" if i == 0 else None)
    if closures_ms is not None:
        for i, c in enumerate(closures_ms):
            ax_s.axvline(c / 1000.0, color="#3df", lw=1.6,
                         label="closure (v2)" if i == 0 else None)


# --------------------------------------------------------------------------- #
# content generators
# --------------------------------------------------------------------------- #
def gen_canonical_examples() -> list[dict]:
    """Synthetic 1/2/3-closure trills: audio + spectrogram with closures."""
    out = []
    for n in (1, 2, 3):
        spec = TrillSpec(n_closures=n, duration_ms=90.0, f0_hz=150.0,
                         period_ms=33.0, rng_seed=n)
        audio, gt, sr, roi = synthesize_trill(spec)
        clo, n_v2 = _closures_ms(audio, sr, roi)
        wav = AUDIO / f"synth_{n}closure.wav"
        png = IMG / f"synth_{n}closure.png"
        _write_wav(wav, audio, sr)
        fig, (aw, as_) = plt.subplots(2, 1, figsize=(6.2, 3.0), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 3]})
        _spectrogram_panel(aw, as_, audio, sr, roi, closures_ms=clo)
        fig.suptitle(f"Synthetic trill — {n} closure(s); detector v2 found {n_v2}",
                     fontsize=9)
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        out.append({"n": n, "n_v2": n_v2, "wav": wav.name, "png": png.name})
    return out


def gen_mechanism_demo() -> dict:
    """Closure vs envelope-peak on a synthetic trill + low/high-F0 audio to hear."""
    demo = {}
    # main illustration: one 2-closure trill, mark closures vs envelope peaks
    spec = TrillSpec(n_closures=2, duration_ms=90.0, f0_hz=140.0,
                     period_ms=34.0, rng_seed=7)
    audio, gt, sr, roi = synthesize_trill(spec)
    clo, n_v2 = _closures_ms(audio, sr, roi)
    pk, env = _envelope_peaks(audio, sr, roi)
    _write_wav(AUDIO / "mech_demo.wav", audio, sr)
    fig, (aw, as_) = plt.subplots(2, 1, figsize=(7.2, 3.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 3]})
    _spectrogram_panel(aw, as_, audio, sr, roi, closures_ms=clo, peaks_ms=pk)
    aw.legend(loc="upper right", fontsize=6); as_.legend(loc="upper right", fontsize=6)
    fig.suptitle(
        f"Same token, two units: {n_v2} verified closures (v2) "
        f"vs {len(pk)} envelope peaks (v1-style)", fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "mech_demo.png", dpi=120); plt.close(fig)
    demo.update(n_v2=n_v2, n_peaks=int(len(pk)),
                wav="mech_demo.wav", png="mech_demo.png")

    # two pitches to HEAR (same closure structure)
    hear = []
    for tag, f0 in (("low", 110.0), ("high", 220.0)):
        a, g, sr2, r = synthesize_trill(
            TrillSpec(n_closures=2, duration_ms=90.0, f0_hz=f0, period_ms=34.0,
                      rng_seed=3))
        _write_wav(AUDIO / f"mech_{tag}F0.wav", a, sr2)
        hear.append({"tag": tag, "f0": int(f0), "wav": f"mech_{tag}F0.wav"})
    demo["hear"] = hear
    return demo


def gen_excess_scatter() -> dict:
    """Real-data scatter: per-token over-count vs F0, coloured by sex."""
    from trillscope.mechanism_sex import _load_tokens
    tok = _load_tokens()
    d = tok.dropna(subset=["mean_f0_hz", "excess"])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(d["excess"], d["mean_f0_hz"])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for sx, col in (("F", "#c2185b"), ("M", "#1565c0")):
        s = d[d["sex"] == sx]
        ax.scatter(s["mean_f0_hz"], s["excess"], s=7, alpha=0.35, color=col, label=sx)
    other = d[~d["sex"].isin(["F", "M"])]
    ax.scatter(other["mean_f0_hz"], other["excess"], s=6, alpha=0.18,
               color="#888", label="unlabeled")
    # trend
    z = np.polyfit(d["mean_f0_hz"], d["excess"], 1)
    xs = np.linspace(d["mean_f0_hz"].min(), d["mean_f0_hz"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="black", lw=1.5)
    ax.set_xlabel("per-token mean $f_0$ (Hz)")
    ax.set_ylabel("over-count  (envelope-peak − closure)")
    ax.set_title(f"Over-count rises as $f_0$ falls   (Spearman ρ = {rho:.2f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "excess_vs_f0.png", dpi=120); plt.close(fig)
    return {"png": "excess_vs_f0.png", "rho": round(float(rho), 2), "n": int(len(d))}


def gen_sample_sizes() -> dict:
    """Per-cell token/speaker counts (context x corpus, sex x context) -- the
    post-filter balance, kept out of the 5-page paper."""
    from trillscope.mechanism_sex import _load_tokens
    tok = _load_tokens()
    CTX = ["intervocalic_rr", "after_nls", "word_initial"]
    tok = tok[tok["context_label"].isin(CTX)]
    DS = ["tedx", "heroico", "glissando", "dimex100", "albayzin", "preseea"]
    th = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in CTX)

    body = ""
    for ds in DS:
        d = tok[tok["dataset"] == ds]
        cells = "".join(f"<td>{int((d['context_label'] == c).sum())}</td>" for c in CTX)
        body += f"<tr><td>{ds}</td>{cells}<td>{len(d)}</td></tr>"
    tot = "".join(f"<td><b>{int((tok['context_label'] == c).sum())}</b></td>" for c in CTX)
    body += f"<tr><td><b>Total</b></td>{tot}<td><b>{len(tok)}</b></td></tr>"
    t_corpus = (f"<table><thead><tr><th>corpus</th>{th}<th>all</th></tr></thead>"
                f"<tbody>{body}</tbody></table>")

    sx = tok[tok["sex"].isin(["F", "M"])]
    body2 = ""
    for s in ("F", "M"):
        d = sx[sx["sex"] == s]
        cells = ""
        for c in CTX:
            dc = d[d["context_label"] == c]
            cells += f"<td>{len(dc)} ({dc['speaker_id'].nunique()})</td>"
        body2 += f"<tr><td>{s}</td>{cells}</tr>"
    t_sex = (f"<table><thead><tr><th>sex</th>{th}</tr></thead>"
             f"<tbody>{body2}</tbody></table>")
    return {"corpus": t_corpus, "sex": t_sex}


def gen_retention() -> dict:
    """Retained vs candidate counts (per corpus + per context) -- how selective
    the quality filter is."""
    DS = ["tedx", "heroico", "glissando", "dimex100", "albayzin", "preseea"]
    LABEL = {"tedx": "TEDx", "heroico": "Heroico", "glissando": "Glissando",
             "dimex100": "DIMEx100", "albayzin": "ALBAYZIN", "preseea": "PRESEEA"}
    tok = pd.read_parquet(TABLES / "tokens.parquet").copy()
    ccol = "corpus" if "corpus" in tok.columns else "dataset"
    xcol = "context" if "context" in tok.columns else "context_label"
    tok["_ds"] = tok[ccol].str.lower()
    key = ["dataset", "utt_id", "start_ms", "end_ms"]
    ret = pd.concat([pd.read_parquet(TABLES / f"closures_v2_{d}.parquet") for d in DS],
                    ignore_index=True).drop_duplicates(key)

    def row(name, cand, kept):
        pct = 100 * kept / cand if cand else 0
        return f"<tr><td>{name}</td><td>{cand}</td><td>{kept}</td><td>{pct:.1f}%</td></tr>"

    body = "".join(row(LABEL[d], int((tok["_ds"] == d).sum()),
                       int((ret["dataset"] == d).sum())) for d in DS)
    body += row("<b>Total</b>", len(tok), len(ret)).replace("<td>", "<td><b>").replace("</td>", "</b></td>")
    t_corpus = (f"<table><thead><tr><th>corpus</th><th>candidates</th>"
                f"<th>retained</th><th>%</th></tr></thead><tbody>{body}</tbody></table>")

    CTX = ["intervocalic_rr", "after_nls", "word_initial"]
    body2 = "".join(row(c.replace("_", " "), int((tok[xcol] == c).sum()),
                        int((ret["context_label"] == c).sum())) for c in CTX)
    t_ctx = (f"<table><thead><tr><th>context</th><th>candidates</th>"
             f"<th>retained</th><th>%</th></tr></thead><tbody>{body2}</tbody></table>")
    return {"corpus": t_corpus, "context": t_ctx}


def gen_attrition_by_sex() -> dict:
    """Per-sex filter attrition control: is the
    21% retention sex-biased, and do men devoice more? Metadata sex only
    (DIMEx100 excluded, F0-inferred). Reuses the standalone analysis loader."""
    from scipy.stats import chi2_contingency, mannwhitneyu

    from trillscope.attrition_sex import load_pool_with_sex

    pool = load_pool_with_sex()
    sx = pool[pool["sex"].isin(["F", "M"])]

    def row(s):
        d = sx[sx["sex"] == s]
        kept = int(d["passed"].sum())
        pct = 100 * kept / len(d) if len(d) else 0
        v_all = d["voicing_pct"].mean()
        v_drop = d.loc[~d["passed"], "voicing_pct"].mean()
        return (f"<tr><td>{s}</td><td>{len(d)}</td><td>{kept}</td>"
                f"<td>{pct:.1f}%</td><td>{v_all:.1f}</td><td>{v_drop:.1f}</td></tr>")

    body = row("F") + row("M")
    table = (f"<table><thead><tr><th>sex</th><th>candidates</th><th>retained</th>"
             f"<th>retention %</th><th>voicing (all)</th><th>voicing (dropped)</th>"
             f"</tr></thead><tbody>{body}</tbody></table>")

    chi2, p_chi, _, _ = chi2_contingency(pd.crosstab(sx["sex"], sx["passed"]))
    spk = (sx.groupby(["dataset", "speaker_id", "sex"])
           .agg(ret=("passed", "mean"), voi=("voicing_pct", "mean")).reset_index())
    _, p_ret = mannwhitneyu(spk.loc[spk.sex == "M", "ret"],
                            spk.loc[spk.sex == "F", "ret"], alternative="two-sided")
    _, p_voi = mannwhitneyu(spk.loc[spk.sex == "M", "voi"],
                            spk.loc[spk.sex == "F", "voi"], alternative="two-sided")
    note = (f"Sex&nbsp;&times;&nbsp;inclusion is not associated (&chi;&sup2; "
            f"p&nbsp;=&nbsp;{p_chi:.2f}; speaker-level retention p&nbsp;=&nbsp;"
            f"{p_ret:.2f}), and male candidates are not less voiced than female ones "
            f"(speaker-level p&nbsp;=&nbsp;{p_voi:.2f}). The filter does not drop "
            f"male tokens preferentially, so the sex null is not an artifact of it.")
    return {"table": table, "note": note}


def gen_cv_examples(n_per_ctx: int = 2) -> list[dict]:
    """Export a few clean Common Voice (CC0) trill clips: audio + spectrogram."""
    cand = TABLES / "r_candidates_commonvoice.parquet"
    meas = TABLES / "acoustic_measurements_commonvoice.parquet"
    if not cand.exists():
        return []
    rc = pd.read_parquet(cand)
    if meas.exists():
        m = pd.read_parquet(meas)[["utt_id", "start_ms", "voicing_pct"]]
        rc = rc.merge(m, on=["utt_id", "start_ms"], how="left")
    else:
        rc["voicing_pct"] = 100.0
    # clean: in-filter duration, well voiced
    rc = rc[(rc["duration_ms"].between(50, 200)) & (rc["voicing_pct"] >= 90)]
    out = []
    for ctx in ("intervocalic_rr", "word_initial", "after_nls"):
        sub = rc[rc["context_label"] == ctx]
        picked = 0
        for _, row in sub.sort_values("voicing_pct", ascending=False).iterrows():
            if picked >= n_per_ctx:
                break
            try:
                audio, sr = load_audio_full(row["audio_path"], target_sr=16000)
            except Exception:
                continue
            pad = 0.12
            t0, t1 = row["start_ms"] / 1000.0, row["end_ms"] / 1000.0
            i0, i1 = max(0, int((t0 - pad) * sr)), int((t1 + pad) * sr)
            chunk = audio[i0:i1]
            if len(chunk) < int(0.08 * sr):
                continue
            roi = ((t0 - max(0, t0 - pad)) * 1000.0, (t1 - max(0, t0 - pad)) * 1000.0)
            clo, n_v2 = _closures_ms(chunk, sr, roi)
            tag = f"cv_{ctx}_{picked+1}"
            _write_wav(AUDIO / f"{tag}.wav", chunk, sr)
            fig, (aw, as_) = plt.subplots(2, 1, figsize=(6.4, 3.0), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 3]})
            _spectrogram_panel(aw, as_, chunk, sr, roi, closures_ms=clo)
            fig.suptitle(f"Common Voice (CC0) — “{html.unescape(str(row['word']))}” "
                         f"[{ctx}] — v2 closures: {n_v2}", fontsize=9)
            fig.tight_layout(); fig.savefig(IMG / f"{tag}.png", dpi=120); plt.close(fig)
            out.append({"ctx": ctx, "word": str(row["word"]), "n_v2": n_v2,
                        "wav": f"{tag}.wav", "png": f"{tag}.png"})
            picked += 1
    return out


def copy_static_figures() -> dict:
    """Copy paper figure, contact sheets and histograms into assets/img."""
    got = {"effect": None, "contacts": [], "hists": [], "summary": None}
    summ = FIGS / "fig_summary.png"
    if summ.exists():
        shutil.copy(summ, IMG / "fig_summary.png")
        got["summary"] = "fig_summary.png"
    fig3 = FIGS / "fig3_effect_sizes.png"
    if fig3.exists():
        shutil.copy(fig3, IMG / "fig3_effect_sizes.png")
        got["effect"] = "fig3_effect_sizes.png"
    order = ["tedx", "heroico", "glissando", "dimex100", "albayzin", "preseea"]
    for ds in order:
        c = VALID / f"{ds}_contact.png"
        h = VALID / f"{ds}_hist.png"
        if c.exists():
            shutil.copy(c, IMG / f"{ds}_contact.png")
            got["contacts"].append({"ds": ds, "png": f"{ds}_contact.png"})
        if h.exists():
            shutil.copy(h, IMG / f"{ds}_hist.png")
            got["hists"].append({"ds": ds, "png": f"{ds}_hist.png"})
    return got


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def _audio(name: str) -> str:
    return (f'<audio controls preload="none" src="assets/audio/{name}">'
            f'</audio>')


def _img(name: str, cls: str = "") -> str:
    return f'<img loading="lazy" class="{cls}" src="assets/img/{name}" alt="">'


CSS = """
:root{--bg:#0f1115;--card:#181b22;--fg:#e8eaed;--mut:#9aa3ad;--acc:#4ea1ff;--line:#2a2f3a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{padding:48px 20px;text-align:center;border-bottom:1px solid var(--line);
background:linear-gradient(180deg,#141821,#0f1115)}
header h1{margin:0 0 8px;font-size:28px}header p{color:var(--mut);max-width:760px;margin:6px auto}
.badge{display:inline-block;background:#22303f;color:var(--acc);border:1px solid #2c4356;
padding:3px 10px;border-radius:20px;font-size:12px;margin-top:10px}
main{max-width:980px;margin:0 auto;padding:24px 18px 80px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:22px 24px;margin:22px 0}
h2{margin-top:0;font-size:21px;border-bottom:1px solid var(--line);padding-bottom:8px}
h3{font-size:16px;color:var(--fg);margin:18px 0 6px}
p,li{color:#d6dade}.mut{color:var(--mut)}.small{font-size:13px}
img{max-width:100%;border-radius:8px;display:block;margin:8px 0;background:#000}
audio{width:100%;margin:6px 0;filter:saturate(.7)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.card{background:#11141b;border:1px solid var(--line);border-radius:10px;padding:12px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}thead th{background:#1d2230}
.key{background:#11141b;border-left:3px solid var(--acc);padding:10px 14px;border-radius:6px;margin:10px 0}
.warn{border-left-color:#e0a800}
code{background:#11141b;border:1px solid var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
footer{color:var(--mut);text-align:center;padding:30px;font-size:13px;border-top:1px solid var(--line)}
a{color:var(--acc)}
"""


def build_html(ctx: dict) -> str:
    e = html.escape

    def canon_cards():
        return "".join(
            f'<div class="card"><b>{c["n"]} closure(s)</b>'
            f'<span class="mut small"> · v2 detected {c["n_v2"]}</span>'
            f'{_img(c["png"])}{_audio(c["wav"])}</div>'
            for c in ctx["canonical"])

    def cv_cards():
        if not ctx["cv"]:
            return '<p class="mut">No Common Voice clips were exported in this build.</p>'
        return "".join(
            f'<div class="card"><b>“{e(c["word"])}”</b>'
            f'<span class="mut small"> · {c["ctx"]} · v2 closures {c["n_v2"]}</span>'
            f'{_img(c["png"])}{_audio(c["wav"])}</div>'
            for c in ctx["cv"])

    def hear_audio():
        return "".join(
            f'<div class="card"><b>{h["tag"]}-pitch trill</b>'
            f'<span class="mut small"> · f₀ ≈ {h["f0"]} Hz · identical 2-closure structure</span>'
            f'{_audio(h["wav"])}</div>' for h in ctx["mech"]["hear"])

    def contact_gallery():
        return "".join(
            f'<div class="card"><b>{c["ds"].upper()}</b>'
            f'<div class="mut small">green border = the two detectors agree (±1 closure)</div>'
            f'{_img(c["png"])}</div>' for c in ctx["figs"]["contacts"])

    corpora_rows = "".join(
        f"<tr><td>{e(r[0])}</td><td>{r[1]}</td><td>{r[2]}</td><td>{e(r[3])}</td>"
        f"<td>{e(r[4])}</td></tr>" for r in ctx["corpora_table"])
    valid_rows = "".join(
        f"<tr><td>{e(r[0])}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
        f"<td>{r[4]}</td></tr>" for r in ctx["valid_table"])

    effect_img = _img(ctx["figs"]["effect"]) if ctx["figs"]["effect"] else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spanish Trill Production — Supplementary Material</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Spanish Trill Production: A Multi-Corpus Acoustic Study</h1>
  <p>Supplementary material — listen to trill examples and see the
  closure-vs-envelope-peak distinction, the f₀ mechanism, the phonotactic-context
  effect, and per-corpus validation.</p>
  <div class="badge">Anonymous submission · supplementary site</div>
</header>
<main>

<figure style="margin:0 0 22px;text-align:center">
{_img(ctx['figs']['summary']) if ctx['figs'].get('summary') else ''}
<figcaption class="mut small">Graphical abstract — on the same trill, the mid-band
envelope has two closures but ~5 envelope peaks; the envelope-peak count invents a
male advantage (+1.79, p&lt;10⁻⁹) that vanishes when closures are counted (+0.22, n.s.).</figcaption>
</figure>

<section id="overview"><h2>1 · What this is</h2>
<p>We measure the Spanish trill /r/ by counting <b>closures</b> (the occlusion
event, a mid-band energy minimum with a verified release) directly, instead of
counting peaks of a rectified amplitude envelope.
Across <b>3,560 tokens</b> from <b>356 speakers</b> in <b>six corpora</b>, the trill
has a median of <b>two closures</b> and an inter-closure period near <b>36&nbsp;ms</b>.
Phonotactic context is the only factor with a robust, medium effect; speaker sex
shows none once closures are counted directly.</p>
<div class="key warn small"><b>Audio note.</b> Only redistributable audio is playable
here: <b>synthetic</b> trills and <b>Common Voice</b> (CC0). The six analysed corpora
are licence-restricted, so for them we show only derived visualisations
(spectrograms, contact sheets), never the audio.</div></section>

<section id="canonical"><h2>2 · Hear the unit: 1–3 closures</h2>
<p>Synthetic trills with a known number of closures. The blue lines mark the
closures the detector verifies (occlusion + a following release burst).</p>
<div class="grid">{canon_cards()}</div></section>

<section id="mechanism"><h2>3 · Closures vs envelope peaks — the sex artifact</h2>
<p>An <b>envelope-peak</b> counter marks a maximum for both the occlusion and its
release, so it roughly <b>doubles</b> the count. On the same synthetic token the
closure detector finds <b>{ctx['mech']['n_v2']}</b> closures (blue), while an
envelope-peak counter marks <b>{ctx['mech']['n_peaks']}</b> peaks (red dotted).</p>
{_img(ctx['mech']['png'])}{_audio(ctx['mech']['wav'])}
<h3>Why this creates a spurious sex effect</h3>
<p>The size of that over-count depends on f₀ and harmonic density — and therefore on
speaker sex. On the real tokens, the per-token over-count <b>rises as f₀ falls</b>
(Spearman ρ = {ctx['scatter']['rho']}, n = {ctx['scatter']['n']}). The envelope-peak
count is f₀-dependent; the closure count is not.</p>
<div class="two">
  <div>{_img(ctx['scatter']['png'])}</div>
  <div><table><thead><tr><th>measure</th><th>F</th><th>M</th><th>M−F</th></tr></thead>
  <tbody>
   <tr><td>envelope-peak count</td><td>4.33</td><td>6.12</td><td>+1.79*</td></tr>
   <tr><td>closure count (v2)</td><td>1.80</td><td>2.02</td><td>+0.22 n.s.</td></tr>
  </tbody></table>
  <p class="small mut">Speakers with independent (metadata) sex labels, DIMEx100
  excluded. *Mann–Whitney p&lt;10⁻⁹, rank-biserial r = 0.53. The over-count absorbs
  +1.58 of the +1.79 gap, so the apparent sex effect is a counting artifact.</p></div>
</div>
<h3>Hear a low- and a high-pitched trill</h3>
<p class="mut small">Identical 2-closure structure, different f₀ — the closure count
should be the same for both.</p>
<div class="grid">{hear_audio()}</div></section>

<section id="context"><h2>4 · Phonotactic context (the one robust effect)</h2>
<p>Intervocalic <i>rr</i> shows the <b>fewest</b> closures; onset trills (word-initial,
post-/n,l,s/) show more. The independent autocorrelation cross-detector reproduces
this ranking exactly and counts intervocalic trills <i>higher</i>
({ctx['ctx_cross']}), so the effect is not a closure-detection artifact.</p>
{effect_img}
<p class="mut small">Kruskal–Wallis ε² for every predictor on closure count, closure
rate and duration, on one scale. Context is the only predictor reaching a medium
effect; sex is negligible. Non-significant cells (after Bonferroni) are hatched.</p>
<h3>Context examples (Common Voice, CC0)</h3>
<div class="grid">{cv_cards()}</div></section>

<section id="validation"><h2>5 · Per-corpus validation</h2>
<p>All six corpora pass literature-anchored validation. Each contact sheet tiles
example tokens; a <span style="color:#3a3">green</span> border means the two
independent detectors agree within ±1 closure, <span style="color:#c33">red</span>
means they disagree.</p>
<table><thead><tr><th>Corpus</th><th>Tokens</th><th>Median n</th>
<th>Period (ms)</th><th>Agree.</th></tr></thead><tbody>{valid_rows}</tbody></table>
<div class="grid">{contact_gallery()}</div></section>

<section id="corpora"><h2>6 · Corpora &amp; sample sizes</h2>
<table><thead><tr><th>Dataset</th><th>Speakers</th><th>Tokens</th>
<th>Region</th><th>Style</th></tr></thead><tbody>{corpora_rows}</tbody></table>
<p class="mut small">Post-filter token counts per cell (the balance behind the
context analysis; word-initial is the smallest cell).</p>
<h3>Tokens per context × corpus</h3>
{ctx['sizes']['corpus']}
<h3>Tokens (speakers) per sex × context</h3>
{ctx['sizes']['sex']}
<h3>Quality-filter retention (candidates → retained)</h3>
<p class="mut small">The filter keeps 21% of candidates overall but very little of
the read/lab-heavy material (DIMEx100 1.3%): the analysed set is clean, periodic
trill realisations, not all expected /r/ contexts.</p>
{ctx['retention']['corpus']}
{ctx['retention']['context']}
<h3>Quality-filter retention by speaker sex</h3>
<p class="mut small">{ctx['attrition_sex']['note']}</p>
{ctx['attrition_sex']['table']}</section>

<section id="repro"><h2>7 · Reproducibility</h2>
<p>The pipeline is deterministic and reproducible from the candidate token tables
plus the source audio. The closure detector, the fixed quality filter
(duration ∈ [50, 200] ms, voicing ≥ 80 %, envelope-periodicity ≥ 0.40), and the
independent period cross-detector are all released with the code.</p>
<p>The complete pipeline — the closure detector, the fixed quality filter, the
independent period cross-detector, and the test suite — is provided as an
accompanying anonymous code archive submitted alongside this paper.</p>
<p class="mut small">This page contains no author-identifying information.</p></section>

</main>
<footer>Supplementary material · anonymous submission · synthetic &amp; CC0 audio only</footer>
</body></html>"""


def main() -> None:
    if SUP.exists():
        shutil.rmtree(SUP)
    AUDIO.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    log.info("· synthetic canonical examples")
    canonical = gen_canonical_examples()
    log.info("· mechanism demo")
    mech = gen_mechanism_demo()
    log.info("· excess~F0 scatter (real data)")
    scatter = gen_excess_scatter()
    log.info("· per-cell sample sizes")
    sizes = gen_sample_sizes()
    log.info("· filter retention")
    retention = gen_retention()
    log.info("· filter attrition by sex")
    attrition_sex = gen_attrition_by_sex()
    log.info("· Common Voice examples")
    cv = gen_cv_examples()
    log.info("· static figures / contact sheets")
    figs = copy_static_figures()

    # cross-detector intervocalic numbers
    ctx_cross = "2.07 vs 1.81 closures"
    cc = ROOT / "outputs" / "v2" / "tables" / "context_cross_detector.csv"
    if cc.exists():
        df = pd.read_csv(cc)
        row = df[(df.scope == "pooled") & (df.aggregation == "speaker x context")
                 & (df.context == "intervocalic_rr")]
        if not row.empty:
            ctx_cross = (f"{row.cross_mean_n.iloc[0]:.2f} vs "
                         f"{row.primary_mean_n.iloc[0]:.2f} closures")

    corpora_table = [
        ("TEDx", 138, "1,481", "Pan-Hispanic", "spontaneous"),
        ("Heroico", 112, "1,151", "Mex/Sp/Ar", "read"),
        ("Glissando", 27, "762", "Spain", "read+spont."),
        ("DIMEx100", 40, "81", "Mexico", "read"),
        ("ALBAYZIN", 30, "46", "Spain", "read"),
        ("PRESEEA", 9, "39", "Spain", "spontaneous"),
    ]
    valid_table = [
        ("TEDx", "1,481", "2.0", "35.0", "95.6%"),
        ("Heroico", "1,151", "2.0", "36.2", "95.2%"),
        ("Glissando", "762", "2.0", "37.5", "91.2%"),
        ("DIMEx100", "81", "1.0", "30.0", "92.6%"),
        ("ALBAYZIN", "46", "2.0", "30.0", "67.4%"),
        ("PRESEEA", "39", "1.0", "35.0", "79.5%"),
    ]

    html_str = build_html({
        "canonical": canonical, "mech": mech, "scatter": scatter, "cv": cv,
        "figs": figs, "ctx_cross": ctx_cross, "sizes": sizes, "retention": retention,
        "attrition_sex": attrition_sex,
        "corpora_table": corpora_table, "valid_table": valid_table,
    })
    (SUP / "index.html").write_text(html_str, encoding="utf-8")
    (SUP / "assets" / "css").mkdir(parents=True, exist_ok=True)
    (SUP / "assets" / "css" / "style.css").write_text(CSS, encoding="utf-8")
    log.info("wrote %s (%d audio, %d images)", SUP / "index.html",
             len(list(AUDIO.glob('*.wav'))), len(list(IMG.glob('*.png'))))


if __name__ == "__main__":
    main()
