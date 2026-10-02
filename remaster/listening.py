"""Blind listening test: build trials from system outputs, serve the test page, analyse the ratings.

  python -m remaster.listening build   --src <dir> --dir listening        # <src>/<trial>/{reference,<system>...}.wav
  python -m remaster.listening serve   --dir listening --port 7871
  python -m remaster.listening analyze --dir listening --target lacquer

The design follows ITU-R BS.1534 (MUSHRA) where a reference exists: the listener hears the labelled reference
and rates every condition, including a hidden copy of the reference, from 0 to 100. Trials without a reference
(mastering) are rated the same way as a plain preference test. Conditions are loudness-matched at build time
and shuffled per listener. Listeners who rate the hidden reference under 90 in more than 15% of trials are
excluded, as the recommendation prescribes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import uuid

import numpy as np
import soundfile as sf

HERE = os.path.dirname(__file__)


def build(src, out, lufs=-18.0, seconds=12.0):
    from .analysis import loudness_stats
    trials = []
    for d in sorted(glob.glob(os.path.join(src, "*", ""))):
        name = os.path.basename(d.rstrip("/"))
        files = sorted(glob.glob(os.path.join(d, "*.wav")))
        if len(files) < 2:
            continue
        os.makedirs(os.path.join(out, "stimuli", name), exist_ok=True)
        conds = []
        for f in files:
            x, sr = sf.read(f, dtype="float32", always_2d=True)
            x = x[: int(seconds * sr)]
            l = loudness_stats(x.T, sr)["lufs"]
            x = x * 10 ** ((lufs - l) / 20) if np.isfinite(l) else x
            peak = np.abs(x).max()
            x = x / peak * 0.98 if peak > 0.98 else x
            fade = int(0.02 * sr)
            x[:fade] *= np.linspace(0, 1, fade)[:, None]; x[-fade:] *= np.linspace(1, 0, fade)[:, None]
            c = os.path.basename(f)[:-4]
            sf.write(os.path.join(out, "stimuli", name, c + ".wav"), x, sr, subtype="PCM_24")
            conds.append(c)
        has_ref = "reference" in conds
        trials.append(dict(name=name, reference=has_ref, conditions=[c for c in conds if c != "reference"] + (["hidden_reference"] if has_ref else [])))
    json.dump(dict(trials=trials, lufs=lufs), open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(len(trials), "trials,", sum(len(t["conditions"]) for t in trials), "ratings per listener")


def serve(root, port):
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    app = FastAPI()
    os.makedirs(os.path.join(root, "results"), exist_ok=True)

    @app.get("/manifest")
    def manifest():
        return JSONResponse(json.load(open(os.path.join(root, "manifest.json"))))

    @app.get("/audio/{trial}/{cond}")
    def audio(trial: str, cond: str):
        cond = "reference" if cond == "hidden_reference" else cond
        path = os.path.realpath(os.path.join(root, "stimuli", trial, cond + ".wav"))
        if not path.startswith(os.path.realpath(os.path.join(root, "stimuli"))) or not os.path.exists(path):
            return JSONResponse(dict(error="not found"), status_code=404)
        return FileResponse(path)

    @app.post("/submit")
    async def submit(req: Request):
        body = await req.json()
        body["received"] = time.time()
        json.dump(body, open(os.path.join(root, "results", uuid.uuid4().hex + ".json"), "w"), indent=1)
        return dict(ok=True)

    app.mount("/", StaticFiles(directory=os.path.join(HERE, "web_listening"), html=True), name="web")
    uvicorn.run(app, host="0.0.0.0", port=port)


def analyze(root, target="lacquer"):
    from scipy import stats
    res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, "results", "*.json")))]
    kept, dropped = [], 0
    for r in res:
        hid = [t["ratings"]["hidden_reference"] for t in r["trials"] if "hidden_reference" in t["ratings"]]
        if hid and np.mean(np.array(hid) < 90) > 0.15:
            dropped += 1
            continue
        kept.append(r)
    print(f"{len(res)} listeners, {dropped} excluded by the hidden-reference rule, {len(kept)} kept")
    groups = {}
    for li, r in enumerate(kept):
        for t in r["trials"]:
            g = t["trial"].split("_")[0]
            for c, v in t["ratings"].items():
                groups.setdefault(g, {}).setdefault(c, {})[(li, t["trial"])] = v
    out = {}
    for g, conds in groups.items():
        out[g] = {}
        print(f"\n{g}")
        others = [c for c in conds if c not in (target, "hidden_reference")]
        pvals = {}
        for c, d in conds.items():
            v = np.array(list(d.values()), dtype=np.float64)
            ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v)) if len(v) > 1 else (v.mean(), v.mean())
            out[g][c] = dict(n=int(len(v)), mean=float(v.mean()), ci95=[float(ci[0]), float(ci[1])])
        if target in conds:
            for c in others:
                keys = sorted(set(conds[target]) & set(conds[c]))
                a, b = np.array([conds[target][k] for k in keys]), np.array([conds[c][k] for k in keys])
                pvals[c] = float(stats.wilcoxon(a, b).pvalue) if len(keys) > 5 and np.any(a != b) else float("nan")
            # Holm correction over the comparisons against the target
            order = sorted([c for c in pvals if np.isfinite(pvals[c])], key=lambda c: pvals[c])
            running = 0.0
            for i, c in enumerate(order):
                running = max(running, min(1.0, pvals[c] * (len(order) - i)))
                out[g][c]["p_vs_target_holm"] = running
        for c, s in sorted(out[g].items(), key=lambda kv: -kv[1]["mean"]):
            print(f"  {c:20s} {s['mean']:5.1f}  [{s['ci95'][0]:5.1f}, {s['ci95'][1]:5.1f}]  n={s['n']}" + (f"  p={s['p_vs_target_holm']:.3g} against {target}" if "p_vs_target_holm" in s else ""))
    json.dump(out, open(os.path.join(root, "analysis.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "serve", "analyze"])
    p.add_argument("--src", default=None)
    p.add_argument("--dir", required=True)
    p.add_argument("--port", type=int, default=7871)
    p.add_argument("--target", default="lacquer")
    a = p.parse_args()
    if a.cmd == "build":
        build(a.src, a.dir)
    elif a.cmd == "serve":
        serve(a.dir, a.port)
    else:
        analyze(a.dir, a.target)


if __name__ == "__main__":
    main()
