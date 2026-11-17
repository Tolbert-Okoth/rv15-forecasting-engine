"""Paired evaluation and aggregate reporting."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .frozen import API, require


def metrics(p):
    loss0,loss2 = p.loss_c0.mean(),p.loss_c2.mean()
    raw = p.raw.sum()
    return dict(n=len(p),qlike_c0=float(loss0),qlike_c2=float(loss2),
                paired_gain=float(loss0-loss2),relative_gain_pct=float(100*(loss0-loss2)/loss0) if loss0 else None,
                calibration_c0=float(p.c0.sum()/raw) if raw else None,
                calibration_c2=float(p.c2.sum()/raw) if raw else None,
                additive_c0=float((p.c0-p.raw).mean()),additive_c2=float((p.c2-p.raw).mean()))


def analysis(key,rows,counts,m):
    require(rows,"ledger has no mature paired outcomes")
    p = pd.DataFrame(rows)
    recalculated = API.score(p.assign(predictor_end=p.origin),["c0","c2"])
    np.testing.assert_allclose(p[["loss_c0","loss_c2"]],recalculated[["loss_c0","loss_c2"]],rtol=1e-12,atol=1e-12)
    local = API.timestamps(p.origin)
    london = local.tz_convert("Europe/London")
    p["year"] = local.year
    p["month"] = local.strftime("%Y-%m")
    p["ny_slot"] = local.hour*4+local.minute//15
    p["bucket"] = np.searchsorted(m["bucket_cuts"],p.c0,side="left")+1
    ldn = (london.hour*60+london.minute>=480)&(london.hour*60+london.minute<990)
    ny = (local.hour>=8)&(local.hour<17)
    p["market"] = np.select([ldn&ny,ldn,ny],["overlap","London only","New York only"],default="outside")
    p["tail"] = False
    for _,g in p.groupby(p.year.ge(2026)):
        n=math.ceil(.01*len(g))
        idx=g.raw.nlargest(n).index.union((g.c0-g.raw).abs().nlargest(n).index)
        p.loc[idx,"tail"]=True
    tail_policy = "CAL04 upper-1% union rule applied to this sample, split pre-2026 versus 2026+."
    p["gain"] = p.loss_c0-p.loss_c2
    p["n"] = 1
    d = API.session_sums(p,["gain","loss_c0","n"])
    intervals={}
    for size in (5,10,20):
        if len(d)<max(20,size*2):
            continue
        draws=API.bootstrap(d,size,2000,150100+size)
        intervals[str(size)] = dict(paired=API.interval(draws[:,0]/draws[:,2]),
                                    gain_pct=API.interval(100*API.divide(draws[:,0],draws[:,1])))
    return dict(identity=key,counts=counts,overall=metrics(p),intervals=intervals,tail_policy=tail_policy,
                groups={c:{str(k):metrics(g) for k,g in p.groupby(c,sort=True)}
                        for c in ("year","month","ny_slot","market","bucket","tail")})


def write_report(result, destination):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    key = result["identity"]
    lines = ["# RV15 paired forecast report","",f'**Mode** — {key["mode"]}',"",
             f'**C0** `{key["c0"]}`  ',f'**C2** `{key["c2"]}`  ',f'**Protocol** `{key["protocol"]}`',"",
             "Target is the sum of the next 15 squared completed-M1 log returns. Both models are scored on identical mature targets.",
             "",f'**Data provenance and quality** — `{json.dumps(key["provenance"],sort_keys=True)}`',
             "",f'**Ledger counts** — `{json.dumps(result["counts"],sort_keys=True)}`',"",
             result["tail_policy"],"",
             "Paired gain is mean C0 minus C2 QLIKE. Aggregate calibration is forecast sum divided by raw realised variance sum. Additive error is mean forecast minus raw variance.",""]
    def fmt(x,format_spec=".6f"):
        return "—" if x is None else format(x,format_spec)
    for label,rows in {"Overall":{"paired":result["overall"]},**result["groups"]}.items():
        lines.extend([f"## {label}","","| Group | N | C0 QLIKE | C2 QLIKE | Paired gain | Gain % | C0 calibration | C2 calibration | C0 additive | C2 additive |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for k,r in rows.items():
            values=[str(r["n"]),*[fmt(r[c]) for c in ("qlike_c0","qlike_c2","paired_gain","relative_gain_pct","calibration_c0","calibration_c2")],
                    fmt(r["additive_c0"],".3e"),fmt(r["additive_c2"],".3e")]
            lines.append("| "+" | ".join([k,*values])+" |")
        lines.append("")
    lines.extend(["## Uncertainty","",
                  "Paired circular session-block bootstrap. Sparse samples suppress intervals. These intervals describe the scored population and do not account for historical model selection.",""])
    for size,interval in result["intervals"].items():
        lo,hi=interval["gain_pct"]["ci"] or (None,None)
        lines.append(f"- {size}-session blocks, relative QLIKE gain 95% interval [{fmt(lo)}, {fmt(hi)}]%.")
    lines.extend(["","## Limitations","",
                  "Retrospective results do not establish prospective forecasting superiority. Synthetic examples are software checks. Variance prediction does not imply directional price prediction or trading profitability. Historical source-event timestamps do not establish live receipt or commitment timing."])
    (destination/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    chart(result,destination/"quality.png")


def chart(result,path):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str(path.parent/".plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = result["groups"]["month"]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(2,1,figsize=(9,5.7),sharex=True,layout="constrained")
    for model,color in (("c0","#315e83"),("c2","#c75b39")):
        axes[0].plot(x,[v["qlike_"+model] for v in groups.values()],marker=".",label=model.upper(),color=color)
        axes[1].plot(x,[v["calibration_"+model] for v in groups.values()],marker=".",color=color)
    axes[0].set_ylabel("Mean QLIKE")
    axes[1].set_ylabel("Forecast / realised")
    axes[1].axhline(1,color="#777777",linestyle=":")
    axes[0].legend(frameon=False)
    step=max(1,len(x)//10)
    axes[1].set_xticks(x[::step],list(groups)[::step],rotation=30,ha="right")
    fig.suptitle(result["identity"]["mode"].replace("_"," ").title())
    fig.savefig(path,dpi=160)
    plt.close(fig)



