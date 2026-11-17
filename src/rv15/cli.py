"""Offline RV15 demo, replay, validation and evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from .data import bars_from_csv, build_panel, fixture
from .evaluation import analysis, write_report
from .frozen import C0, C2, manifest, require, sha
from .ledger import MODES, append, config, reader, validate


def slots(m):
    return map(int,m["model"]["training_season_by_slot"])


def compatible(path,m):
    before=sha(path)
    panel=build_panel(bars_from_csv(path),slots(m))
    require(sha(path)==before,"CSV changed during load")
    panel.quality.update(dataset_sha256=before,adapter="opening-labelled offset-aware M1 CSV",
                         cutoff="archival completion only; actual receipt unavailable")
    return panel


def synthetic_panel(m):
    frame=fixture()
    panel=build_panel(frame,slots(m))
    panel.quality.update(adapter="seeded synthetic fixture v1",
                         dataset_sha256=hashlib.sha256(frame.index.asi8.tobytes()+frame.close.to_numpy().tobytes()).hexdigest())
    return panel


def evaluate(path,destination,m):
    with reader(path) as db:
        key,rows,counts=validate(db,m)
    result=analysis(key,rows,counts,m)
    write_report(result,destination)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo",help="run a synthetic software example")
    demo.add_argument("--out",type=Path,required=True)
    rep=sub.add_parser("replay",help="replay an authorized M1 CSV with external frozen parameters")
    rep.add_argument("--dataset",type=Path,required=True)
    rep.add_argument("--model",type=Path,required=True,help="authorized external frozen manifest")
    rep.add_argument("--ledger",type=Path,required=True)
    ev=sub.add_parser("evaluate",help="validate and score a ledger read-only")
    ev.add_argument("--ledger",type=Path,required=True)
    ev.add_argument("--out",type=Path,required=True)
    ev.add_argument("--model",type=Path,help="required for ledgers using external frozen parameters")
    val=sub.add_parser("validate",help="validate identities and optionally an input or ledger")
    val.add_argument("--ledger",type=Path)
    val.add_argument("--dataset",type=Path)
    val.add_argument("--model",type=Path,help="authorized external frozen manifest")
    args=parser.parse_args()
    try:
        m=manifest(getattr(args,"model",None))
        if args.command=="replay":
            panel=compatible(args.dataset,m)
            key=config(MODES[0],m["protocol"],panel.quality,m)
            path=append(args.ledger,key,panel,m)
            result=dict(ledger=str(path),quality=panel.quality)
        elif args.command=="evaluate":
            result=evaluate(args.ledger,args.out,m)["overall"]
        elif args.command=="validate":
            result=dict(active_model_kind=m["kind"],active_c0=m["c0"],active_c2=m["c2"],
                        original_c0=C0,original_c2=C2)
            if args.dataset:
                result["dataset"]=compatible(args.dataset,m).quality
            if args.ledger:
                with reader(args.ledger) as db:
                    _,_,result["ledger"]=validate(db,m)
        else:
            panel=synthetic_panel(m)
            key=config(MODES[1],m["protocol"],panel.quality,m)
            path=append(args.out/"demo.sqlite",key,panel,m)
            result=evaluate(path,args.out,m)["overall"]
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,KeyError,OSError,AssertionError,sqlite3.Error) as error:
        parser.exit(1,f"RV15 error: {error}\n")


if __name__=="__main__":
    main()
