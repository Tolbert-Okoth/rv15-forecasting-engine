"""Append-only SQLite replay ledger, using FWD01 forecast and timing primitives."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .data import FEATURES
from .frozen import API, FWD, MINUTE, calibration, digest, forecast, require, stamp, utc

canonical = FWD.canonical
TABLES = ("identity","sessions","opportunities","forecasts","outcomes")
MODES = ("RETROSPECTIVE_BAR_REPLAY","SYNTHETIC_DEMONSTRATION")
SCHEMA = """
CREATE TABLE identity (key INTEGER PRIMARY KEY CHECK(key=1), value TEXT NOT NULL);
CREATE TABLE sessions (session TEXT PRIMARY KEY, signature TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE opportunities (origin TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(session), status TEXT NOT NULL);
CREATE TABLE forecasts (id TEXT PRIMARY KEY, origin TEXT UNIQUE NOT NULL REFERENCES opportunities(origin), payload TEXT NOT NULL);
CREATE TABLE outcomes (id TEXT PRIMARY KEY REFERENCES forecasts(id), payload TEXT NOT NULL);
"""


@contextmanager
def reader(path):
    path = Path(path).resolve()
    with contextmanager_close(sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        yield db


@contextmanager
def contextmanager_close(db):
    try:
        yield db
    finally:
        db.close()


def config(mode,protocol,provenance,m):
    expected = MODES[1] if m["kind"]=="SYNTHETIC_FIXTURE" else MODES[0]
    require(mode==expected and protocol==m["protocol"],"replay mode or protocol mismatch")
    return dict(schema="RV15-PUBLISH01/1",mode=mode,protocol=protocol,c0=m["c0"],c2=m["c2"],provenance=provenance)


def identity(db):
    row = db.execute("SELECT value FROM identity WHERE key=1").fetchone()
    require(row is not None, "missing ledger identity")
    return json.loads(row[0])


def configure(db,key):
    db.execute("BEGIN IMMEDIATE")
    existing = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    if not existing:
        for statement in SCHEMA.strip().split(";"):
            if statement.strip():
                db.execute(statement)
        for table in TABLES:
            for operation in ("UPDATE","DELETE"):
                db.execute(f"""CREATE TRIGGER immutable_{table}_{operation} BEFORE {operation} ON {table}
                            BEGIN SELECT RAISE(ABORT,'append only'); END""")
        db.execute("INSERT INTO identity VALUES(1,?)",(canonical(key),))
    require(identity(db)==key,"ledger identity mismatch")
    db.commit()


def time_fields(origin):
    return dict(source_event=None,hypothetical_commit=None,model_asof=stamp(origin),
                actual_receipt=None,actual_commit=None)


def output(raw,q,c0,c2,origin):
    available = stamp(origin+16*MINUTE)
    payload = dict(target_end=stamp(origin+15*MINUTE), available_at=available,
                   actual_receipt=None,status="MISSING_TARGET" if raw is None else "SCORED")
    if raw is not None:
        require(math.isfinite(raw) and raw>=0,"invalid realised variance")
        y=max(raw,q)
        payload.update(raw=raw,loss_c0=FWD.qlike(y,c0),loss_c2=FWD.qlike(y,c2))
    return payload


def append(path,key,panel,m):
    p = panel.inputs
    require(not p.empty,"no eligible predictor origins")
    local, sessions = API.origin_sessions(p.origin)
    require(sessions.equals(pd.DatetimeIndex(p.session)),"invalid predictor sessions")
    require(np.array_equal(p.slot,local.hour*4+local.minute//15),"invalid predictor slots")
    require(panel.targets.index.equals(p.index),"target alignment mismatch")
    require(key==config(key["mode"],key["protocol"],key["provenance"],m),"ledger identity mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with contextmanager_close(sqlite3.connect(path)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        with db:
            configure(db,key)
        prior = [json.loads(x[0]) for x in db.execute("SELECT payload FROM sessions ORDER BY session")]
        completed = {d:s for d,s in db.execute("SELECT session,signature FROM sessions")}
        for session,g in p.groupby("session",sort=False):
            day = str(session.date())
            data = []
            for row in g.itertuples():
                o = stamp(row.origin)
                target = panel.targets.loc[row.Index]
                raw = None if pd.isna(target) else float(target)
                data.append([o,int(row.slot),[float(getattr(row,k)) for k in FEATURES],float(row.q),raw])
            signature = digest(data)
            if day in completed:
                require(completed[day]==signature,"conflicting immutable session")
                continue
            with db:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute("SELECT session,signature FROM sessions ORDER BY session DESC LIMIT 1").fetchone()
                if previous and previous[0]==day and previous[1]==signature:
                    prior = [json.loads(x[0]) for x in db.execute("SELECT payload FROM sessions ORDER BY session")]
                    continue
                require(not previous or previous[0]<day,"out-of-order or conflicting session")
                # Refresh if another process appended before the writer lock.
                if previous and (not prior or prior[-1]["session"]!=previous[0]):
                    prior = [json.loads(x[0]) for x in db.execute("SELECT payload FROM sessions ORDER BY session")]
                origin0 = data[0][0]
                factor,n = calibration(prior,day,origin0)
                ratios, rows = [], []
                for o,slot,x,q,raw in data:
                    origin = utc(o)
                    c0,c2 = forecast(x,q,slot,m["model"],factor)
                    if raw is not None:
                        require(math.isfinite(raw) and raw>=0,"invalid calibration observation")
                        ratios.append(max(raw,q)/c0)
                    fields = time_fields(origin)
                    fid = digest([key,o])
                    payload = dict(id=fid,origin=o,session=day,slot=slot,features=x,q=q,c0=c0,c2=c2,
                                   factor=factor,history=n,mode=key["mode"],protocol=key["protocol"],
                                   c0_hash=key["c0"],c2_hash=key["c2"],**fields)
                    rows.append((o,fid,payload,output(raw,q,c0,c2,origin)))
                available = max(FWD.session_end(day),utc(data[-1][0])+16*MINUTE)
                sealed = dict(session=day,first=origin0,last=data[-1][0],available=stamp(available),
                              ratio=math.fsum(ratios)/len(ratios) if ratios else None,
                              n=len(ratios),recorded_at=stamp(pd.Timestamp.now(tz="UTC")))
                db.execute("INSERT INTO sessions VALUES(?,?,?)",(day,signature,canonical(sealed)))
                db.executemany("INSERT INTO opportunities VALUES(?,?,?)",
                               ((o,day,"ISSUED") for o,fid,_,_ in rows))
                db.executemany("INSERT INTO forecasts VALUES(?,?,?)",
                               ((fid,o,canonical(f)) for o,fid,f,_ in rows if fid))
                db.executemany("INSERT INTO outcomes VALUES(?,?)",
                               ((fid,canonical(y)) for _,fid,_,y in rows if fid))
                prior.append(sealed)
    return path


def records(db):
    for fid,o,day,status,f,y in db.execute(
            """SELECT f.id,p.origin,p.session,p.status,f.payload,y.payload
               FROM opportunities p LEFT JOIN forecasts f ON p.origin=f.origin
               LEFT JOIN outcomes y ON f.id=y.id ORDER BY p.origin"""):
        yield fid,o,day,status,json.loads(f) if f else None,json.loads(y) if y else None


def validate(db,m):
    require(db.execute("PRAGMA integrity_check").fetchone()==("ok",),"SQLite integrity failure")
    require(not db.execute("PRAGMA foreign_key_check").fetchall(),"ledger foreign key failure")
    key = identity(db)
    require(key.get("schema")=="RV15-PUBLISH01/1" and key["mode"] in MODES
            and key["c0"]==m["c0"] and key["c2"]==m["c2"],"ledger schema or model mismatch")
    config(key["mode"],key["protocol"],key["provenance"],m)
    trigger = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    require(all(f"immutable_{t}_{op}" in trigger for t in TABLES for op in ("UPDATE","DELETE")),
            "missing append-only protection")
    sessions = [json.loads(x[0]) for x in db.execute("SELECT payload FROM sessions ORDER BY session")]
    by_day = {s["session"]:s for s in sessions}
    scales = {s["session"]:calibration(sessions[:i],s["session"],s["first"]) for i,s in enumerate(sessions)}
    sizes = {s["session"]:[] for s in sessions}
    ratios = {s["session"]:[] for s in sessions}
    paired = []
    missing = 0
    for fid,o,day,status,f,y in records(db):
        origin,expected_day,slot = FWD.origin_parts(o)
        require(day==expected_day and day in by_day,"invalid opportunity session")
        sizes[day].append(o)
        require(f is not None and status=="ISSUED" and fid==f["id"]==digest([key,o]) and f["origin"]==o
                and f["session"]==day and f["slot"]==slot and y is not None,"invalid forecast identity or outcome")
        for field,want in dict(mode=key["mode"],protocol=key["protocol"],c0_hash=key["c0"],c2_hash=key["c2"]).items():
            require(f[field]==want,"forecast provenance mismatch")
        require(all(f[k]==v for k,v in time_fields(origin).items()),"issuance chronology mismatch")
        factor,n = scales[day]
        c0,c2 = forecast(f["features"],f["q"],slot,m["model"],factor)
        require(f["history"]==n and np.allclose([f["factor"],f["c0"],f["c2"]],[factor,c0,c2],rtol=1e-12,atol=0),
                "forecast numerical or calibration mismatch")
        require(y==output(y.get("raw"),f["q"],f["c0"],f["c2"],origin),"outcome loss or maturity mismatch")
        if y["status"]=="SCORED":
            ratios[day].append(max(y["raw"],f["q"])/f["c0"])
            paired.append(dict(origin=o,session=day,q=f["q"],c0=f["c0"],c2=f["c2"],**y))
        else:
            missing += 1
    for s in sessions:
        day=s["session"]
        values=sizes[day]
        require(values and values[0]==s["first"] and values[-1]==s["last"] and
                s["available"]==stamp(max(FWD.session_end(day),utc(values[-1])+16*MINUTE)),
                "session chronology mismatch")
        r=ratios[day]
        require(s["n"]==len(r) and s["ratio"]==(math.fsum(r)/len(r) if r else None),
                    "calibration session aggregate mismatch")
    return key,paired,dict(sessions=len(sessions),missing_target=missing,
                           paired=len(paired),opportunities=sum(map(len,sizes.values())))



