"""Timestamp, forecast and loss primitives from RV15-FWD01."""
import hashlib
import json
import math

import numpy as np
import pandas as pd

NY = "America/New_York"
MIN = pd.Timedelta(minutes=1)

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",",":"),allow_nan=False)

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def utc(x):
    t=pd.Timestamp(x)
    if t.tzinfo is None or pd.isna(t):
        raise ValueError("explicit timestamp offset required")
    return t.tz_convert("UTC")

def stamp(x):
    return utc(x).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def origin_parts(value):
    t=utc(value)
    wall=t.tz_convert(NY).tz_localize(None)
    if wall!=wall.floor("15min"):
        raise ValueError("off-grid origin")
    return t,str((wall+pd.Timedelta(hours=6)).date()),wall.hour*4+wall.minute//15

def session_end(day):
    return (pd.Timestamp(day)+pd.Timedelta(hours=18)).tz_localize(NY).tz_convert("UTC")

def frozen_from_floor(features,q,slot,model):
    values=np.array([*features,model["training_season_by_slot"][str(slot)]],dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or not math.isfinite(q) or q<=0:
        raise ValueError("invalid variance or floor")
    x=np.r_[1.,np.log(np.maximum(values,q))]
    return max(q,float(np.exp(x@model["frozen"]["beta"])*model["frozen"]["smear"]))

def qlike(y,h):
    x=y/h
    return x-math.log(x)-1

