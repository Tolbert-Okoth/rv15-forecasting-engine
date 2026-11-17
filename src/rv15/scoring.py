"""Matched loss and session-block utilities from RV15-RELEASE01."""
import math

import numpy as np
import pandas as pd

NY = "America/New_York"
MINUTE = pd.Timedelta(minutes=1)

def timestamps(values):
    """Require either explicit offsets on all rows or unambiguous New York wall times."""
    strings = pd.Series(values, dtype="string")
    if strings.isna().any():
        raise ValueError("missing timestamp")
    aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$")
    if aware.any() and not aware.all():
        raise ValueError("mixed naive and offset timestamps")
    if aware.all():
        return pd.DatetimeIndex(pd.to_datetime(strings, format="mixed", utc=True)).tz_convert(NY)
    return pd.DatetimeIndex(pd.to_datetime(strings, format="mixed")).tz_localize(NY, ambiguous="raise", nonexistent="raise")

def origin_sessions(values):
    local = timestamps(values)
    wall = local.tz_localize(None)
    if not wall.equals(wall.floor("15min")):
        raise ValueError("off-grid forecast origin")
    if len(local) > 1 and (np.diff(local.asi8) < pd.Timedelta(minutes=15).value).any():
        raise ValueError("unsorted, duplicate or overlapping targets")
    return local, (wall + pd.Timedelta(hours=6)).floor("D")

def score(frame, forecasts):
    if frame.empty or not forecasts or len(set(forecasts)) != len(forecasts):
        raise ValueError("need rows and unique forecast columns")
    local, sessions = origin_sessions(frame.origin)
    if "session" in frame:
        supplied = pd.DatetimeIndex(pd.to_datetime(frame.session))
        if not supplied.equals(sessions):
            raise ValueError("session does not match New York date rollover")
    if ("predictor_end" in frame) != ("target_end" in frame):
        raise ValueError("supply both predictor_end and target_end, or neither")
    if "predictor_end" in frame:
        if (timestamps(frame.predictor_end) > local).any() or not timestamps(frame.target_end).equals(local+15*MINUTE):
            raise ValueError("predictor look-ahead or wrong target endpoint")
    raw = frame.raw.to_numpy(dtype=float)
    floor = frame.q.to_numpy(dtype=float)
    f = frame[forecasts].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([raw, floor, f])).all() or (raw<0).any() or (floor<=0).any() or (f<=0).any():
        raise ValueError("nonfinite value, negative target, or nonpositive floor/forecast")
    h = np.maximum(f, floor[:,None])
    x = np.maximum(raw, floor)[:,None]/h
    loss = x-np.log(x)-1
    if not np.isfinite(loss).all():
        raise ValueError("nonfinite normalized QLIKE")
    return pd.DataFrame({"session":sessions, "raw":raw, "floor":floor,
                         **{m:h[:,j] for j,m in enumerate(forecasts)},
                         **{"loss_"+m:loss[:,j] for j,m in enumerate(forecasts)}})

def session_sums(frame, columns, dates=None):
    daily = frame.groupby("session", sort=True)[columns].sum()
    if dates is not None:
        daily = daily.reindex(dates, fill_value=0)
    return daily.to_numpy()

def bootstrap(daily, block, count, seed):
    """Return observed sums followed by circular session-block draws."""
    n = len(daily)
    if not 1 <= block <= n or count < 1:
        raise ValueError("block must fit the session count and draws must be positive")
    starts = np.random.default_rng(seed).integers(n, size=(count, math.ceil(n/block), 1))
    index = ((starts+np.arange(block))%n).reshape(count,-1)[:,:n]
    multiplicity = np.array([np.bincount(row,minlength=n) for row in index])
    return np.vstack([daily.sum(axis=0),multiplicity@daily])

def interval(samples, alpha=.05):
    finite = samples[1:][np.isfinite(samples[1:])]
    return {"estimate": float(samples[0]) if np.isfinite(samples[0]) else None,
            "ci": np.quantile(finite,[alpha/2,1-alpha/2]).tolist() if len(finite) else None,
            "valid_draws":len(finite)}

def divide(num, den):
    return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den!=0)

