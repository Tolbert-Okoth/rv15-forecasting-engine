"""Opening-labelled CSV adapter and separated predictor/target construction."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .frozen import MINUTE, NY, require

FEATURES = ["rv15","rv60","day"]


@dataclass
class Panel:
    inputs: pd.DataFrame
    targets: pd.Series
    quality: dict


def bars_from_csv(path):
    frame = pd.read_csv(path, dtype={"open_time":"string", "source":"string"})
    require({"open_time","close"}.issubset(frame), "CSV requires open_time and close")
    raw = frame.open_time
    require(raw.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$",na=False).all(), "open_time requires explicit offset")
    frame.index = pd.DatetimeIndex(pd.to_datetime(raw,format="ISO8601",utc=True))
    return frame


def build_panel(frame, supported_slots):
    t = frame.index
    require(len(t)>0 and isinstance(t,pd.DatetimeIndex) and t.tz is not None,
            "require nonempty offset-aware M1 opening times")
    t = t.tz_convert("UTC").as_unit("ns")
    require(not t.hasnans and t.is_monotonic_increasing and t.is_unique and t.equals(t.floor("min")),
            "minute openings must be aligned, unique and increasing")
    prices = pd.to_numeric(frame.close,errors="raise").to_numpy(float)
    require((np.isfinite(prices)&(prices>0)).all(), "close must be finite and positive")
    labels = frame.get("source",pd.Series("single_source",index=frame.index))
    require(labels.notna().all() and labels.astype(str).str.strip().ne("").all(), "missing source")
    labels = labels.to_numpy()
    source_pair = np.r_[False,labels[1:]==labels[:-1]]
    minute_pair = np.r_[False,np.diff(t.asi8)==MINUTE.value]
    consecutive = source_pair & minute_pair
    square = np.r_[0.,np.diff(np.log(prices))**2]
    square[~consecutive] = 0
    def rolling(n):
        values = pd.Series(square).rolling(n).sum()
        return values.where(pd.Series(consecutive).rolling(n).sum().eq(n)).to_numpy()
    wall = t.tz_convert(NY).tz_localize(None)
    days = (wall+6*pd.Timedelta(hours=1)).floor("D")
    within = consecutive & np.r_[False,days[1:]==days[:-1]]
    daily = pd.DataFrame(dict(day=days,value=square*within,count=within,source=labels)).groupby("day").agg(
        value=("value","sum"), count=("count","sum"), n_sources=("source","nunique"), source=("source","first"))
    origins = t+MINUTE
    local = origins.tz_convert(NY).tz_localize(None)
    sessions = (local+pd.Timedelta(hours=6)).floor("D")
    slots = local.hour*4+local.minute//15
    grid = (local.minute%15==0)&(sessions.dayofweek<5)&np.isin(slots,list(supported_slots))
    past = daily.reindex(sessions-pd.offsets.BDay())
    prior_ok = (past["count"].ge(1200)&past.n_sources.eq(1)&past.source.eq(labels)).to_numpy()
    rv15,rv60 = rolling(15),rolling(60)
    eligible = grid & prior_ok & np.isfinite(rv60)
    rows = pd.DataFrame(dict(origin=origins[eligible],session=sessions[eligible],slot=slots[eligible],
                             rv15=rv15[eligible],rv60=rv60[eligible],day=past.value.to_numpy()[eligible],
                             q=np.log1p(.001/prices[eligible])**2))
    targets = pd.Series(rolling(15)).shift(-15).to_numpy()[eligible]
    missing = np.diff(t.asi8)//MINUTE.value-1
    return Panel(rows,pd.Series(targets,index=rows.index),
                 dict(bars=len(t), missing_minutes=int(missing.sum()), gaps=int((missing>0).sum()),
                      source_changes=int((~source_pair[1:]).sum()), candidates=int(grid.sum()),
                      rejected_inputs=int((grid&~eligible).sum()), missing_targets=int(np.isnan(targets).sum()),
                      start=origins[0].isoformat(),end=origins[-1].isoformat()))


def fixture():
    """Seeded 27 weekday sessions with gaps between sessions and extreme variance."""
    rng = np.random.default_rng(19081)
    blocks = [pd.date_range(d+pd.Timedelta(hours=19),periods=1320,freq="min",tz=NY).tz_convert("UTC")
              for d in pd.bdate_range("2026-01-05",periods=38) if d.dayofweek!=4]
    time = blocks[0].append(blocks[1:])
    sigma = np.where(np.arange(len(time))//1320%7==0,.0018,.00010)
    return pd.DataFrame({"close":2000*np.exp(np.cumsum(rng.normal(size=len(time))*sigma))},index=time)

