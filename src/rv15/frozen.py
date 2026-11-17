"""Frozen identities, software fixture and causal calibration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import primitives as FWD
from . import scoring as API
from .primitives import digest, stamp, utc

MINUTE = pd.Timedelta(minutes=1)
NY = "America/New_York"
C0 = "efdf79e7d406521770b67c1de744fb3b2af2a082b5c9de0c1b9ebe77369afd48"
C2 = "7426520c260a2de0e35749b8531c983aa2e7cbe42290d28cd1fe9771508fe5f1"
CUTS_HASH = "c6076860f6dddd9f0bdb9e2244f357ec7840c9b2e1f75f21e5d6bfbf57f42762"
ARCHIVE_PROTOCOL = digest("RV15-PUBLISH01/ARCHIVAL-COMPLETION/1")
DEMO_PROTOCOL = digest("RV15-PUBLISH01/SOFTWARE-FIXTURE/1")


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest(path=None):
    if path is None:
        # Arbitrary, unfitted constants for exercising the same software.
        spec = dict(frozen=dict(beta=[0., .4, .3, .2, .1],smear=1.),
                    training_season_by_slot={str(slot):1e-6 for slot in range(96)})
        c0 = digest([spec["frozen"],spec["training_season_by_slot"]])
        c2_spec = dict(parent=c0, fixture="RV15 software demonstration v1",half_life_sessions=10,
                       minimum_sessions=20,init=1,clip=[.5,2],post_target_minutes=1)
        return dict(model=spec,c0=c0,c2=digest(c2_spec),c2_spec=c2_spec,
                    bucket_cuts=[1e-7,3e-7,1e-6,3e-6],protocol=DEMO_PROTOCOL,kind="SYNTHETIC_FIXTURE")
    with Path(path).open(encoding="utf-8-sig") as file:
        raw = json.load(file)
    spec, c2_spec, cuts = raw["model"], raw["c2_spec"], raw["bucket_cuts"]
    require(digest([spec["frozen"],spec["training_season_by_slot"]])==raw["c0"]==C0,"C0 identity mismatch")
    require(digest(c2_spec)==raw["c2"]==C2,"C2 identity mismatch")
    require(digest(cuts)==CUTS_HASH,"frozen bucket metadata mismatch")
    model = dict(frozen=spec["frozen"],training_season_by_slot=spec["training_season_by_slot"])
    return dict(model=model,c0=C0,c2=C2,c2_spec=c2_spec,bucket_cuts=cuts,
                protocol=ARCHIVE_PROTOCOL,kind="AUTHORIZED_EXTERNAL_FROZEN")


def forecast(features,q,slot,model,factor):
    require(np.isfinite(factor) and .5<=factor<=2., "invalid C2 factor")
    h = FWD.frozen_from_floor(features,q,slot,model)
    result = h,max(q,h*factor)
    require(np.isfinite(result).all(), "nonfinite forecast")
    return result


def calibration(prior, day, origin):
    """Fold only complete, mature prior sessions. Keep state in levels."""
    decay = 2**(-1/10)
    state, count = 1., 0
    for row in prior:
        if row["session"] < day and row["available"] <= origin and row["ratio"] is not None:
            state = decay*state+(1-decay)*row["ratio"]
            count += 1
    return (float(np.clip(state,.5,2.)) if count>=20 else 1.), count
