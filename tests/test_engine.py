from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rv15.cli import compatible, evaluate, synthetic_panel
from rv15.data import FEATURES, build_panel, fixture
from rv15.frozen import C0,C2,FWD,calibration,forecast,manifest,stamp
from rv15.ledger import MODES,append,config,output,reader,time_fields,validate


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=manifest()
        cls.frame=fixture()
        cls.panel=synthetic_panel(cls.m)
        cls.slots=list(map(int,cls.m["model"]["training_season_by_slot"]))

    def key(self,mode=MODES[1]):
        return config(mode,self.m["protocol"],self.panel.quality,self.m)

    def test_frozen_identities_and_deterministic_extremes(self):
        self.assertEqual(self.m["kind"],"SYNTHETIC_FIXTURE")
        self.assertNotEqual(self.m["c0"],C0)
        self.assertNotEqual(self.m["c2"],C2)
        self.assertEqual(self.m,manifest())
        slot=self.slots[0]
        for scale in (0.,1e-6,.1,40.):
            values=(scale,2*scale,10*scale)
            a=forecast(values,1e-13,slot,self.m["model"],2.)
            self.assertEqual(a,forecast(values,1e-13,slot,self.m["model"],2.))
            self.assertGreaterEqual(a[0],1e-13)
            self.assertAlmostEqual(a[1],a[0]*2.)

    def test_causal_features_and_alignment(self):
        row=self.panel.inputs.iloc[300]
        t=row.origin
        actual=self.frame.loc[t-pd.Timedelta(minutes=61):t-pd.Timedelta(minutes=1),"close"]
        target=self.frame.loc[t-pd.Timedelta(minutes=1):t+pd.Timedelta(minutes=14),"close"]
        v=np.diff(np.log(actual))**2
        np.testing.assert_allclose([row.rv15,row.rv60,self.panel.targets.loc[row.name]],
                                   [v[-15:].sum(),v.sum(),(np.diff(np.log(target))**2).sum()],rtol=1e-12)
        changed=self.frame.copy()
        changed.loc[changed.index>=t,"close"] *= 3
        modified=build_panel(changed,self.slots)
        keep=self.panel.inputs.origin.le(t)
        pd.testing.assert_frame_equal(self.panel.inputs[keep],modified.inputs[modified.inputs.origin.le(t)])
        self.assertNotEqual(self.panel.targets.loc[row.name],modified.targets.loc[row.name])

    def test_gap_and_source_splice(self):
        row=self.panel.inputs.iloc[300]
        t=row.origin
        without=self.frame.drop(t-pd.Timedelta(minutes=10))
        self.assertFalse(build_panel(without,self.slots).inputs.origin.eq(t).any())
        without=self.frame.drop(t+pd.Timedelta(minutes=7))
        g=build_panel(without,self.slots)
        pos=g.inputs.index[g.inputs.origin.eq(t)][0]
        self.assertTrue(pd.isna(g.targets.loc[pos]))
        split=self.frame.assign(source=np.where(self.frame.index<t-pd.Timedelta(minutes=10),"A","B"))
        self.assertFalse(build_panel(split,self.slots).inputs.origin.eq(t).any())

    def test_csv_timezone_and_input_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"bars.csv"
            self.frame.rename_axis("open_time").to_csv(f)
            p=compatible(f,self.m)
            pd.testing.assert_frame_equal(p.inputs,self.panel.inputs,check_exact=False)
            f.write_text("open_time,close\n2026-01-01T00:00:00,10\n")
            with self.assertRaisesRegex(ValueError,"explicit offset"):
                compatible(f,self.m)
        with self.assertRaisesRegex(ValueError,"unique"):
            build_panel(pd.concat([self.frame.iloc[:1],self.frame]),self.slots)

    def test_c2_prior_sessions_and_maturity(self):
        history=[dict(session=f"2026-01-{d:02}",available="2026-01-23T00:00:00Z",ratio=2.)
                 for d in range(1,22)]
        self.assertEqual(calibration(history[:19],"2026-01-22","2026-01-24T00:00:00Z"),(1.,19))
        self.assertEqual(calibration(history,"2026-01-22","2026-01-22T00:00:00Z"),(1.,0))
        factor,n=calibration(history,"2026-01-21","2026-01-24T00:00:00Z")
        self.assertEqual(n,20)
        self.assertAlmostEqual(factor,1.75)

    def test_target_maturity_and_qlike(self):
        t=pd.Timestamp("2026-03-09T14:00Z")
        y=output(2.,.1,1.,2.,t)
        self.assertEqual(y["available_at"],stamp(t+pd.Timedelta(minutes=16)))
        self.assertEqual(y["target_end"],stamp(t+pd.Timedelta(minutes=15)))
        self.assertAlmostEqual(y["loss_c0"],1-np.log(2.))
        self.assertEqual(y["loss_c2"],0.)
        self.assertEqual(output(None,.1,1.,2.,t)["status"],"MISSING_TARGET")
        f=time_fields(t)
        self.assertIsNone(f["actual_receipt"])
        self.assertIsNone(f["actual_commit"])
        self.assertEqual(f["model_asof"],stamp(t))

    def test_append_retry_and_read_only_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"demo.sqlite"
            key=self.key()
            append(path,key,self.panel,self.m)
            append(path,key,self.panel,self.m)
            with reader(path) as db:
                _,rows,counts=validate(db,self.m)
                self.assertEqual(counts["missing_target"],int(self.panel.targets.isna().sum()))
                self.assertEqual(counts["opportunities"],len(self.panel.inputs))
                self.assertEqual(len(rows),counts["paired"])
            with closing(sqlite3.connect(path)) as db, db:
                with self.assertRaisesRegex(sqlite3.IntegrityError,"append only"):
                    db.execute("DELETE FROM outcomes")
                self.assertGreater(max(json.loads(x[0])["history"] for x in db.execute("SELECT payload FROM forecasts")),20)
            a=evaluate(path,Path(d)/"report",self.m)
            self.assertEqual(a["overall"]["n"],counts["paired"])
            self.assertTrue((Path(d)/"report/quality.png").is_file())
            changed=config(MODES[1],self.m["protocol"],{"other":"dataset"},self.m)
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                append(path,changed,self.panel,self.m)

    def test_interrupted_transaction_and_maturity_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"demo.sqlite"
            real=forecast
            counter=iter(range(100000))
            def interrupted(*args):
                if next(counter)==3:
                    raise RuntimeError("interrupted")
                return real(*args)
            with patch("rv15.ledger.forecast",side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    append(path,self.key(),self.panel,self.m)
            with reader(path) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM sessions").fetchone()[0],0)
            append(path,self.key(),self.panel,self.m)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("DROP TRIGGER immutable_outcomes_UPDATE")
                fid,payload=db.execute("SELECT id,payload FROM outcomes LIMIT 1").fetchone()
                outcome=json.loads(payload)
                outcome["available_at"]=outcome["target_end"]
                db.execute("UPDATE outcomes SET payload=? WHERE id=?",(json.dumps(outcome),fid))
                db.execute("""CREATE TRIGGER immutable_outcomes_UPDATE BEFORE UPDATE ON outcomes
                              BEGIN SELECT RAISE(ABORT,'append only'); END""")
            with reader(path) as db:
                with self.assertRaisesRegex(ValueError,"maturity mismatch"):
                    validate(db,self.m)


    def test_external_identity_rejects_substitution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"model.json"
            fake=dict(self.m,c0=C0,c2=C2)
            p.write_text(json.dumps(fake),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"C0 identity mismatch"):
                manifest(p)


if __name__=="__main__":
    unittest.main(verbosity=2)


