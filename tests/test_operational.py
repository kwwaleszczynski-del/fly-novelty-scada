import numpy as np

from fly_novelty.operational import CLEAR_ROWS, _cleared_after, cost_table


def test_cleared_after_counts_until_a_quiet_day():
    flags = np.zeros(1000, bool)
    assert _cleared_after(flags, 100) == 0.0  # never in alarm
    flags[100:244] = True  # one day of alarm, then quiet
    assert _cleared_after(flags, 100) == (243 - 100) / 144
    flags[300:310] = True  # a short relapse within the next day: the first quiet day starts later
    assert _cleared_after(flags, 100) == (309 - 100) / 144
    flags[:] = False
    flags[200:] = True  # alarm to the end: never cleared
    assert np.isnan(_cleared_after(flags, 100))
    flags[:] = False
    flags[1000 - CLEAR_ROWS // 2] = True  # too little data after the last flag to call it cleared
    assert np.isnan(_cleared_after(flags, 100))


def test_cost_table_reports_every_method(wiring, scada):
    t = cost_table(scada[:3000], scada[3000:3500], wiring, n_single=20)
    assert t["method"].tolist()[0].startswith("fly") and len(t) == 6
    fly = t.iloc[0]
    assert fly["learns_online"] and fly["learned_bytes"] == 2 * 8 * wiring.n_kc
    assert (t[["fit_s", "state_bytes", "batch_us_per_row", "single_us"]] > 0).all().all()
