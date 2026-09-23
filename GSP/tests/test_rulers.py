"""Rulers == brute force (PLAN §5 S1), chunk invariance, and the n = 20 memory bound."""

import itertools
import tracemalloc

import numpy as np
import pytest

from gsp.instances.draws import make_draw
from gsp.instances.encode import encode
from gsp.instances.freeze import draw_band
from gsp.instances.rulers import TIE_RTOL, band, ladder_gap, rulers_from_band


def brute(E, eps):
    """Plain-python enumeration: band by |P.x - 1| <= eps, f(x) = -(x^T QU_obj x)."""
    n = E.n
    idx, f = [], []
    for k, x in enumerate(itertools.product((0, 1), repeat=n)):    # x_0 first = MSB order
        xv = np.array(x, dtype=np.float64)
        if abs(float(E.P_bb @ xv) - 1.0) <= eps:
            idx.append(k)
            f.append(-float(xv @ E.QU_obj @ xv))
    return np.array(idx), np.array(f)


@pytest.mark.parametrize("N,e,q", [(4, 4, 1.0), (4, 6, 3.0), (5, 0, 1.5), (5, 7, 1.0), (6, 0, 1.5)])
def test_rulers_match_brute_force(market, N, e, q):
    d = make_draw(market, N, e)
    E = encode(d.B, d.P, d.ret, d.cov, q)
    r = rulers_from_band(draw_band(d), E.QU_obj)
    idx, f = brute(E, 0.1)
    assert r.F_size == idx.size
    assert np.array_equal(r.band_idx, idx)
    rng = f.max() - f.min()
    assert np.max(np.abs(r.f_band - f)) <= 1e-15 * max(1.0, np.max(np.abs(f))) * 10
    assert r.E_min == pytest.approx(f.min(), abs=1e-15) and r.E_max == pytest.approx(f.max(), abs=1e-15)
    xstar = set(idx[f <= f.min() + TIE_RTOL * rng])
    assert set(r.xstar_idx) == xstar
    order = np.lexsort((idx, f))
    k10 = f[order[min(9, f.size - 1)]]
    top = set(idx[f <= k10 + TIE_RTOL * rng])
    assert set(r.top10_idx) == top and r.top10_size == len(top) >= min(10, f.size)
    # ladder: sorted distinct normalized values, median adjacent gap
    v = np.unique(np.round((f - f.min()) / rng, 12))
    assert r.gap_band == pytest.approx(float(np.median(np.diff(v))), rel=1e-9)
    assert r.n_distinct == v.size
    assert r.n_direct_disagree == 0


def test_rulers_chunk_invariant(market):
    d = make_draw(market, 6, 3)
    E = encode(d.B, d.P, d.ret, d.cov, 1.5)
    a = rulers_from_band(band(E.QU_pen, E.P_bb, 0.1, chunk=1 << 16), E.QU_obj, chunk=1 << 16)
    b = rulers_from_band(band(E.QU_pen, E.P_bb, 0.1, chunk=128), E.QU_obj, chunk=128)
    for field in ("band_idx", "band_pen", "f_band", "xstar_idx", "top10_idx"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert (a.E_min, a.E_max, a.gap_band, a.f_all_min, a.f_all_max) == (b.E_min, b.E_max, b.gap_band,
                                                                       b.f_all_min, b.f_all_max)


def test_ladder_gap_ties_and_degenerate():
    gap, nd = ladder_gap(np.array([0.0, 1.0, 2.0, 3.0]), 0.0, 3.0)
    assert nd == 4 and gap == pytest.approx(1 / 3)
    # 1 + 1e-13 is a distinct double but within TIE_RTOL * range of 1: merged
    gap, nd = ladder_gap(np.array([0.0, 1.0, 1.0 + 1e-13, 3.0]), 0.0, 3.0)
    assert nd == 3 and gap == pytest.approx(0.5)
    assert np.isnan(ladder_gap(np.array([2.0]), 2.0, 2.0)[0])


def test_rulers_n20_memory_bounded(market):
    """n = 20 (N = 10): chunked enumeration; peak traced allocation far below the ~2 GB cap."""
    d = make_draw(market, 10, 0)
    E = encode(d.B, d.P, d.ret, d.cov, 1.5)
    assert E.n == 20
    tracemalloc.start()
    try:
        r = rulers_from_band(draw_band(d), E.QU_obj)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 256 * 2**20, f"peak {peak / 2**20:.0f} MiB"
    assert r.F_size > 0 and r.n == 20 and r.n_direct_disagree == 0
    # spot-check band energies against a direct evaluation
    from gsp.instances.bits import index_to_bits
    x = index_to_bits(r.band_idx[:50], 20).astype(np.float64)
    f = -np.einsum("ki,ij,kj->k", x, E.QU_obj, x)
    assert np.max(np.abs(f - r.f_band[:50])) <= 1e-17 + 1e-14 * np.max(np.abs(f))
