"""Unit tests for the interval-hinge losses and the two-head formulation."""

import torch

from cyp.losses import (
    LOG2,
    TDI_INFERRED,
    TDI_MU_THRESHOLD,
    interval_hinge,
    make_intervals,
    tdi_label_from_arms,
    two_head_loss,
    width_weighted_l1,
)


def test_interval_hinge_zero_inside_l1_outside():
    pred = torch.tensor([5.0, 3.0, 7.0])
    lo = torch.tensor([4.0, 4.0, 4.0])
    hi = torch.tensor([6.0, 6.0, 6.0])
    per = interval_hinge(pred, lo, hi, reduction="none")
    # 5.0 is inside [4,6] -> 0; 3.0 is 1.0 below; 7.0 is 1.0 above.
    assert torch.allclose(per, torch.tensor([0.0, 1.0, 1.0]))


def test_interval_hinge_open_bounds_contribute_nothing():
    pred = torch.tensor([100.0, -100.0])
    lo = torch.tensor([float("-inf"), 4.0])
    hi = torch.tensor([6.0, float("inf")])
    per = interval_hinge(pred, lo, hi, reduction="none")
    # 100 vs (-inf, 6]  -> 94 above ; -100 vs [4, inf) -> 104 below
    assert torch.allclose(per, torch.tensor([94.0, 104.0]))


def test_make_intervals_never_lo_gt_hi():
    point = torch.tensor([5.0, 5.0, 5.0, 5.0])
    # row 2 is deliberately malformed (lo > hi); row 3 has NaN bounds.
    lo = torch.tensor([4.0, 4.5, 6.0, float("nan")])
    hi = torch.tensor([6.0, 5.5, 4.0, float("nan")])
    out_lo, out_hi = make_intervals(point, lo, hi, widen=0.25)
    assert torch.all(out_lo <= out_hi)
    # NaN bounds fall back to the point estimate (± widen).
    assert torch.allclose(out_lo[3], torch.tensor(4.75))
    assert torch.allclose(out_hi[3], torch.tensor(5.25))


def test_missing_direct_arm_contributes_zero_gradient_to_mu():
    # Two rows; the second has NO direct arm (mask=0, NaN bounds).
    mu = torch.tensor([5.0, 5.0], requires_grad=True)
    delta = torch.zeros(2)
    direct_lo = torch.tensor([4.0, float("nan")])
    direct_hi = torch.tensor([6.0, float("nan")])
    direct_point = torch.tensor([5.0, float("nan")])
    direct_mask = torch.tensor([True, False])

    # Direct head only (no TDI term): the masked row must not touch mu.
    loss = two_head_loss(
        mu, delta, direct_lo, direct_hi, direct_mask,
        direct_point=direct_point, shift_prior=0.0,
    )
    loss.backward()
    assert mu.grad is not None
    assert mu.grad[1].item() == 0.0


def test_tdi_arm_supervises_mu_on_direct_less_rows():
    # The same row, now WITH a measured TDI arm, SHOULD move mu (via mu+delta).
    mu = torch.tensor([5.0], requires_grad=True)
    delta = torch.tensor([0.5], requires_grad=True)
    direct_lo = torch.tensor([float("nan")])
    direct_hi = torch.tensor([float("nan")])
    direct_mask = torch.tensor([False])
    tdi_lo = torch.tensor([7.0])          # mu+delta = 5.5 is below the interval
    tdi_hi = torch.tensor([8.0])
    tdi_mask = torch.tensor([True])
    loss = two_head_loss(
        mu, delta, direct_lo, direct_hi, direct_mask,
        tdi_lo=tdi_lo, tdi_hi=tdi_hi, tdi_mask=tdi_mask, shift_prior=0.0,
    )
    loss.backward()
    assert mu.grad[0].item() != 0.0        # mu is identified through the TDI arm


def test_width_pull_gives_gradient_inside_wide_interval():
    # Prediction sits inside a wide interval -> hinge is flat (zero gradient),
    # but the width-weighted L1 pull toward the point must still move it.
    pred = torch.tensor([5.0], requires_grad=True)
    point = torch.tensor([4.0])
    width = torch.tensor([3.0])
    loss = width_weighted_l1(pred, point, width)
    loss.backward()
    assert abs(pred.grad.item()) > 0.0


def test_tdi_label_rule_matches_definition():
    mu = torch.tensor([5.0, 5.0, 3.0, 3.0])
    delta = torch.tensor([0.5, 0.1, 2.0, 0.1])
    got = tdi_label_from_arms(mu, delta, hard=True)
    # row0: mu>=4 & delta>0.301 -> pos
    # row1: mu>=4 & delta<=0.301 -> neg
    # row2: mu<4 & mu+delta=5.0>4.301 -> inferred pos
    # row3: mu<4 & mu+delta=3.1<4.301 -> assigned neg
    assert got.tolist() == [True, False, True, False]


def test_label_constants():
    assert TDI_MU_THRESHOLD == 4.0
    assert abs(TDI_INFERRED - (TDI_MU_THRESHOLD + LOG2)) < 1e-9
