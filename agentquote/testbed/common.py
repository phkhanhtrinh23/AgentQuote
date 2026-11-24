"""Backbone behaviour shared by the testbed environments."""
from __future__ import annotations

import numpy as np

from ..config import BACKBONE_NAMES, price_in, price_out

NB = len(BACKBONE_NAMES)
PIN = np.array([price_in(b) for b in BACKBONE_NAMES])
POUT = np.array([price_out(b) for b in BACKBONE_NAMES])
TPS = np.array([45.0, 25.0, 70.0])            # output tokens per second
SYS_TOK = np.array([3200, 3200, 3600])        # system prompt and tool schema
OUT_ATT = np.array([700, 900, 600])           # median output tokens of an attempt turn
TURN_SD = 0.25                                # turn-latency noise
CACHE_HIT_P, CACHE_DISC = 0.6, 0.5            # prompt cache of the API backbone on re-read history
REWRITE_P, REWRITE_SD = 0.35, 0.7             # the executing backbone rewrites the draft before attempt 1
FRAIL_X_CORR = 0.6                            # exploration frailty vs failure frailty
CONTINENTS = ("africa", "asia", "europe", "north_america", "south_america", "oceania")


def lognormal(rng, med, sd, size=None):
    return med * np.exp(rng.normal(0, sd, size))


def bill_in(ctx, ctx0, b, rng):
    """Input-token USD of one turn; the API backbone serves re-read history from a prompt cache on some turns."""
    hist = ctx - ctx0
    if b == 2 and rng.random() < CACHE_HIT_P:
        return (ctx - CACHE_DISC * hist) * PIN[b]
    return ctx * PIN[b]
