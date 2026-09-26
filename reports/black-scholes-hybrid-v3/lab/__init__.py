"""Single-core numerical experiments for the Black--Scholes book."""

from .black_scholes import (
    DetailedPrice,
    PricePair,
    naive_prices,
    stable_detailed,
    stable_prices,
)

__all__ = [
    "DetailedPrice",
    "PricePair",
    "naive_prices",
    "stable_detailed",
    "stable_prices",
]
