"""Explicit safe account/risk inputs for tests that exercise order execution."""

from pathlib import Path
import tempfile
import uuid


def provide_exposure_state(bot, orders, *, balance=100_000.0):
    """Give an executor fake a coherent, deliberately roomy account snapshot.

    The strategy cap remains the configured 1.5x; this fixture balance merely
    ensures these tests reach their intended connector/bookkeeping path.
    Each bot gets an isolated reservation journal under the system temp dir.
    """
    symbols = {
        str(order.get("symbol"))
        for order in orders
        if isinstance(order, dict) and order.get("symbol")
    }
    bot.exchange = getattr(bot, "exchange", "fake")
    bot.user = getattr(bot, "user", f"test-{uuid.uuid4().hex}")
    bot.balance = float(balance)
    bot.get_raw_balance = lambda: float(balance)
    bot.positions = {symbol: {"long": None, "short": None} for symbol in symbols}
    bot.open_orders = {symbol: [] for symbol in symbols}
    bot.c_mults = {symbol: 1.0 for symbol in symbols}
    bot._get_exchange_fee_rates = lambda _symbol: (0.0, 0.0)
    original_bot_value = getattr(bot, "bot_value", None)
    risk_values = {
        "n_positions": 7,
        "total_wallet_exposure_limit": 1.5,
        "total_exposure_enforcer_threshold": 1.0,
        "total_exposure_entry_gate_enabled": False,
    }

    def bot_value(pside, name):
        if name in risk_values:
            return risk_values[name]
        if callable(original_bot_value):
            return original_bot_value(pside, name)
        return None

    bot.bot_value = bot_value
    tempdir = tempfile.TemporaryDirectory(prefix="passivbot-exposure-test-")
    bot._exposure_reservation_test_tempdir = tempdir
    bot._exposure_reservation_journal_path = (
        Path(tempdir.name) / "exposure_reservations.json"
    )
    return bot
