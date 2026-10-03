from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid
import unittest

from live import exposure_reservations as er

_TEST_JOURNAL_ROOT = tempfile.TemporaryDirectory(prefix="pb-exposure-tests-")


class FakeBot:
    def __init__(self, *, balance=100.0, positions=None, open_orders=None, n_positions=7, journal_path=None):
        self.balance = balance
        self.positions = positions or {}
        self.open_orders = open_orders or {}
        self.c_mults = {"SOL/USDT:USDT": 1.0, "XRP/USDT:USDT": 1.0}
        self.fees = {symbol: (0.0, 0.0) for symbol in self.c_mults}
        self.risk = {
            "long": {
                "total_wallet_exposure_limit": 1.5,
                "total_exposure_enforcer_threshold": 0.966,
                "total_exposure_entry_gate_enabled": True,
                "we_excess_allowance_pct": 0.22,
                "n_positions": n_positions,
            },
            "short": {
                "total_wallet_exposure_limit": 0.0,
                "total_exposure_enforcer_threshold": 0.95,
                "total_exposure_entry_gate_enabled": True,
                "n_positions": n_positions,
            },
        }
        if journal_path is not None:
            self._exposure_reservation_journal_path = str(journal_path)
        else:
            self._exposure_reservation_journal_path = str(Path(_TEST_JOURNAL_ROOT.name) / f"{uuid.uuid4().hex}.json")

    def get_raw_balance(self):
        return self.balance

    def bot_value(self, side, key):
        return self.risk[side][key]

    def _get_exchange_fee_rates(self, symbol):
        return self.fees[symbol]

    def _is_market_execution_order(self, order):
        return order.get("type") == "market"


def positions(long_size, symbol="SOL/USDT:USDT", price=100.0):
    return {
        symbol: {
            "long": {"size": long_size, "price": price},
            "short": {"size": 0.0, "price": 0.0},
        }
    }


def entry(symbol="SOL/USDT:USDT", qty=0.1, price=100.0, client_id=None):
    out = {
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "qty": qty,
        "price": price,
        "type": "limit",
        "reduce_only": False,
    }
    if client_id:
        out["custom_id"] = client_id
    return out


def resting(order_id, qty, *, symbol="SOL/USDT:USDT", filled=0.0, price=100.0, client_id=None):
    order = {
        "id": order_id,
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "amount": qty,
        "filled": filled,
        "remaining": qty - filled,
        "qty": qty - filled,
        "price": price,
        "reduceOnly": False,
        "status": "open",
    }
    if client_id:
        order["clientOrderId"] = client_id
    return order


def admit(bot, orders):
    accepted, blocked, reason = er.admit_entry_batch(bot, orders)
    if reason is not None:
        raise AssertionError(reason)
    return accepted, blocked


class ExposureReservationTests(unittest.TestCase):
    def test_one_and_multiple_resting_orders_reserve_exposure(self):
        bot = FakeBot(positions=positions(1.30), open_orders={"SOL/USDT:USDT": [resting("a", 0.08)]})
        accepted, blocked = admit(bot, [entry(qty=0.08, client_id="b")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)
        bot.open_orders["SOL/USDT:USDT"].append(resting("c", 0.06))
        accepted, blocked = admit(bot, [entry(qty=0.01, client_id="d")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)

    def test_two_entries_in_one_batch_are_reserved_atomically(self):
        bot = FakeBot(positions=positions(1.30))
        accepted, blocked = admit(bot, [entry(qty=0.08, client_id="a"), entry(qty=0.08, client_id="b")])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(blocked), 1)

    def test_partial_fill_ratios_and_full_fill(self):
        for fraction in (0.25, 0.50, 0.75, 1.0):
            with self.subTest(fraction=fraction):
                total_qty = 0.14
                filled = total_qty * fraction
                remaining = total_qty - filled
                bot = FakeBot(positions=positions(1.30))
                accepted, blocked = admit(bot, [entry(qty=total_qty, client_id="fill")])
                self.assertEqual(len(accepted), 1)
                self.assertFalse(blocked)
                er.record_create_ack(bot, accepted[0], {"id": "fill-order", "clientOrderId": "fill"})
                bot.positions["SOL/USDT:USDT"]["long"]["size"] += filled
                if remaining:
                    bot.open_orders = {"SOL/USDT:USDT": [resting("fill-order", total_qty, filled=filled, client_id="fill")]}
                else:
                    bot.open_orders = {}
                er.reconcile_authoritative_snapshot(bot)
                observed = bot.positions["SOL/USDT:USDT"]["long"]["size"] + remaining
                self.assertAlmostEqual(observed, 1.30 + total_qty)
                accepted, blocked = admit(bot, [entry(qty=0.01, client_id="next")])
                self.assertFalse(accepted)
                self.assertEqual(len(blocked), 1)

    def test_partial_fill_then_later_complete_fill_is_atomic(self):
        qty = 0.14
        bot = FakeBot(positions=positions(1.30))
        accepted, blocked = admit(bot, [entry(qty=qty, client_id="two-stage")])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(blocked)
        er.record_create_ack(bot, accepted[0], {"id": "two-stage-id", "clientOrderId": "two-stage"})
        partial = qty * 0.25
        bot.positions["SOL/USDT:USDT"]["long"]["size"] += partial
        bot.open_orders = {"SOL/USDT:USDT": [resting("two-stage-id", qty, filled=partial, client_id="two-stage")]}
        er.reconcile_authoritative_snapshot(bot)
        self.assertAlmostEqual(bot.positions["SOL/USDT:USDT"]["long"]["size"] + qty - partial, 1.30 + qty)
        bot.positions["SOL/USDT:USDT"]["long"]["size"] += qty - partial
        bot.open_orders = {}
        er.reconcile_authoritative_snapshot(bot)
        self.assertAlmostEqual(bot.positions["SOL/USDT:USDT"]["long"]["size"], 1.30 + qty)
        accepted, blocked = admit(bot, [entry(qty=0.01, client_id="after-complete")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)

    def test_partial_fill_then_cancel_releases_only_after_coherent_snapshot(self):
        partial = resting("part", 0.12, filled=0.06, client_id="part-client")
        bot = FakeBot(positions=positions(1.36), open_orders={"SOL/USDT:USDT": [partial]})
        er.record_cancel_intent(bot, [partial])
        accepted, blocked = admit(bot, [entry(qty=0.08, client_id="replacement")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)
        bot.open_orders = {}
        er.reconcile_authoritative_snapshot(bot)
        accepted, blocked = admit(bot, [entry(qty=0.08, client_id="replacement-2")])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(blocked)

    def test_replacement_waits_for_cancel_confirmation(self):
        old = resting("old", 0.12)
        bot = FakeBot(positions=positions(1.30), open_orders={"SOL/USDT:USDT": [old]})
        er.record_cancel_intent(bot, [old])
        accepted, blocked = admit(bot, [entry(qty=0.12, client_id="replacement")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)
        bot.open_orders = {}
        er.reconcile_authoritative_snapshot(bot)
        accepted, blocked = admit(bot, [entry(qty=0.12, client_id="replacement-2")])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(blocked)

    def test_pending_symbols_consume_n_positions_across_symbols(self):
        bot = FakeBot(n_positions=1, open_orders={"SOL/USDT:USDT": [resting("sol", 0.01)]})
        candidate = entry(symbol="XRP/USDT:USDT", qty=0.01, price=1.0, client_id="xrp")
        accepted, blocked = admit(bot, [candidate])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)

    def test_restart_rebuilds_reservations_from_positions_and_resting_orders(self):
        pending = resting("restart", 0.20, symbol="XRP/USDT:USDT", price=100.0)
        restarted = FakeBot(positions=positions(1.30), open_orders={"XRP/USDT:USDT": [pending]})
        accepted, blocked = admit(restarted, [entry(symbol="XRP/USDT:USDT", qty=0.10, price=100.0, client_id="after-restart")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)

    def test_restart_restores_ambiguous_reservation_from_private_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "reservations.json"
            first = FakeBot(positions=positions(1.30), journal_path=journal)
            accepted, blocked = admit(first, [entry(qty=0.14, client_id="journal")])
            self.assertEqual(len(accepted), 1)
            self.assertFalse(blocked)
            self.assertTrue(journal.exists())
            self.assertEqual(os.stat(journal).st_mode & 0o777, 0o600)
            stored = json.loads(journal.read_text())
            self.assertEqual(stored["schema_version"], 1)
            self.assertNotIn("key", journal.read_text().lower())

            # Simulate a new process: no exchange open order is visible yet.
            restarted = FakeBot(positions=positions(1.30), journal_path=journal)
            accepted, blocked = admit(restarted, [entry(qty=0.14, client_id="after-crash")])
            self.assertFalse(accepted)
            self.assertEqual(len(blocked), 1)

    def test_acknowledged_reservation_reconciles_and_removes_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "reservations.json"
            bot = FakeBot(positions=positions(1.30), journal_path=journal)
            accepted, blocked = admit(bot, [entry(qty=0.01, client_id="acked")])
            self.assertEqual(len(accepted), 1)
            self.assertFalse(blocked)
            er.record_create_ack(bot, accepted[0], {"id": "exchange-id"})
            bot.open_orders = {"SOL/USDT:USDT": [resting("exchange-id", 0.01, client_id="acked")]}
            er.reconcile_authoritative_snapshot(bot)
            self.assertFalse(journal.exists())

    def test_concurrent_batches_cannot_spend_same_headroom_twice(self):
        bot = FakeBot(positions=positions(1.30))
        barrier = threading.Barrier(2)

        def submit(client_id):
            order = entry(qty=0.08, client_id=client_id)
            barrier.wait()
            return admit(bot, [order])

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, ("race-a", "race-b")))
        self.assertEqual(sum(len(accepted) for accepted, _blocked in results), 1)
        self.assertEqual(sum(len(blocked) for _accepted, blocked in results), 1)

    def test_ambiguous_create_reservation_fails_closed(self):
        bot = FakeBot(positions=positions(1.30))
        accepted, blocked = admit(bot, [entry(qty=0.1, client_id="ambiguous")])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(blocked)
        # No acknowledgement and no authoritative open-order match: keep reserve.
        er.reconcile_authoritative_snapshot(bot)
        accepted, blocked = admit(bot, [entry(qty=0.1, client_id="next")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)

    def test_market_entry_fails_closed_without_blocking_reduce_only(self):
        bot = FakeBot(positions=positions(1.0))
        market = entry(qty=0.1)
        market["type"] = "market"
        close = {"symbol": "SOL/USDT:USDT", "side": "sell", "position_side": "long", "qty": -0.1, "price": 100.0, "reduce_only": True}
        accepted, blocked, reason = er.admit_entry_batch(bot, [market, close])
        self.assertEqual(accepted, [close])
        self.assertEqual(blocked, [market])
        self.assertIsNone(reason)

    def test_fee_reserve_is_included_in_cap(self):
        bot = FakeBot(positions=positions(1.448))
        bot.fees["SOL/USDT:USDT"] = (0.01, 0.02)
        accepted, blocked = admit(bot, [entry(qty=0.001, client_id="fee")])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)

    def test_we_excess_allowance_is_not_applied_twice_to_portfolio_cap(self):
        outcomes = []
        for allowance in (0.0, 0.22, 0.40):
            bot = FakeBot(positions=positions(1.30))
            bot.risk["long"]["we_excess_allowance_pct"] = allowance
            accepted, blocked = admit(bot, [entry(qty=0.14, client_id=f"allow-{allowance}")])
            outcomes.append((len(accepted), len(blocked)))
        self.assertEqual(outcomes, [(1, 0), (1, 0), (1, 0)])

    def test_disabled_side_limit_blocks_entry(self):
        bot = FakeBot()
        short = {"symbol": "SOL/USDT:USDT", "side": "sell", "position_side": "short", "qty": 0.1, "price": 100.0, "type": "limit"}
        accepted, blocked = admit(bot, [short])
        self.assertFalse(accepted)
        self.assertEqual(len(blocked), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
