import json
import os
import unittest
from unittest.mock import AsyncMock

from live.write_barrier import (
    WriteBlockedError,
    guard_client,
    guard_native_http_request,
    record_order_plan_blocked,
)


class MockExchangeClient:
    def __init__(self):
        self.read_calls = []
        self.external_calls = []
        self.has = {"fetchBalance": True}
        self.options = {}

    async def fetch_balance(self):
        self.read_calls.append("fetch_balance")
        return {"USDT": 100.0}

    async def fetch_positions(self, symbols=None):
        self.read_calls.append("fetch_positions")
        return await self.fetch2("private/position/list", "private", "GET")

    async def fetch_open_orders(self, symbol=None):
        self.read_calls.append("fetch_open_orders")
        return []

    async def fetch_ticker(self, symbol):
        self.read_calls.append("fetch_ticker")
        return {"symbol": symbol}

    async def fetch_ohlcv(self, symbol, timeframe="1m"):
        self.read_calls.append("fetch_ohlcv")
        return []

    async def fetch_my_trades(self, symbol=None):
        self.read_calls.append("fetch_my_trades")
        return []

    def private_get_v5_position_closed_pnl(self, params=None):
        # CCXT implicit API methods are regular functions returning coroutines.
        return self._fetch_closed_pnl(params)

    async def _fetch_closed_pnl(self, params=None):
        self.read_calls.append("private_get_v5_position_closed_pnl")
        return await self.fetch2("private/v5/position/closed-pnl", "private", "GET")

    async def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.external_calls.append(("create_order", symbol))

    async def cancel_order(self, order_id, symbol=None, params=None):
        self.external_calls.append(("cancel_order", symbol))

    async def set_leverage(self, leverage, symbol=None):
        self.external_calls.append(("set_leverage", symbol))

    async def set_margin_mode(self, margin_mode, symbol=None):
        self.external_calls.append(("set_margin_mode", symbol))

    async def set_position_mode(self, hedged):
        self.external_calls.append(("set_position_mode", hedged))

    async def fetch2(
        self,
        path,
        api="public",
        method="GET",
        params=None,
        headers=None,
        body=None,
        config=None,
    ):
        self.external_calls.append(("fetch2", method))
        return await self.fetch("https://unit.test/" + path, method, headers, body)

    async def fetch(self, url, method="GET", headers=None, body=None):
        self.external_calls.append(("fetch", method))
        return {"ok": True}

    async def close(self):
        return None


class WriteBarrierTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous = os.environ.get("PASSIVBOT_WRITE_BARRIER")
        os.environ["PASSIVBOT_WRITE_BARRIER"] = "1"
        self.client = MockExchangeClient()
        self.guarded = guard_client(self.client, "bybit")

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("PASSIVBOT_WRITE_BARRIER", None)
        else:
            os.environ["PASSIVBOT_WRITE_BARRIER"] = self.previous

    async def test_known_order_and_account_writes_are_blocked(self):
        calls = [
            ("create_order", ("SOL/USDT", "limit", "buy", 1.0, 100.0), {}),
            ("cancel_order", ("id", "SOL/USDT"), {}),
            ("set_leverage", (5, "SOL/USDT"), {}),
            ("set_margin_mode", ("isolated", "SOL/USDT"), {}),
            ("set_position_mode", (True,), {}),
        ]
        for name, args, kwargs in calls:
            with self.subTest(name=name):
                with self.assertRaises(WriteBlockedError) as raised:
                    await getattr(self.guarded, name)(*args, **kwargs)
                self.assertEqual(raised.exception.code, "WRITE_BLOCKED")
        self.assertEqual(self.client.external_calls, [])

    async def test_order_variants_and_unknown_operations_fail_closed(self):
        calls = [
            ("create_limit_order", ("SOL/USDT", "buy", 1.0, 100.0)),
            ("create_market_order", ("SOL/USDT", "buy", 1.0)),
            ("create_conditional_order", ("SOL/USDT",)),
            ("cancel_all_orders", ()),
            ("edit_order", ("id", "SOL/USDT")),
            ("withdraw", ("USDT", 1.0, "address")),
            ("private_post_v5_order_create", ()),
            ("unknown_write_operation", ()),
        ]
        for name, args in calls:
            with self.subTest(name=name):
                with self.assertRaises(WriteBlockedError) as raised:
                    await getattr(self.guarded, name)(*args)
                self.assertEqual(raised.exception.code, "WRITE_BLOCKED")
        self.assertEqual(self.client.external_calls, [])

    async def test_allowlisted_read_operations_continue(self):
        reads = [
            ("fetch_balance", ()),
            ("fetch_positions", ()),
            ("fetch_open_orders", ()),
            ("fetch_ticker", ("SOL/USDT",)),
            ("fetch_ohlcv", ("SOL/USDT",)),
            ("fetch_my_trades", ()),
        ]
        for name, args in reads:
            with self.subTest(name=name):
                await getattr(self.guarded, name)(*args)
        self.assertEqual(
            self.client.read_calls,
            [name for name, _ in reads],
        )
        self.assertEqual(self.client.external_calls, [("fetch2", "GET"), ("fetch", "GET")])

    async def test_ccxt_sync_factory_returning_coroutine_keeps_read_context(self):
        result = await self.guarded.private_get_v5_position_closed_pnl({"limit": 1})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.client.read_calls, ["private_get_v5_position_closed_pnl"])
        self.assertEqual(self.client.external_calls, [("fetch2", "GET"), ("fetch", "GET")])

    async def test_unclassified_transport_and_http_mutation_are_blocked(self):
        await self.guarded.fetch_positions()
        with self.assertRaises(WriteBlockedError):
            await self.client.fetch2("private/order/create", "private", "POST")
        with self.assertRaises(WriteBlockedError):
            await self.client.fetch("https://unit.test/unknown", "GET")

    async def test_order_executor_stops_before_processing_a_nonempty_plan(self):
        from live.executor import execute_order_plan

        bot = type("Bot", (), {"exchange": "bybit", "write_barrier_active": True})()
        with self.assertRaises(WriteBlockedError) as raised:
            await execute_order_plan(
                bot,
                [{"symbol": "SOL/USDT", "side": "sell", "qty": 1.0}],
                [{"symbol": "XRP/USDT", "side": "buy", "type": "limit", "qty": 2.0}],
            )
        self.assertEqual(raised.exception.code, "WRITE_BLOCKED")
        self.assertEqual(self.client.external_calls, [])

    async def test_pinned_ccxt_bybit_client_is_guarded_without_network(self):
        import ccxt.async_support as ccxt_async

        client = ccxt_async.bybit({})
        guarded = guard_client(client, "bybit")
        with self.assertRaises(WriteBlockedError):
            await guarded.create_order("SOL/USDT", "limit", "buy", 1.0, 100.0)
        with self.assertRaises(WriteBlockedError):
            await client.fetch2("v5/order/create", "private", "POST")

    def test_disabled_environment_preserves_original_client(self):
        os.environ.pop("PASSIVBOT_WRITE_BARRIER", None)
        client = MockExchangeClient()
        self.assertIs(guard_client(client, "bybit"), client)

    def test_credentials_and_transport_handles_are_not_exposed_by_proxy(self):
        self.client.apiKey = "fake-key-for-test"
        self.client.secret = "fake-secret-for-test"
        self.client.session = object()
        for name in ("apiKey", "secret", "session", "headers"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    getattr(self.guarded, name)

    async def test_block_log_redacts_client_credentials_and_request_params(self):
        with self.assertLogs(level="ERROR") as captured:
            with self.assertRaises(WriteBlockedError):
                await self.guarded.create_order(
                    "SOL/USDT",
                    "limit",
                    "buy",
                    1.0,
                    100.0,
                    {"apiKey": "request-key", "secret": "request-secret"},
                )
        text = "\n".join(captured.output)
        for secret in (
            "fake-key-for-test",
            "fake-secret-for-test",
            "request-key",
            "request-secret",
        ):
            self.assertNotIn(secret, text)

    def test_native_bitunix_read_allowlist_and_fail_closed(self):
        for path in (
            "/api/v1/futures/account",
            "/api/v1/futures/market/kline",
            "/api/v1/futures/trade/get_pending_orders",
        ):
            with self.subTest(path=path):
                guard_native_http_request("bitunix", "GET", path)
        for method, path in (
            ("POST", "/api/v1/futures/trade/place_order"),
            ("POST", "/api/v1/futures/trade/cancel_orders"),
            ("POST", "/api/v1/futures/account/change_leverage"),
            ("GET", "/api/v1/futures/unknown/future-write"),
        ):
            with self.subTest(method=method, path=path):
                with self.assertRaises(WriteBlockedError):
                    guard_native_http_request("bitunix", method, path)

    async def test_native_bitunix_write_is_blocked_before_session_or_network(self):
        from exchanges.bitunix import BitunixClient

        client = object.__new__(BitunixClient)
        client.exchange = "bitunix"
        client.rest_url = "https://unit.test"
        client.headers = {}
        client._throttle = AsyncMock()
        client._get_session = AsyncMock()
        with self.assertRaises(WriteBlockedError):
            await client._request(
                "POST",
                "/api/v1/futures/trade/place_order",
                body={"symbol": "SOLUSDT"},
                private=True,
            )
        client._throttle.assert_not_awaited()
        client._get_session.assert_not_awaited()

    def test_hyperliquid_readonly_post_allowlist(self):
        from live.write_barrier import _http_method_allowed

        self.assertTrue(
            _http_method_allowed(
                "POST", exchange="hyperliquid", body=json.dumps({"type": "allMids"})
            )
        )
        self.assertFalse(
            _http_method_allowed(
                "POST", exchange="hyperliquid", body=json.dumps({"type": "order"})
            )
        )

    async def test_hyperliquid_info_read_post_is_allowed_but_order_post_is_blocked(self):
        client = MockExchangeClient()
        guarded = guard_client(client, "hyperliquid")
        await guarded.fetch(
            "https://unit.test/info",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"type": "allMids"}),
        )
        self.assertEqual(client.external_calls, [("fetch", "POST")])
        with self.assertRaises(WriteBlockedError):
            await guarded.fetch(
                "https://unit.test/exchange",
                method="POST",
                body=json.dumps({"type": "order"}),
            )

    def test_planned_actions_are_logged_without_client_mutation(self):
        bot = type("Bot", (), {"exchange": "bybit"})()
        record_order_plan_blocked(
            bot,
            [{"symbol": "SOL/USDT", "side": "sell", "qty": 1.0}],
            [{"symbol": "XRP/USDT", "side": "buy", "type": "limit", "qty": 2.0}],
        )
        self.assertEqual(self.client.external_calls, [])


if __name__ == "__main__":
    unittest.main()
