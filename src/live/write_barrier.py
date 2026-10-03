"""Fail-closed local exchange write barrier for isolated runtime inspection.

Set ``PASSIVBOT_WRITE_BARRIER=1`` before starting Passivbot to allow an explicit
set of read APIs while recording and blocking every other client operation.
This is an application guard, not an OS/network sandbox.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import inspect
import json
import logging
import os
from typing import Any


WRITE_BARRIER_ENV = "PASSIVBOT_WRITE_BARRIER"


class WriteBlockedError(RuntimeError):
    """An exchange mutation or unrecognized client operation was denied."""

    code = "WRITE_BLOCKED"

    def __init__(self, operation: str, *, exchange: str = "unknown"):
        self.operation = str(operation)[:100]
        self.exchange = str(exchange)[:40]
        super().__init__(f"WRITE_BLOCKED: {self.operation} ({self.exchange})")


_active_operation: ContextVar[str | None] = ContextVar(
    "passivbot_write_barrier_operation", default=None
)

# Exact CCXT read entry points used by Passivbot and its exchange adapters.
# Unknown callables are denied, even if their name appears read-like.
_READ_METHODS = frozenset(
    {
        "close",
        "load_markets",
        "fetch",
        "fetch_balance",
        "fetch_bids_asks",
        "fetch_closed_orders",
        "fetch_currencies",
        "fetch_funding_history",
        "fetch_funding_rate",
        "fetch_funding_rates",
        "fetch_leverage",
        "fetch_leverage_margin_mode",
        "fetch_margin_mode",
        "fetch_markets",
        "fetch_my_trades",
        "fetch_ohlcv",
        "fetch_open_orders",
        "fetch_order",
        "fetch_order_book",
        "fetch_orders",
        "fetch_position_mode",
        "fetch_positions",
        "fetch_positions_history",
        "fetch_status",
        "fetch_ticker",
        "fetch_tickers",
        "fetch_time",
        "fetch_trading_fee",
        "private_get_account_config",
        "private_get_v5_position_closed_pnl",
        "private_mix_get_v2_mix_order_detail",
        "private_mix_get_v2_mix_order_fill_history",
        "private_uta_get_v3_account_assets",
        "private_uta_get_v3_trade_fills",
        "private_uta_get_v3_trade_order_info",
        "fapiprivate_get_apireferral_ifnewuser",
        "fapiprivate_get_income",
        "fapiprivatev3_get_positionrisk",
        "fapipublic_get_ticker_bookticker",
        "contract_get_capi_v3_market_ticker_bookticker",
        "publicPostInfo",
        "watch_balance",
        "watch_ohlcv",
        "watch_order_book",
        "watch_orders",
        "watch_ticker",
        "watch_tickers",
        "watch_trades",
    }
)

_SAFE_LOCAL_METHODS = frozenset(
    {
        "market",
        "market_id",
        "safe_currency_code",
        "safe_integer",
        "safe_integer_2",
        "safe_number",
        "safe_number_2",
        "safe_string",
        "safe_string_2",
        "safe_value",
        "safe_value_2",
        "milliseconds",
        "iso8601",
        "parse_timeframe",
        "price_to_precision",
        "amount_to_precision",
        "cost_to_precision",
        "decimal_to_precision",
        "precision_from_string",
        "precision_to_string",
        "precision_mode_string",
        "safe_market",
        "safe_market_structure",
        "safe_symbol",
        "safe_timestamp",
        "safe_timestamp_2",
        "safe_trade",
        "safe_order",
        "safe_position",
        "safe_balance",
    }
)

_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_SAFE_HYPERLIQUID_INFO_POST_TYPES = frozenset(
    {"allMids", "metaAndAssetCtxs", "userAbstraction"}
)
_SENSITIVE_ATTRIBUTES = frozenset(
    {
        "apiKey",
        "secret",
        "password",
        "token",
        "privateKey",
        "walletAddress",
        "accountId",
        "session",
        "aiohttp_session",
        "socks_proxy_sessions",
        "websocket",
        "websocket_client",
        "last_request_headers",
        "last_request",
        "headers",
    }
)
_MUTATING_NAME_PREFIXES = (
    "create_",
    "cancel_",
    "edit_",
    "set_",
    "add_",
    "reduce_",
    "transfer",
    "withdraw",
    "deposit",
    "borrow",
    "repay",
    "place_",
    "submit_",
    "modify_",
    "delete_",
    "remove_",
    "update_",
)


def write_barrier_requested() -> bool:
    """Return whether the barrier is requested in the current process env."""
    return os.environ.get(WRITE_BARRIER_ENV) == "1"


def _caller_name() -> str:
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame else None
        while frame:
            module = str(frame.f_globals.get("__name__", ""))
            if module != __name__:
                return str(frame.f_code.co_name)[:100]
            frame = frame.f_back
    finally:
        del frame
    return "unknown"


def _safe_details(operation: str, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Extract only bounded order/config metadata; never serialize params/headers."""
    details: dict[str, Any] = {}
    values = dict(kwargs)
    if args:
        if operation == "create_order":
            for key, value in zip(("symbol", "order_type", "side", "quantity"), args):
                values.setdefault(key, value)
        elif operation == "create_limit_order":
            for key, value in zip(("symbol", "side", "quantity"), args):
                values.setdefault(key, value)
            values.setdefault("order_type", "limit")
        elif operation == "create_market_order":
            for key, value in zip(("symbol", "side", "quantity"), args):
                values.setdefault(key, value)
            values.setdefault("order_type", "market")
        elif operation in {"cancel_order", "edit_order"}:
            if len(args) > 1:
                values.setdefault("symbol", args[1])
            if operation == "edit_order":
                for key, value in zip(("_id", "symbol", "order_type", "side", "quantity"), args):
                    values.setdefault(key, value)
        elif operation in {"set_leverage", "set_margin_mode", "set_position_mode"}:
            if operation != "set_position_mode" and len(args) > 1:
                values.setdefault("symbol", args[1])
    aliases = {
        "symbol": ("symbol",),
        "side": ("side", "order_side"),
        "order_type": ("order_type", "type"),
        "quantity": ("quantity", "amount", "qty"),
    }
    for output_key, candidate_keys in aliases.items():
        value = next((values[key] for key in candidate_keys if key in values), None)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            details[output_key] = str(value)[:80] if isinstance(value, str) else value
    return details


def record_write_blocked(
    operation: str,
    *,
    exchange: str = "unknown",
    caller: str | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
) -> dict[str, Any]:
    """Log a bounded, credential-free WRITE_BLOCKED record and return it."""
    record = {
        "status": "WRITE_BLOCKED",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "caller": (caller or _caller_name())[:100],
        "operation": str(operation)[:100],
        "exchange": str(exchange)[:40],
        **_safe_details(str(operation), args, kwargs or {}),
    }
    logging.error("[write-barrier] %s", json.dumps(record, sort_keys=True))
    return record


def block_write(
    operation: str,
    *,
    exchange: str = "unknown",
    caller: str | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
) -> None:
    record_write_blocked(
        operation, exchange=exchange, caller=caller, args=args, kwargs=kwargs
    )
    raise WriteBlockedError(operation, exchange=exchange)


def record_order_plan_blocked(bot: object, cancels: list[dict], creates: list[dict]) -> None:
    """Record planned actions before the executor exits without any exchange call."""
    exchange = str(getattr(bot, "exchange", "unknown"))
    for action, orders in (("cancel_order", cancels), ("create_order", creates)):
        for order in orders:
            if isinstance(order, dict):
                record_write_blocked(
                    action,
                    exchange=exchange,
                    caller="execute_order_plan",
                    kwargs=order,
                )
    create_symbols = sorted(
        {
            str(order["symbol"])
            for order in creates
            if isinstance(order, dict) and order.get("symbol")
        }
    )
    for symbol in create_symbols:
        for operation in ("set_leverage", "set_margin_mode"):
            record_write_blocked(
                operation,
                exchange=exchange,
                caller="execute_order_plan",
                kwargs={"symbol": symbol},
            )


def _http_method_allowed(method: str, *, exchange: str, body: Any) -> bool:
    method = str(method or "GET").upper()
    if method in _SAFE_HTTP_METHODS:
        return True
    if method != "POST" or exchange.lower() not in {"hyperliquid", "hyperliquidusdm"}:
        return False
    try:
        payload = json.loads(body) if isinstance(body, str) else body
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("type") in _SAFE_HYPERLIQUID_INFO_POST_TYPES


def _guard_ccxt_transport(client: object, exchange: str) -> None:
    """Guard the final CCXT REST boundary as defense in depth."""
    if getattr(client, "_passivbot_write_barrier_transport", False):
        return
    original_fetch = getattr(client, "fetch", None)
    if callable(original_fetch):
        async def guarded_fetch(url, method="GET", headers=None, body=None):
            if not _http_method_allowed(method, exchange=exchange, body=body):
                block_write(
                    f"http_{str(method).lower()}",
                    exchange=exchange,
                    caller="ccxt.fetch",
                )
            if _active_operation.get() is None:
                block_write("unclassified_http_request", exchange=exchange, caller="ccxt.fetch")
            return await original_fetch(url, method, headers, body)

        client.fetch = guarded_fetch
    original_fetch2 = getattr(client, "fetch2", None)
    if callable(original_fetch2):
        async def guarded_fetch2(
            path,
            api="public",
            method="GET",
            params=None,
            headers=None,
            body=None,
            config=None,
        ):
            if not _http_method_allowed(method, exchange=exchange, body=body):
                block_write(
                    f"ccxt_{str(method).lower()}",
                    exchange=exchange,
                    caller="ccxt.fetch2",
                    kwargs={"operation": str(path)[:100]},
                )
            if _active_operation.get() is None:
                block_write(
                    "unclassified_ccxt_request",
                    exchange=exchange,
                    caller="ccxt.fetch2",
                )
            return await original_fetch2(
                path, api, method, params or {}, headers, body, config or {}
            )

        client.fetch2 = guarded_fetch2
    client._passivbot_write_barrier_transport = True


class WriteBarrierClient:
    """Proxy that allows known reads/local helpers and denies all unknown calls."""

    def __init__(self, client: object, exchange: str):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_exchange", str(exchange or "unknown"))
        _guard_ccxt_transport(client, str(exchange or "unknown"))

    def __getattribute__(self, name: str) -> Any:
        if name in {"_client", "_exchange", "__dict__"}:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        if name in _SENSITIVE_ATTRIBUTES:
            raise AttributeError(name)
        target = object.__getattribute__(self, "_client")
        try:
            value = getattr(target, name)
        except AttributeError:
            if not (
                name.startswith(_MUTATING_NAME_PREFIXES)
                or "write" in name.lower()
                or name.startswith(("private_post_", "public_post_"))
            ):
                raise
            def blocked_missing(*args, **kwargs):
                block_write(
                    name,
                    exchange=object.__getattribute__(self, "_exchange"),
                    args=args,
                    kwargs=kwargs,
                )
            blocked_missing.__name__ = name
            return blocked_missing
        if not callable(value):
            return value
        if name not in _READ_METHODS and name not in _SAFE_LOCAL_METHODS:
            def blocked(*args, **kwargs):
                block_write(
                    name,
                    exchange=object.__getattribute__(self, "_exchange"),
                    args=args,
                    kwargs=kwargs,
                )
            blocked.__name__ = name
            return blocked
        if inspect.iscoroutinefunction(value):
            async def allowed(*args, **kwargs):
                token = _active_operation.set(name)
                try:
                    if name == "fetch":
                        method = kwargs.get("method", args[1] if len(args) > 1 else "GET")
                        body = kwargs.get("body", args[3] if len(args) > 3 else None)
                        exchange = object.__getattribute__(self, "_exchange")
                        if (
                            exchange.lower() not in {"hyperliquid", "hyperliquidusdm"}
                            or not _http_method_allowed(method, exchange=exchange, body=body)
                        ):
                            block_write(
                                "unclassified_raw_fetch",
                                exchange=exchange,
                                args=args,
                                kwargs=kwargs,
                            )
                    return await value(*args, **kwargs)
                finally:
                    _active_operation.reset(token)
            allowed.__name__ = name
            return allowed
        def allowed_sync(*args, **kwargs):
            token = _active_operation.set(name)
            try:
                result = value(*args, **kwargs)
            except BaseException:
                _active_operation.reset(token)
                raise
            if inspect.isawaitable(result):
                async def await_result():
                    try:
                        return await result
                    finally:
                        _active_operation.reset(token)

                return await_result()
            _active_operation.reset(token)
            return result
        allowed_sync.__name__ = name
        return allowed_sync

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"write-barrier client is read-only: {name}")

    def __repr__(self) -> str:
        exchange = object.__getattribute__(self, "_exchange")
        return f"<WriteBarrierClient exchange={exchange!r} status=ACTIVE>"


def guard_client(client: object | None, exchange: str) -> object | None:
    """Wrap an exchange client when the barrier was requested at startup."""
    if client is None or not write_barrier_requested():
        return client
    if isinstance(client, WriteBarrierClient):
        return client
    return WriteBarrierClient(client, exchange)


def client_is_guarded(client: object | None) -> bool:
    """Whether a client is wrapped by the active deny-all-write proxy."""
    return isinstance(client, WriteBarrierClient)


def guard_native_http_request(
    exchange: str, method: str, path: str, *, body: dict | None = None
) -> None:
    """Fail-closed gate for the repository's native Bitunix HTTP transport."""
    if not write_barrier_requested():
        return
    read_paths = {
        "/api/v1/futures/account",
        "/api/v1/futures/account/get_leverage_margin_mode",
        "/api/v1/futures/market/depth",
        "/api/v1/futures/market/kline",
        "/api/v1/futures/market/trading_pairs",
        "/api/v1/futures/position/get_pending_positions",
        "/api/v1/futures/trade/get_history_trades",
        "/api/v1/futures/trade/get_order_detail",
        "/api/v1/futures/trade/get_pending_orders",
    }
    if str(method).upper() != "GET" or path not in read_paths:
        block_write(
            f"native_http_{str(method).lower()}",
            exchange=exchange,
            caller="Bitunix._request",
            kwargs=body or {},
        )
