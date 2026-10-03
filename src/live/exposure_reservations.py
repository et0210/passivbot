"""Fail-closed portfolio exposure reservations for live order creation.

This is a final execution-admission guard. Strategy sizing remains in Rust; this
module accounts for already executed positions, exchange resting orders, and
orders admitted by concurrent executor batches but not yet reflected by a
coherent account snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

from live.reconciler import extract_order_remaining_qty


class ExposureStateUnavailable(RuntimeError):
    """Required exposure inputs are missing or invalid; new entries fail closed."""


def _finite_positive(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExposureStateUnavailable(f"{label} unavailable") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ExposureStateUnavailable(f"{label} invalid")
    return result


def _order_identities(order: dict) -> set[str]:
    identities = set()
    sources = [order]
    info = order.get("info")
    if isinstance(info, dict):
        sources.append(info)
    for source in sources:
        for key in ("id", "clientOrderId", "client_id", "custom_id", "orderLinkId"):
            value = source.get(key)
            if value not in (None, ""):
                identities.add(f"{key}:{value}")
    return identities


def _position_side(order: dict) -> str | None:
    for source in (order, order.get("info", {})):
        if not isinstance(source, dict):
            continue
        value = source.get("position_side") or source.get("positionSide") or source.get("posSide")
        if value:
            value = str(value).lower()
            if value in {"long", "short"}:
                return value
            if value in {"both", "net"}:
                side = str(source.get("side") or "").lower()
                return "long" if side == "buy" else "short" if side == "sell" else None
    return None


def _is_entry(order: dict, pside: str | None = None) -> bool:
    if bool(order.get("reduce_only") or order.get("reduceOnly")):
        return False
    pside = pside or _position_side(order)
    side = str(order.get("side") or "").lower()
    if pside == "long":
        return side == "buy"
    if pside == "short":
        return side == "sell"
    return False


def _get_value(bot, pside: str, names: tuple[str, ...], default=None):
    for name in names:
        try:
            value = bot.bot_value(pside, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _fee_rate(bot, symbol: str) -> float:
    getter = getattr(bot, "_get_exchange_fee_rates", None)
    if not callable(getter):
        raise ExposureStateUnavailable(f"fee rates unavailable for {symbol}")
    try:
        maker, taker = getter(symbol)
        maker, taker = float(maker), float(taker)
    except Exception as exc:
        raise ExposureStateUnavailable(f"fee rates unavailable for {symbol}") from exc
    if not math.isfinite(maker) or not math.isfinite(taker):
        raise ExposureStateUnavailable(f"fee rates invalid for {symbol}")
    return max(0.0, maker, taker)


def _journal_path(bot) -> Path:
    explicit = getattr(bot, "_exposure_reservation_journal_path", None)
    if explicit:
        return Path(explicit).expanduser()
    exchange = str(getattr(bot, "exchange", "") or "")
    user = str(getattr(bot, "user", "") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", exchange) or not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        raise ExposureStateUnavailable("reservation journal identity unavailable")
    monitor = getattr(bot, "config", {}).get("monitor", {})
    root = Path(str(monitor.get("root_dir") or "monitor")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root / exchange / user / "exposure_reservations.json"


def _journal_order(order: dict) -> dict:
    """Persist only order facts needed to reserve and match an in-flight entry."""
    keys = (
        "id", "clientOrderId", "client_id", "custom_id", "orderLinkId",
        "symbol", "side", "position_side", "positionSide", "posSide",
        "qty", "amount", "remaining", "filled", "price", "type",
        "reduce_only", "reduceOnly",
    )
    return {
        key: value
        for key in keys
        if (value := order.get(key)) is None or isinstance(value, (str, bool, int, float))
    }


def _notional(bot, symbol: str, qty: float, price: float) -> float:
    c_mults = getattr(bot, "c_mults", None)
    if not isinstance(c_mults, dict) or symbol not in c_mults:
        raise ExposureStateUnavailable(f"contract multiplier unavailable for {symbol}")
    c_mult = _finite_positive(c_mults[symbol], f"contract multiplier for {symbol}")
    qty = _finite_positive(abs(qty), f"quantity for {symbol}")
    price = _finite_positive(price, f"price for {symbol}")
    result = qty * price * c_mult
    if not math.isfinite(result):
        raise ExposureStateUnavailable(f"notional invalid for {symbol}")
    return result


@dataclass
class _Reservation:
    token: str
    order_object_id: int | None
    order: dict
    identities: set[str]
    acknowledged: bool = False
    cancel_pending: bool = False


@dataclass
class _Ledger:
    lock: threading.RLock = field(default_factory=threading.RLock)
    reservations: dict[str, _Reservation] = field(default_factory=dict)
    by_order_object: dict[int, str] = field(default_factory=dict)
    journal_file: Path | None = None
    journal_error: bool = False

    def initialize(self, bot):
        try:
            self.journal_file = _journal_path(bot)
            if not self.journal_file.exists():
                return
            document = json.loads(self.journal_file.read_text())
            if not isinstance(document, dict) or document.get("schema_version") != 1:
                raise ValueError("unsupported reservation journal schema")
            rows = document.get("reservations")
            if not isinstance(rows, list):
                raise ValueError("reservation journal rows malformed")
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("token"), str)
                    or not isinstance(row.get("order"), dict)
                    or not isinstance(row.get("identities", []), list)
                ):
                    raise ValueError("reservation journal row malformed")
                reservation = _Reservation(
                    token=row["token"],
                    order_object_id=None,
                    order=dict(row["order"]),
                    identities=set(str(value) for value in row.get("identities", []) if value),
                    acknowledged=bool(row.get("acknowledged", False)),
                    cancel_pending=bool(row.get("cancel_pending", False)),
                )
                self.reservations[reservation.token] = reservation
        except Exception:
            # An unreadable journal is a hard entry blocker, never an empty ledger.
            self.journal_error = True

    def _persist(self):
        if self.journal_file is None or self.journal_error:
            raise ExposureStateUnavailable("reservation journal unavailable")
        if not self.reservations:
            try:
                self.journal_file.unlink(missing_ok=True)
                return
            except OSError as exc:
                self.journal_error = True
                raise ExposureStateUnavailable("reservation journal cleanup failed") from exc
        document = {
            "schema_version": 1,
            "reservations": [
                {
                    "token": reservation.token,
                    "order": _journal_order(reservation.order),
                    "identities": sorted(reservation.identities),
                    "acknowledged": reservation.acknowledged,
                    "cancel_pending": reservation.cancel_pending,
                }
                for reservation in self.reservations.values()
            ],
        }
        parent = self.journal_file.parent
        temporary = None
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, name = tempfile.mkstemp(prefix=".exposure-reservations-", dir=parent)
            temporary = Path(name)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(document, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.journal_file)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as exc:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            self.journal_error = True
            raise ExposureStateUnavailable("reservation journal write failed") from exc

    def _snapshot_entries(self, bot):
        positions = getattr(bot, "positions", None)
        open_orders = getattr(bot, "open_orders", None)
        if not isinstance(positions, dict) or not isinstance(open_orders, dict):
            raise ExposureStateUnavailable("positions/open_orders snapshot unavailable")
        balance_getter = getattr(bot, "get_raw_balance", None)
        if callable(balance_getter):
            balance = _finite_positive(balance_getter(), "wallet balance")
        else:
            balance = _finite_positive(getattr(bot, "balance", None), "wallet balance")

        executed = {"long": 0.0, "short": 0.0}
        position_symbols = {"long": set(), "short": set()}
        for symbol, sides in positions.items():
            if not isinstance(sides, dict):
                raise ExposureStateUnavailable("position snapshot malformed")
            for pside in ("long", "short"):
                pos = sides.get(pside)
                if not pos:
                    continue
                if not isinstance(pos, dict):
                    raise ExposureStateUnavailable("position snapshot malformed")
                try:
                    qty = float(pos.get("size", 0.0) or 0.0)
                except (TypeError, ValueError) as exc:
                    raise ExposureStateUnavailable("position quantity invalid") from exc
                if not math.isfinite(qty):
                    raise ExposureStateUnavailable("position quantity invalid")
                if qty == 0.0:
                    continue
                price = pos.get("price") or pos.get("entry_price")
                executed[pside] += _notional(bot, str(symbol), qty, price)
                position_symbols[pside].add(str(symbol))

        resting = {"long": 0.0, "short": 0.0}
        resting_fees = {"long": 0.0, "short": 0.0}
        pending_symbols = {"long": set(), "short": set()}
        open_identity_map: dict[str, dict] = {}
        for symbol, orders in open_orders.items():
            if not isinstance(orders, list):
                raise ExposureStateUnavailable("open order snapshot malformed")
            for order in orders:
                if not isinstance(order, dict):
                    raise ExposureStateUnavailable("open order snapshot malformed")
                pside = _position_side(order)
                if not _is_entry(order, pside):
                    # Unclassified non-reduce orders are not safe to ignore.
                    if not bool(order.get("reduce_only") or order.get("reduceOnly")) and pside is None:
                        raise ExposureStateUnavailable("open order side semantics unavailable")
                    continue
                remaining = extract_order_remaining_qty(order)
                price = order.get("price")
                if remaining is None or remaining <= 0.0:
                    if remaining == 0.0:
                        continue
                    raise ExposureStateUnavailable("open order remaining quantity unavailable")
                amount = _notional(bot, str(order.get("symbol") or symbol), remaining, price)
                resting[pside] += amount
                resting_fees[pside] += amount * _fee_rate(bot, str(order.get("symbol") or symbol))
                pending_symbols[pside].add(str(order.get("symbol") or symbol))
                for identity in _order_identities(order):
                    open_identity_map[identity] = order

        return balance, executed, position_symbols, resting, resting_fees, pending_symbols, open_identity_map

    def _caps(self, bot):
        caps = {}
        slots = {}
        for pside in ("long", "short"):
            twel = _get_value(bot, pside, ("total_wallet_exposure_limit",), None)
            threshold = _get_value(
                bot, pside,
                ("total_exposure_enforcer_threshold", "risk_twel_enforcer_threshold"),
                None,
            )
            gate = _get_value(
                bot, pside,
                ("total_exposure_entry_gate_enabled", "risk_twel_entry_gate_enabled"),
                None,
            )
            n_positions = _get_value(bot, pside, ("n_positions",), None)
            try:
                twel = float(twel)
                threshold = float(threshold) if threshold is not None else 1.0
                n_positions = int(float(n_positions))
            except (TypeError, ValueError) as exc:
                raise ExposureStateUnavailable(f"risk parameters unavailable for {pside}") from exc
            if not math.isfinite(twel) or not math.isfinite(threshold) or n_positions < 0:
                raise ExposureStateUnavailable(f"risk parameters invalid for {pside}")
            if twel <= 0.0:
                caps[pside] = 0.0
            else:
                caps[pside] = min(twel, twel * threshold) if bool(gate) else twel
            slots[pside] = n_positions
        return caps, slots

    def _active_reservations(self, open_identity_map):
        # Reservations already represented by authoritative resting orders are
        # not counted twice. They remain in the ledger until a coherent refresh
        # reconciles them, protecting against a local snapshot being replaced.
        out = []
        for reservation in self.reservations.values():
            if reservation.cancel_pending and any(
                identity in open_identity_map for identity in reservation.identities
            ):
                continue
            if any(identity in open_identity_map for identity in reservation.identities):
                continue
            out.append(reservation)
        return out

    def admit(self, bot, orders: list[dict]):
        with self.lock:
            try:
                if self.journal_error:
                    raise ExposureStateUnavailable("reservation journal unavailable")
                balance, executed, position_symbols, resting, resting_fees, pending_symbols, open_ids = self._snapshot_entries(bot)
                caps, slots = self._caps(bot)
                active = self._active_reservations(open_ids)
                projected = dict(executed)
                fee_reserve = {"long": 0.0, "short": 0.0}
                occupied = {side: set(position_symbols[side]) | set(pending_symbols[side]) for side in ("long", "short")}
                for side in ("long", "short"):
                    projected[side] += resting[side]
                    fee_reserve[side] += resting_fees[side]
                for reservation in active:
                    order = reservation.order
                    side = _position_side(order)
                    symbol = str(order.get("symbol") or "")
                    qty = order.get("qty", order.get("amount"))
                    price = order.get("price")
                    if side not in projected or not symbol:
                        raise ExposureStateUnavailable("in-flight order semantics unavailable")
                    amount = _notional(bot, symbol, qty, price)
                    projected[side] += amount
                    fee_reserve[side] += amount * _fee_rate(bot, symbol)
                    occupied[side].add(symbol)
            except ExposureStateUnavailable:
                safe_exits = [
                    order for order in orders
                    if isinstance(order, dict)
                    and bool(order.get("reduce_only") or order.get("reduceOnly"))
                ]
                safe_ids = {id(order) for order in safe_exits}
                return safe_exits, [order for order in orders if id(order) not in safe_ids], "exposure_state_unavailable"

            accepted, blocked = [], []
            new_tokens: list[str] = []
            new_order_ids: set[int] = set()
            for order in orders:
                if not isinstance(order, dict):
                    blocked.append(order)
                    continue
                pside = _position_side(order)
                if bool(order.get("reduce_only") or order.get("reduceOnly")):
                    accepted.append(order)
                    continue
                side = str(order.get("side") or "").lower()
                if pside in {"long", "short"} and side in {"buy", "sell"}:
                    if (pside == "long" and side == "sell") or (pside == "short" and side == "buy"):
                        # Closing orders do not reserve entry exposure.
                        accepted.append(order)
                        continue
                try:
                    if pside not in projected:
                        raise ExposureStateUnavailable("entry position side unavailable")
                    if side not in {"buy", "sell"} or not _is_entry(order, pside):
                        raise ExposureStateUnavailable("entry/exit semantics unavailable")
                    symbol = str(order.get("symbol") or "")
                    if not symbol:
                        raise ExposureStateUnavailable("entry symbol unavailable")
                    if bool(getattr(bot, "_is_market_execution_order", lambda _o: False)(order)):
                        raise ExposureStateUnavailable("market entry has no proven price bound")
                    amount = _notional(bot, symbol, order.get("qty"), order.get("price"))
                    fee = amount * _fee_rate(bot, symbol)
                    slot = occupied[pside]
                    if symbol not in slot and len(slot) >= slots[pside]:
                        blocked.append(order)
                        continue
                    cap = caps[pside]
                    denominator = balance - sum(fee_reserve.values()) - fee
                    if denominator <= 0.0:
                        blocked.append(order)
                        continue
                    # TWEL is side-specific in the strategy model. Other-side
                    # exposure is not compared to this side's independent cap.
                    side_ratio = (projected[pside] + amount) / denominator
                    if cap <= 0.0 or side_ratio > cap + 1e-12:
                        blocked.append(order)
                        continue
                    accepted.append(order)
                    projected[pside] += amount
                    fee_reserve[pside] += fee
                    slot.add(symbol)
                    token = uuid.uuid4().hex
                    reservation = _Reservation(
                        token=token,
                        order_object_id=id(order),
                        order=dict(order),
                        identities=_order_identities(order),
                    )
                    self.reservations[token] = reservation
                    self.by_order_object[id(order)] = token
                    new_tokens.append(token)
                    new_order_ids.add(id(order))
                except ExposureStateUnavailable:
                    blocked.append(order)
            if new_tokens:
                try:
                    self._persist()
                except ExposureStateUnavailable:
                    for token in new_tokens:
                        reservation = self.reservations.pop(token, None)
                        if reservation and reservation.order_object_id is not None:
                            self.by_order_object.pop(reservation.order_object_id, None)
                    failed_entries = [order for order in accepted if id(order) in new_order_ids]
                    accepted = [order for order in accepted if id(order) not in new_order_ids]
                    blocked.extend(failed_entries)
                    return accepted, blocked, "reservation_journal_unavailable"
            return accepted, blocked, None

    def record_ack(self, order: dict, result: dict):
        with self.lock:
            token = self.by_order_object.get(id(order))
            reservation = self.reservations.get(token) if token else None
            if reservation is not None:
                reservation.acknowledged = True
                reservation.identities.update(_order_identities(result))
                try:
                    self._persist()
                except ExposureStateUnavailable:
                    self.journal_error = True

    def record_not_sent(self, order: dict):
        with self.lock:
            token = self.by_order_object.pop(id(order), None)
            if token:
                self.reservations.pop(token, None)
                try:
                    self._persist()
                except ExposureStateUnavailable:
                    self.journal_error = True

    def record_cancel(self, bot, orders: list[dict]):
        with self.lock:
            open_orders = getattr(bot, "open_orders", {}) or {}
            by_ids = {}
            for symbol, items in open_orders.items():
                for item in items or []:
                    for identity in _order_identities(item):
                        by_ids[identity] = item
            for order in orders:
                matches = [by_ids[key] for key in _order_identities(order) if key in by_ids]
                source = matches[0] if matches else order
                if not _is_entry(source):
                    continue
                token = uuid.uuid4().hex
                self.reservations[token] = _Reservation(
                    token=token,
                    order_object_id=None,
                    order=dict(source),
                    identities=_order_identities(source) | _order_identities(order),
                    acknowledged=True,
                    cancel_pending=True,
                )
            try:
                self._persist()
            except ExposureStateUnavailable:
                self.journal_error = True

    def reconcile_authoritative_snapshot(self, bot):
        """Drop reservations only after positions/open-orders cohort is committed."""
        with self.lock:
            open_orders = getattr(bot, "open_orders", {}) or {}
            open_ids = set()
            for items in open_orders.values():
                for order in items or []:
                    open_ids.update(_order_identities(order))
            for token, reservation in list(self.reservations.items()):
                matches = bool(reservation.identities & open_ids)
                if reservation.cancel_pending:
                    # Present means the authoritative resting-order snapshot
                    # itself continues to reserve; absent means it is terminal.
                    self.reservations.pop(token, None)
                    continue
                if matches or reservation.acknowledged:
                    self.reservations.pop(token, None)
                    if reservation.order_object_id is not None:
                        self.by_order_object.pop(reservation.order_object_id, None)
            try:
                self._persist()
            except ExposureStateUnavailable:
                self.journal_error = True


_LEDGER_INIT_LOCK = threading.Lock()


def _ledger(bot) -> _Ledger:
    ledger = getattr(bot, "_exposure_reservation_ledger", None)
    if ledger is None:
        # Two executor threads may reach first admission together. Installing
        # separate ledgers would let both spend the same portfolio headroom.
        with _LEDGER_INIT_LOCK:
            ledger = getattr(bot, "_exposure_reservation_ledger", None)
            if ledger is None:
                ledger = _Ledger()
                ledger.initialize(bot)
                setattr(bot, "_exposure_reservation_ledger", ledger)
    return ledger


def admit_entry_batch(bot, orders: list[dict]):
    return _ledger(bot).admit(bot, list(orders))


def record_create_ack(bot, order: dict, result: dict):
    _ledger(bot).record_ack(order, result)


def record_create_not_sent(bot, order: dict):
    _ledger(bot).record_not_sent(order)


def record_cancel_intent(bot, orders: list[dict]):
    _ledger(bot).record_cancel(bot, orders)


def reconcile_authoritative_snapshot(bot):
    ledger = getattr(bot, "_exposure_reservation_ledger", None)
    if ledger is not None:
        ledger.reconcile_authoritative_snapshot(bot)
