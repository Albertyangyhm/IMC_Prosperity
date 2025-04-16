from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np
import math
from math import log
from statistics import NormalDist


class BlackScholes:
    """Utility class for Black–Scholes analytics."""

    @staticmethod
    def black_scholes_call(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        """Return Black‑Scholes price of a European call option."""
        if spot <= 0:
            return 0.0
        if volatility <= 0 or time_to_expiry <= 0:
            return max(0.0, spot - strike)
        d1 = (
            (log(spot / strike) + 0.5 * volatility * volatility * time_to_expiry)
            / (volatility * math.sqrt(time_to_expiry))
        )
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        return spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)

    @staticmethod
    def implied_volatility(call_price: float, spot: float, strike: float, time_to_expiry: float,
                           max_iterations: int = 200, tolerance: float = 1e-10) -> float:
        """Implied volatility via bisection solving C_BS(σ)=call_price."""
        if call_price < 0 or spot <= 0:
            return 0.01
        if time_to_expiry <= 0:
            return 0.01
        low_vol, high_vol = 0.001, 2.0
        volatility = 0.5 * (low_vol + high_vol)
        for _ in range(max_iterations):
            est_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = est_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = 0.5 * (low_vol + high_vol)
        return volatility

    @staticmethod
    def delta_call(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        """Return Black‑Scholes delta of a European call option."""
        if spot <= 0 or volatility <= 0 or time_to_expiry <= 0:
            return 0.0
        d1 = (
            (log(spot / strike) + 0.5 * volatility * volatility * time_to_expiry)
            / (volatility * math.sqrt(time_to_expiry))
        )
        return NormalDist().cdf(d1)


class Trader:
    """Market‑maker for VOLCANIC_ROCK and its option vouchers with delta hedging."""

    LIMIT_UNDERLYING = 400
    LIMIT_CALLS = 200

    def __init__(self):
        # Underlying mid‑prices for moving average of underlying (mavo)
        self.timestamp = 0
        self.underlying_mavo: List[float] = []
        self.last_underlying_mid: float | None = None

        # Per‑product IV and diff histories
        self.iv_history: Dict[str, List[float]] = {}
        self.iv_diff_history: Dict[str, List[float]] = {}
        self.last_iv: Dict[str, float] = {}

        # Portfolio net delta (options + underlying)
        self.net_delta: float = 0.0

        # Parameters
        self.default_underlying_window = 5
        self.default_call_mid_window = 30
        self.diff_window = 20
        self.min_volatility = 0.001

        self.option_half_spreads = {
            "VOLCANIC_ROCK_VOUCHER_9750": 0.003,
            "VOLCANIC_ROCK_VOUCHER_10000": 5.310120e-03,
            "VOLCANIC_ROCK_VOUCHER_10250": 7.577622e-04,
            "VOLCANIC_ROCK_VOUCHER_10500": 1.519836e-03,
            "VOLCANIC_ROCK_VOUCHER_9500": 8.380629e-03,
        }

    # ---------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0  # not used here

        # -----------------------------------------------------------------
        # 1) Update underlying mid price & store for moving average
        # -----------------------------------------------------------------
        if "VOLCANIC_ROCK" in state.order_depths:
            od_rock = state.order_depths["VOLCANIC_ROCK"]
            if od_rock.buy_orders and od_rock.sell_orders:
                best_bid = max(od_rock.buy_orders)
                best_ask = min(od_rock.sell_orders)
                self.last_underlying_mid = 0.5 * (best_bid + best_ask)
                self.underlying_mavo.append(self.last_underlying_mid)

        # # -----------------------------------------------------------------
        # # 2) Recompute current portfolio delta from positions (robust against fills)
        # # -----------------------------------------------------------------
        # self.net_delta = 0.0
        # spot_for_delta = self.last_underlying_mid or 0.0
        ttm_const = 5.0 / 252.0  # 5 trading days to expiry
        # for product, pos in state.position.items():
        #     if product == "VOLCANIC_ROCK":
        #         self.net_delta += pos  # delta 1 per share
        #     elif product.startswith("VOLCANIC_ROCK_VOUCHER_"):
        #         strike = float(product.split("_")[-1])
        #         iv_guess = self.last_iv.get(product, 0.2)
        #         delta_opt = BlackScholes.delta_call(spot_for_delta, strike, ttm_const, iv_guess)
        #         self.net_delta += delta_opt * pos

        # -----------------------------------------------------------------
        # 3) Option market‑making & delta tracking
        # -----------------------------------------------------------------
        for product, od in state.order_depths.items():
            if not product.startswith("VOLCANIC_ROCK_VOUCHER_"):
                continue

            # Skip if we don't have an underlying mid yet
            if self.last_underlying_mid is None:
                continue

            all_asks = sorted(od.sell_orders.items(), key=lambda x: x[0])[:3]
            all_bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])[:3]
            if not all_asks or not all_bids:
                continue

            best_ask = all_asks[0][0]
            best_bid = all_bids[0][0]
            option_mid = 0.5 * (best_ask + best_bid)
            spot = self.last_underlying_mid

            # Strike & time to maturity
            strike = float(product.split("_")[-1])
            ttm = ttm_const
            ttm -= self.timestamp * 1/10000 * 1/252

            # ---------------------------
            # 3A) Implied volatility
            # ---------------------------
            new_iv = BlackScholes.implied_volatility(option_mid, spot, strike, ttm)
            if product in self.last_iv and abs(new_iv - self.last_iv[product]) > 0.1:
                new_iv = self.last_iv[product]
            self.last_iv[product] = new_iv

            # ---------------------------
            # 3B) IV history & vol‑of‑vol
            # ---------------------------
            self.iv_history.setdefault(product, []).append(new_iv)
            vol_window = min(len(self.iv_history[product]), self.default_call_mid_window)
            vol_of_vol = float(np.std(self.iv_history[product][-vol_window:])) if vol_window >= 2 else 0.0

            # Moving average window selection
            ma_window = self.choose_ma_window(product, vol_of_vol)
            iv_ma = np.mean(self.iv_history[product][-ma_window:]) if len(self.iv_history[product]) >= ma_window else np.mean(self.iv_history[product])

            # ---------------------------
            # 3C) Threshold & quoting vols
            # ---------------------------
            abs_diff = abs(new_iv - iv_ma)
            self.iv_diff_history.setdefault(product, []).append(abs_diff)
            diff_list = self.iv_diff_history[product][-self.diff_window:]
            threshold = np.percentile(diff_list, 80) if len(diff_list) >= 5 else self.option_half_spreads[product]
            vol_minus = max(iv_ma - threshold, self.min_volatility)
            vol_plus = max(iv_ma + threshold, self.min_volatility)

            buy_quote = BlackScholes.black_scholes_call(spot, strike, ttm, vol_minus)
            sell_quote = BlackScholes.black_scholes_call(spot, strike, ttm, vol_plus)
            spot_delta = BlackScholes.delta_call(spot, strike, ttm, iv_ma)
            
            print(f"Product: {product} | Spot_Delta: {spot_delta:.2f} | Spot: {spot:.2f} | Strike: {strike:.2f} | IV_MA: {iv_ma:.4f}")
            
            pos = state.position.get(product, 0)
            max_pos = self.LIMIT_CALLS
            buyable_qty = max_pos - pos
            sellable_qty = pos + max_pos

            product_orders: List[Order] = []

            # ---------------------------
            # 3D) Lift asks within quote
            # ---------------------------
            for ask_price, ask_vol in all_asks:
                if ask_price <= buy_quote and buyable_qty > 0:
                    qty = min(buyable_qty, -ask_vol)
                    if qty > 0:
                        product_orders.append(Order(product, ask_price, qty))
                        iv_delta = BlackScholes.implied_volatility(ask_price, spot, strike, ttm)
                        delta_per_contract = BlackScholes.delta_call(spot, strike, ttm, iv_delta)
                        buyable_qty -= qty
                        self.net_delta += delta_per_contract * qty
                        print(f"BUY {product} @ {ask_price} qty {qty} delta/contract {delta_per_contract:.2f} delta {delta_per_contract * qty:.2f}")
                else:
                    break
            # if buyable_qty > 0:
            #     px = int(round(buy_quote))
            #     product_orders.append(Order(product, px, buyable_qty))
            #     iv_delta = BlackScholes.implied_volatility(px, spot, strike, ttm)
            #     delta_per_contract = BlackScholes.delta_call(spot, strike, ttm, iv_delta)
            #     self.net_delta += delta_per_contract * buyable_qty
            #     print(f"BUY {product} @ {ask_price} buyable_qty {buyable_qty} delta/contract {delta_per_contract:.2f} delta {delta_per_contract * buyable_qty:.2f}")

            # ---------------------------
            # 3E) Hit bids within quote
            # ---------------------------
            for bid_price, bid_vol in all_bids:
                if bid_price >= sell_quote and sellable_qty > 0:
                    qty = min(sellable_qty, bid_vol)
                    if qty > 0:
                        product_orders.append(Order(product, bid_price, -qty))
                        iv_delta = BlackScholes.implied_volatility(bid_price, spot, strike, ttm)
                        delta_per_contract = BlackScholes.delta_call(spot, strike, ttm, iv_delta)
                        sellable_qty -= qty
                        self.net_delta -= delta_per_contract * qty
                        print(f"SELL {product} @ {bid_price} qty {qty} delta/contract {-delta_per_contract:.2f} delta {-delta_per_contract * qty:.2f}")
                else:
                    break
            # if sellable_qty > 0:
            #     px = int(round(sell_quote))
            #     product_orders.append(Order(product, px, -sellable_qty))
            #     iv_delta = BlackScholes.implied_volatility(px, spot, strike, ttm)
            #     delta_per_contract = BlackScholes.delta_call(spot, strike, ttm, iv_delta)
            #     self.net_delta -= delta_per_contract * sellable_qty
            #     print(f"SELL {product} @ {bid_price} sellable_qty {sellable_qty} delta/contract {-delta_per_contract:.2f} delta {-delta_per_contract * sellable_qty:.2f}")

            orders[product] = product_orders

        # -----------------------------------------------------------------
        # 4) Underlying hedge to flatten delta
        # -----------------------------------------------------------------
        print(f"Net Delta Before Hedging: {self.net_delta:.2f}")
        
        hedge_orders: List[Order] = []
        if self.last_underlying_mid is not None and "VOLCANIC_ROCK" in state.order_depths:
            od_rock = state.order_depths["VOLCANIC_ROCK"]
            best_bid = max(od_rock.buy_orders) if od_rock.buy_orders else None
            best_ask = min(od_rock.sell_orders) if od_rock.sell_orders else None

            # Desired hedge quantity (integer)
            hedge_qty = -round(self.net_delta)
            
            current_pos = state.position.get("VOLCANIC_ROCK", 0)
            capacity_buy = self.LIMIT_UNDERLYING - current_pos
            capacity_sell = current_pos + self.LIMIT_UNDERLYING

            if hedge_qty > 0:  # need to buy underlying
                qty = min(hedge_qty, capacity_buy)
                if qty > 0:
                    px = best_ask if best_ask is not None else int(round(self.last_underlying_mid))
                    hedge_orders.append(Order("VOLCANIC_ROCK", px, qty))
                    self.net_delta += qty  # delta of underlying = qty
                    print(f"HEDGE: BUY underlying @ {px} qty {qty} delta {qty:.2f}")
            elif hedge_qty < 0:  # need to sell underlying
                qty = min(-hedge_qty, capacity_sell)
                if qty > 0:
                    px = best_bid if best_bid is not None else int(round(self.last_underlying_mid))
                    hedge_orders.append(Order("VOLCANIC_ROCK", px, -qty))
                    self.net_delta -= qty
                    print(f"HEDGE: SELL underlying @ {px} qty {qty} delta {-qty:.2f}")

        if hedge_orders:
            orders.setdefault("VOLCANIC_ROCK", []).extend(hedge_orders)

        print(f"Net Delta After Hedging : {self.net_delta:.2f}")
        self.timestamp += 1
        trader_data = "VOLCANO_TRADER_STATE"
        return orders, conversions, trader_data

    # ---------------------------------------------------------------------
    # Helper for dynamic MA window selection
    # ---------------------------------------------------------------------
    def choose_ma_window(self, product: str, vol_of_vol: float) -> int:
        if product == "VOLCANIC_ROCK_VOUCHER_9500":
            if vol_of_vol > 0.008:
                return 4
            if vol_of_vol > 0.006:
                return 8
            return 12
        
        if product == "VOLCANIC_ROCK_VOUCHER_9750":
            return 5 if vol_of_vol > 0.015 else 8
        if product == "VOLCANIC_ROCK_VOUCHER_10000":
            if vol_of_vol > 0.015:
                return 4
            if vol_of_vol > 0.013:
                return 8
            if vol_of_vol > 0.010:
                return 12
            return 15
        if product == "VOLCANIC_ROCK_VOUCHER_10250":
            if vol_of_vol > 0.001:
                return 5
            if vol_of_vol > 0.0008:
                return 10
            return 30
        if product == "VOLCANIC_ROCK_VOUCHER_10500":
            if vol_of_vol > 0.002:
                return 5
            if vol_of_vol > 0.0015:
                return 8
            if vol_of_vol > 0.0008:
                return 15
            return 30
        return 10
