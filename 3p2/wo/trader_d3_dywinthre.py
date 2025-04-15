from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import numpy as np
import pandas as pd
import math
from math import log, sqrt, exp
from statistics import NormalDist

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if spot <= 0:
            return 0.0
        # avoid domain errors
        if volatility <= 0 or time_to_expiry <= 0:
            return max(0.0, spot - strike)
        d1 = (
            (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry)
            / (volatility * math.sqrt(time_to_expiry))
        )
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry,
                           max_iterations=200, tolerance=1e-10):
        """
        A simple bisection approach to solve for volatility:
          black_scholes_call(spot, strike, vol) = call_price.
        We do minimal error checking here; user should pass valid inputs.
        """
        if call_price < 0:
            return 0.01  # fallback
        if spot <= 0:
            return 0.01
        if time_to_expiry <= 0:
            # If no time left, the call is basically max(spot - strike, 0)
            intrinsic = max(spot - strike, 0)
            # If the observed price < intrinsic, there's no good solution
            return 0.01
        
        low_vol = 0.001
        high_vol = 2.0   # Let’s allow up to 200% just in case
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            est_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = est_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self):
        # Store the underlying mid prices
        self.underlying_mavo = []
        self.last_underlying_mid = None

        # For each option, store all IV readings and also store the difference (abs(IV - MA_IV))
        self.iv_history = {}
        self.iv_diff_history = {}

        # We keep the last IV used for rejection of large jumps
        self.last_iv = {}

        self.stateHistory = []
        self.timestamp = 0

        # Position limits
        self.LIMIT_UNDERLYING = 400
        self.LIMIT_CALLS = 200

        # Basic parameters
        self.default_underlying_window = 5
        self.default_call_mid_window = 30  # # points to compute vol_of_vol
        self.diff_window = 20             # # points to compute 80th percentile threshold
        self.min_volatility = 0.001       # Floor on volatility if needed

        self.option_half_spreads = {
            'VOLCANIC_ROCK_VOUCHER_9750': 8.380629e-03,
            'VOLCANIC_ROCK_VOUCHER_10000': 5.310120e-03,
            'VOLCANIC_ROCK_VOUCHER_10250': 7.577622e-04,
            'VOLCANIC_ROCK_VOUCHER_10500': 1.519836e-03
        }        

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # 1) Handle underlying: VOLCANIC_ROCK
        if 'VOLCANIC_ROCK' in state.order_depths:
            od_rock = state.order_depths['VOLCANIC_ROCK']
            if od_rock.buy_orders and od_rock.sell_orders:
                best_bid = max(od_rock.buy_orders.keys())
                best_ask = min(od_rock.sell_orders.keys())
                mid_price = 0.5*(best_ask + best_bid)
                self.underlying_mavo.append(mid_price)
                self.last_underlying_mid = mid_price

            # Example: you might do the same “market making” as before,
            # but for brevity we skip final order creation. You can re-add it.
            # ...
            # result["VOLCANIC_ROCK"] = [... orders ...]

        # 2) For each voucher, run the new dynamic logic
        for product in state.order_depths:
            if not product.startswith('VOLCANIC_ROCK_VOUCHER_'):
                continue

            od = state.order_depths[product]
            all_asks = sorted(od.sell_orders.items(), key=lambda x: x[0])[:3]
            all_bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])[:3]
            if not all_asks or not all_bids or self.last_underlying_mid is None:
                # Not enough data
                continue

            # Basic data
            best_ask = all_asks[0][0]
            best_bid = all_bids[0][0]
            option_mid = 0.5 * (best_ask + best_bid)
            spot = self.last_underlying_mid

            # Extract the strike K from the product name
            try:
                strike = float(product.split('_')[-1])
            except:
                continue

            # Time to expiry (example: 5/252 minus a small portion if wanted)
            ttm = 5.0/252.0

            # Compute new IV from the observed mid
            new_iv = BlackScholes.implied_volatility(option_mid, spot, strike, ttm)

            # Reject big jumps in new_iv if we have a previous reading
            if product in self.last_iv:
                if abs(new_iv - self.last_iv[product]) > 0.1:
                    # revert to old IV
                    new_iv = self.last_iv[product]
            self.last_iv[product] = new_iv

            # 2A) Update self.iv_history
            if product not in self.iv_history:
                self.iv_history[product] = []
            self.iv_history[product].append(new_iv)

            # 2B) Calculate vol_of_vol from the last ~30 points (or fewer if we have fewer)
            vol_window = min(len(self.iv_history[product]), self.default_call_mid_window)
            recent_ivs_for_vol = self.iv_history[product][-vol_window:]
            vol_of_vol = float(np.std(recent_ivs_for_vol)) if len(recent_ivs_for_vol) >= 2 else 0.0

            # 2C) Choose the dynamic MA window size based on vol_of_vol
            ma_window = self.choose_ma_window(product, vol_of_vol)

            # 2D) Compute the moving average IV using ma_window
            if len(self.iv_history[product]) < ma_window:
                iv_ma = np.mean(self.iv_history[product])  # if we have fewer than ma_window data
            else:
                iv_ma = np.mean(self.iv_history[product][-ma_window:])

            # 2E) Compute the absolute difference for threshold array
            abs_diff = abs(new_iv - iv_ma)
            if product not in self.iv_diff_history:
                self.iv_diff_history[product] = []
            self.iv_diff_history[product].append(abs_diff)

            # 2F) Compute the dynamic threshold from the last 20 diff samples, 80th percentile
            diff_list = self.iv_diff_history[product][-self.diff_window:]  # up to 20
            if len(diff_list) >= 5:
                # if we have at least 5 differences, let's compute the percentile
                threshold = np.percentile(diff_list, 80)
            else:
                # fallback if we have too few points
                threshold = self.option_half_spreads[product]

            # 2G) Form buy/sell quotes based on (iv_ma ± threshold)
            vol_minus = max(iv_ma - threshold, self.min_volatility)
            vol_plus  = max(iv_ma + threshold, self.min_volatility)

            buy_quote = BlackScholes.black_scholes_call(spot, strike, ttm, vol_minus)
            sell_quote = BlackScholes.black_scholes_call(spot, strike, ttm, vol_plus)

            # 2H) Market making logic, similar to SQUID_INK approach
            pos = state.position.get(product, 0)
            max_pos = self.LIMIT_CALLS
            buyable_qty = max_pos - pos
            sellable_qty = pos + max_pos

            orders = []
            # Take ask liquidity if ask <= buy_quote
            for ask_price, ask_vol in all_asks:
                if ask_price <= buy_quote and buyable_qty > 0:
                    can_buy = min(buyable_qty, -ask_vol)  # remember ask_vol < 0
                    if can_buy > 0:
                        orders.append(Order(product, ask_price, can_buy))
                        buyable_qty -= can_buy
                else:
                    break
            # Post buy order if capacity remains
            if buyable_qty > 0:
                px = int(round(buy_quote))
                orders.append(Order(product, px, buyable_qty))

            # Take bid liquidity if bid >= sell_quote
            for bid_price, bid_vol in all_bids:
                if bid_price >= sell_quote and sellable_qty > 0:
                    can_sell = min(sellable_qty, bid_vol)  # bid_vol>0
                    if can_sell > 0:
                        orders.append(Order(product, bid_price, -can_sell))
                        sellable_qty -= can_sell
                else:
                    break
            # Post sell order if capacity remains
            if sellable_qty > 0:
                px = int(round(sell_quote))
                orders.append(Order(product, px, -sellable_qty))

            # Store orders
            result[product] = orders

        # Save the state
        self.stateHistory.append(state)
        self.timestamp += 1

        traderData = "VOLCANO_TRADER_STATE"
        return result, conversions, traderData

    def choose_ma_window(self, product: str, vol_of_vol: float) -> int:
        """
        Returns a dynamic window size for MA_IV based on 'vol_of_vol' 
        and the piecewise rules provided by product.
        """

        # Example logic given in the request:
        # 1) VOLCANIC_ROCK_VOUCHER_9750:
        #    if vol_of_vol > 0.015 => window=5
        #    else => window=8

        # 2) VOLCANIC_ROCK_VOUCHER_10000:
        #    if vol_of_vol > 0.015 => window=4
        #    elif vol_of_vol > 0.013 => window=8
        #    elif vol_of_vol > 0.010 => window=12
        #    else => window=15

        # 3) VOLCANIC_ROCK_VOUCHER_10250:
        #    if vol_of_vol > 0.001 => window=5
        #    elif vol_of_vol > 0.0008 => window=10
        #    else => window=30

        # 4) VOLCANIC_ROCK_VOUCHER_10500:
        #    if vol_of_vol > 0.002 => window=5
        #    elif vol_of_vol > 0.0015 => window=8
        #    elif vol_of_vol > 0.0008 => window=15
        #    else => window=30

        if product == "VOLCANIC_ROCK_VOUCHER_9750":
            if vol_of_vol > 0.015:
                return 5
            else:
                return 8

        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            if vol_of_vol > 0.015:
                return 4
            elif vol_of_vol > 0.013:
                return 8
            elif vol_of_vol > 0.010:
                return 12
            else:
                return 15

        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            if vol_of_vol > 0.001:
                return 5
            elif vol_of_vol > 0.0008:
                return 10
            else:
                return 30

        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            # We'll interpret the user logic in ascending order:
            if vol_of_vol > 0.002:
                return 5
            elif vol_of_vol > 0.0015:
                return 8
            elif vol_of_vol > 0.0008:
                return 15
            else:
                return 30

        # Default if not recognized
        return 10
