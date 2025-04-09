from typing import Dict, List
import math
import statistics
import pandas as pd
import numpy as np
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def __init__(self):
        # dictionary of lists
        self.stateHistory = []
        self.enteredLong = False
        self.enteredShort = False
        self.prevValue = 0
        self.time = 0
        self.r4long = False
        self.r4short = False

    def updateProductHist(self, productName, state, ifPop):

        self.stateHistory.append(state)
        startDay = 0 if not ifPop else len(self.stateHistory) - 6
        order_depth_hist = [
        i.order_depths[productName] for i in self.stateHistory[startDay:-1]
        ]
        sell_orders_hist = [i.sell_orders for i in order_depth_hist]
        buy_orders_hist = [i.buy_orders for i in order_depth_hist]

        best_ask_hist = [[min(orders.keys()), orders[min(orders.keys())]] \
                        for orders in sell_orders_hist]
        best_bid_hist = [[max(orders.keys()), orders[max(orders.keys())]] \
                        for orders in buy_orders_hist]

        best_ask_vol_hist = [i[1] for i in best_ask_hist]
        best_bid_vol_hist = [i[1] for i in best_bid_hist]
        best_ask_hist = [i[0] for i in best_bid_hist]
        best_bid_hist = [i[0] for i in best_bid_hist]

        return best_ask_hist, best_ask_vol_hist, best_bid_hist, best_bid_vol_hist
  
    def extract_orderbook_series(self, product: str, window: int = 100):
        # Get last `window` states (excluding current one)
        states = self.stateHistory[-window-1:-1]  # last 100 states

        bid_prices = []
        bid_volumes = []
        ask_prices = []
        ask_volumes = []

        for state in states:
            order_depth: OrderDepth = state.order_depths[product]

            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                bid_prices.append(best_bid)
                bid_volumes.append(order_depth.buy_orders[best_bid])
            else:
                bid_prices.append(None)
                bid_volumes.append(0)

            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                ask_prices.append(best_ask)
                ask_volumes.append(order_depth.sell_orders[best_ask])
            else:
                ask_prices.append(None)
                ask_volumes.append(0)

        return bid_prices, bid_volumes, ask_prices, ask_volumes


    def get_arma_forecast(self, product: str):
        """
        Computes a one-step ARMA(1,1) forecast for the log differences of the mid prices for a given product.
        Uses a simple OLS approach to estimate the AR(1) and then an approximate MA(1) coefficient.
        Returns a forecast (expected log difference for the next period).
        """
        window = 100  # Using the last 100 states; adjust as needed.
        if len(self.stateHistory) < window:
            return 1.0  # Not enough data

        # Extract mid prices for the last `window` states.
        mid_prices = []
        for s in self.stateHistory[-window:]:
            order_depth = s.order_depths[product]
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_prices.append((best_bid + best_ask) / 2)
        mid_prices = np.array(mid_prices)
        
        # Calculate log prices and log returns (differenced log prices)
        log_prices = np.log(mid_prices)
        x = np.diff(log_prices)  # x[t] = log(mid[t+1]) - log(mid[t])
        n = len(x)
        if n < 2:
            return 1.0

        # --- Estimate AR(1) coefficient (phi) ---
        # Regress x[1:] on x[:-1] (no intercept)
        X_ar = x[:-1]
        Y_ar = x[1:]
        denom_phi = np.dot(X_ar, X_ar)
        phi = np.dot(X_ar, Y_ar) / denom_phi if denom_phi != 0 else 0.0

        # --- Estimate MA(1) coefficient (theta) ---
        # Compute residuals for t = 1, ..., n-1: e[t] = x[t] - phi*x[t-1]
        e = Y_ar - phi * X_ar  # length n-1
        if len(e) < 2:
            theta = 0.0
        else:
            e_lag = e[:-1]
            e_current = e[1:]
            denom_theta = np.dot(e_lag, e_lag)
            theta = np.dot(e_lag, e_current) / denom_theta if denom_theta != 0 else 0.0

        # --- One-step Forecast ---
        # For ARMA(1,1): forecast = phi * (last log return) + theta * (last residual)
        x_last = x[-1]
        e_last = x_last - phi * x[-2] if n >= 2 else 0.0
        forecast = phi * x_last + theta * e_last

        return forecast

    def run(self, state):
        """
        Only method required. It takes all buy and sell orders for all symbols as input,
        and outputs a list of orders to be sent.
        """
        self.stateHistory.append(state)
        result = {}
        sq_orders = []
        product = 'SQUID_INK'
        order_depth = state.order_depths[product]
        bid_prices, bid_vols, ask_prices, ask_vols = self.extract_orderbook_series(product)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) // 2

        position = state.position.get(product, 0)
        limit = 50

        buyable = max(0, limit - position)
        sellable = max(0, limit + position)
        
        if len(self.stateHistory) < 101:
            # OBI
            mid_prices = [(bid + ask)/2 for bid, ask in zip(bid_prices, ask_prices)]
            obi_values = []
            for b_vol, a_vol in zip(bid_vols, ask_vols):
                total_vol = b_vol + abs(a_vol)
                obi = (b_vol - abs(a_vol)) / total_vol if total_vol != 0 else 0
                obi_values.append(obi)
            obi_rolling = sum(obi_values[-100:]) / 100

            # PVM
            pvm_values = []
            for i in range(1, len(mid_prices)):
                delta_p = mid_prices[i] - mid_prices[i-1]
                total_vol = bid_vols[i] + abs(ask_vols[i])
                pvm_values.append(delta_p * total_vol)
            pvm_rolling = sum(pvm_values[-50:]) / 50

            # enter long 
            if obi_rolling > 0.004 and pvm_rolling > 5 and buyable > 0:
                sq_orders.append(Order(product,best_ask-1,buyable))

            # enter short 
            elif obi_rolling < -0.004 and pvm_rolling < -5 and sellable > 0:
                sq_orders.append(Order(product,best_bid+1,-sellable))   

            if position > 0 and ( abs(obi_rolling) < 0.004 or abs(pvm_rolling) < 5):
                sq_orders.append(Order(product,best_ask-1,-position))

            elif position < 0 and ( abs(obi_rolling) < 0.004 or abs(pvm_rolling) < 5):
                sq_orders.append(Order(product,best_bid+1,position)) 

        # === Book Mid Adjustments with ARMA Signal Integration ===
        curr = order_depth
        bid_prices_now, bid_vols_now = [], []
        ask_prices_now, ask_vols_now = [], []

        # Get top 3 bid prices/volumes
        for i in range(1, 4):
            try:
                bid_p = sorted(curr.buy_orders.keys(), reverse=True)[i - 1]
                bid_v = curr.buy_orders[bid_p]
                bid_prices_now.append(bid_p)
                bid_vols_now.append(bid_v)
            except Exception:
                continue

        # Get top 3 ask prices/volumes
        for i in range(1, 4):
            try:
                ask_p = sorted(curr.sell_orders.keys())[i - 1]
                ask_v = curr.sell_orders[ask_p]
                ask_prices_now.append(ask_p)
                ask_vols_now.append(ask_v)
            except Exception:
                continue

        # Remove NaNs or zeros from the lists
        bid_filtered = [(p, v) for p, v in zip(bid_prices_now, bid_vols_now) if p is not None and v is not None and v != 0]
        ask_filtered = [(p, v) for p, v in zip(ask_prices_now, ask_vols_now) if p is not None and v is not None and v != 0]

        if bid_filtered:
            book_bid = sum(p * v for p, v in bid_filtered) / sum(v for _, v in bid_filtered)
        else:
            book_bid = best_bid

        if ask_filtered:
            book_ask = sum(p * v for p, v in ask_filtered) / sum(v for _, v in ask_filtered)
        else:
            book_ask = best_ask

        book_mid = (book_bid + book_ask) / 2
        current_mid = (best_bid + best_ask) / 2
        mid_diff = current_mid - book_mid  # difference between current mid and weighted mid
        
        # Define a threshold for taking directional action.
        threshold = 0.05

        # # --- ARMA Signal Integration ---
        # arma_forecast = self.get_arma_forecast(product)
        # # Convert the forecast (in log-return terms) into an approximate price adjustment.
        # # For small returns, log return ≈ percentage change. Multiply by current_mid to get a price difference.
        # price_adjustment = current_mid * arma_forecast

        # # Combine the Book Mid signal and the ARMA signal:
        # combined_signal = mid_diff + price_adjustment

        # # Use a scaling factor to convert the ARMA forecast magnitude into an additional order size
        # scale_factor = 1e3  # Tuning parameter: adjust based on backtest/market conditions.
        # additional_size = int(np.ceil(scale_factor * abs(arma_forecast))) if abs(arma_forecast) > 0 else 0

  

        # # --- Combined Quoting and Sizing Decision ---
        # if combined_signal > threshold:
        #     # Bullish combined signal:
        #     # Adjust the buy quote slightly lower (for a better fill) and the sell quote slightly higher.
        #     buy_price = int(book_mid - (price_adjustment * 0.5))
        #     sell_price = int(current_mid + (price_adjustment * 0.5))
        #     order_size = buyable + additional_size  # Increase long size if ARMA shows strong upward bias.
        #     sq_orders.append(Order(product, buy_price, order_size))
        #     sq_orders.append(Order(product, sell_price, -sellable))
        # elif combined_signal < -threshold:
        #     # Bearish combined signal:
        #     buy_price = int(book_mid - (price_adjustment * 0.5))
        #     sell_price = int(current_mid + (price_adjustment * 0.5))
        #     order_size = sellable + additional_size  # Increase short size if ARMA shows strong downward bias.
        #     sq_orders.append(Order(product, buy_price, -sellable))
        #     sq_orders.append(Order(product, sell_price, order_size))
        # else:
        #     # When the signals are weak or conflicting, unwind positions if necessary.
        #     if position > 0:
        #         sq_orders.append(Order(product, int(current_mid), -position))
        #     elif position < 0:
        #         sq_orders.append(Order(product, int(current_mid), position))
                
        # if combined_signal > threshold:
        #     sq_orders.append(Order(product, int(book_mid), buyable))
        #     sq_orders.append(Order(product, int(current_mid), -sellable))
        # elif combined_signal < -threshold:
        #     sq_orders.append(Order(product, int(book_mid), -sellable))
        #     sq_orders.append(Order(product, int(current_mid), buyable))
        # else:
        #     if position > 0:
        #         sq_orders.append(Order(product, int(current_mid), -position))
        #     elif position < 0:
        #         sq_orders.append(Order(product, int(current_mid), position))
        if mid_diff > threshold:
            book_mid_signal = 1
        elif mid_diff < -threshold:
            book_mid_signal = -1
        else:
            book_mid_signal = 0

        # === Compute ARMA Signal (signal 2) ===
        arma_forecast = self.get_arma_forecast(product)
        forecast_threshold = 0.0005  # threshold for ARMA forecast
        # A positive ARMA forecast indicates upward bias; negative indicates downward bias.
        if arma_forecast > forecast_threshold:
            arma_signal = 1
        elif arma_forecast < -forecast_threshold:
            arma_signal = -1
        else:
            arma_signal = 0

        # === Decision Making using OR Logic for Signals ===
        # If either signal is bullish, go long; if either is bearish, go short.
        # If signals conflict or are neutral, try unwinding positions.
        bullish_signal = (arma_signal == 1 or book_mid_signal == 1)
        bearish_signal = (arma_signal == -1 or book_mid_signal == -1)

        if bullish_signal and buyable > 0:
            sq_orders.append(Order(product, int(book_mid), buyable))
            sq_orders.append(Order(product, int(current_mid), -sellable))
        elif bearish_signal and sellable > 0:
            sq_orders.append(Order(product, int(book_mid), -sellable))
            sq_orders.append(Order(product, int(current_mid), buyable))
        else:
            # If no signal is strong enough, unwind any positions.
            if position > 0:
                sq_orders.append(Order(product, int(current_mid), -position))
            elif position < 0:
                sq_orders.append(Order(product, int(current_mid), position))

        # === Finalize output ===
        result[product] = sq_orders
        self.time += 1
        traderData = 'SquidInkTrader_SGC_v1'
        conversions = 0
        return result, conversions, traderData
