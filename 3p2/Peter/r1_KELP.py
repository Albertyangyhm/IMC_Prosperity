from typing import Dict, List
import numpy as np
from datamodel import OrderDepth, TradingState, Order

class Trader:

    def __init__(self):
        # Optional: maintain a history of TradingState objects for diagnostics.
        self.stateHistory: List[TradingState] = []
        # Mean-reversion state: maintain rolling mid-price history for KELP.
        self.mid_price_history: List[float] = []
        # Various flags (can be used to refine strategy later).
        self.enteredLong = False
        self.enteredShort = False
        self.prevValue = 0
        self.time = 0
        self.r4long = False
        self.r4short = False

    def updateProductHist(self, productName: str, state: TradingState, ifPop: bool):
        """
        Optional: Updates the state history and extracts historical best bid/ask and volumes.
        """
        self.stateHistory.append(state)
        startDay = 0 if not ifPop else len(self.stateHistory) - 6
        order_depth_hist = [s.order_depths[productName] for s in self.stateHistory[startDay:-1]]
        sell_orders_hist = [od.sell_orders for od in order_depth_hist]
        buy_orders_hist = [od.buy_orders for od in order_depth_hist]

        best_ask_hist = [[min(orders.keys()), orders[min(orders.keys())]] for orders in sell_orders_hist]
        best_bid_hist = [[max(orders.keys()), orders[max(orders.keys())]] for orders in buy_orders_hist]

        best_ask_vol_hist = [entry[1] for entry in best_ask_hist]
        best_bid_vol_hist = [entry[1] for entry in best_bid_hist]
        best_ask_hist = [entry[0] for entry in best_ask_hist]
        best_bid_hist = [entry[0] for entry in best_bid_hist]

        return best_ask_hist, best_ask_vol_hist, best_bid_hist, best_bid_vol_hist

    def run(self, state: TradingState):

        result: Dict[str, List[Order]] = {}
        kelp_orders: List[Order] = []
        product = "KELP"
        position = state.position.get(product, 0)
        limit = 50
        base_size = 10

        order_depth: OrderDepth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            print(f"Incomplete order depth for {product}")
            return {}, 0, state.traderData
        
        # Get best bid and best ask with their volumes.
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = order_depth.sell_orders[best_ask]

        # === Improved Quoting Logic ===
        # Compute the mid-price and update rolling history.
        mid_price = (best_bid + best_ask) / 2.0
        self.mid_price_history.append(mid_price)
        window = 40
        history = self.mid_price_history[-window:]
        rolling_mean = np.mean(history)
        rolling_std = np.std(history, ddof=1) if len(history) > 1 else 1.0
        z_score = (mid_price - rolling_mean) / rolling_std
        print(f"[{product}] Best Bid: {best_bid}, Best Ask: {best_ask}, Mid Price: {mid_price:.2f}")
        print(f"[{product}] Rolling Mean: {rolling_mean:.2f}, Rolling Std: {rolling_std:.2f}, Z-Score: {z_score:.2f}")

        # Determine signal based on z-score.
        signal = 0
        if z_score < -1.0:
            signal = 1    # BUY signal.
        elif z_score > 1.0:
            signal = -1   # SELL signal.

        # === Inventory-Safe Quoting with Mean-Reversion Adjustment ===

        sensitivity = 0.5  # Factor to determine how much mispricing (delta) affects the quote
        
        if signal == 1:
            # Undervalued: mid_price below rolling mean => BUY signal.
            # Compute delta (how much lower the price is compared to the fair value).
            delta = rolling_mean - mid_price   # positive if undervalued
            # Raise our bid aggressively: add a premium proportional to delta.
            my_bid = best_bid + 1 + int(delta * sensitivity)
        elif signal == -1:
            # Overvalued: mid_price above rolling mean => SELL signal.
            delta = mid_price - rolling_mean   # positive if overvalued
            # Lower our ask aggressively: subtract a discount proportional to delta.
            my_ask = best_ask - 1 - int(delta * sensitivity)
        else:
            # No strong signal: use default quoting based solely on order book plus inventory skew.
            if len(history) > 10:
                my_bid = best_bid + 1 
                my_ask = best_ask - 1 

        # --- Size Adjustment ---
        # Base order size increases when inventory is flat.
        # Additionally, we apply a multiplier based on how extreme the mispricing is.
        if abs(z_score) > 1.0:
            # Scale order size as a function of the z_score beyond the threshold.
            # For example, if z_score is -2, multiplier becomes 1 + (2-1)*0.5 = 1.5.
            order_multiplier = 1 + (abs(z_score) - 1) * 0.5
        else:
            order_multiplier = 1

        # Compute size based on flat inventory, then scale it.
        size = int((base_size + max(0, (limit - abs(position)) // 10)) * order_multiplier)
        buyable = max(0, limit - position)  # maximum additional units we can buy
        sellable = max(0, limit + position)  # maximum additional units we can sell

        # --- Issue Orders Based on Signal ---
        if signal == 1 and buyable > 0:
            order_quantity = min(size, buyable)
            kelp_orders.append(Order(product, my_bid, order_quantity))
            print(f"[{product}] Issuing BUY order for {order_quantity} units at price {my_bid}")
        elif signal == -1 and sellable > 0:
            order_quantity = min(size, sellable)
            kelp_orders.append(Order(product, my_ask, -order_quantity))
            print(f"[{product}] Issuing SELL order for {order_quantity} units at price {my_ask}")
        else:
            if len(history) > 10:
                if buyable > 0:
                    kelp_orders.append(Order(product, my_bid, min(size, buyable)))
                if sellable > 0:
                    kelp_orders.append(Order(product, my_ask, -min(size, sellable)))
            
        # === Finalize Orders ===    
        result[product] = kelp_orders
        self.time += 1
        traderData = "KelpTrader_SGC_v1"
        conversions = 0
        
        return result, conversions, traderData