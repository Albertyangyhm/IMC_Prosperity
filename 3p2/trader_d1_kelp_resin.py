from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import numpy as np
import pandas as pd
import string
import math 
from math import erf

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
      self.ame_hist_info = pd.DataFrame()
      self.sf_hist_info = pd.DataFrame()
      self.star_prev_bid = 0 
      self.star_prev_ask = 0
      self.r3_opened = 0
      self.bs_cont = 0
      self.cnut_iv_list = []
      self.spike_list = []
      self.CLEAR_POS = False


      self.makelp = []
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

    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        market_trades = state.market_trades
        result = {}
        conversions = 0 
        for product in state.order_depths:
            if product == 'KELP':
                sf_position = state.position.get('KELP', 0)
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []

                # 1) Gather up to 3 levels of asks (sell_orders) and bids (buy_orders)
                #    NOTE: By convention, 'sell_orders' often hold negative volumes in your framework
                all_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])[:3]
                all_bids = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])[:3]

                # 2) Compute the live mid price & store in self.makelp
                best_ask = all_asks[0][0]
                best_bid = all_bids[0][0]
                mid_price = 0.5 * (best_ask + best_bid) 
                self.makelp.append(mid_price)

                # 3) Only proceed if you have at least 50 data points for a MA(50)
                window = 5
                if len(self.makelp) >= 1:
                    fair_price = np.mean(self.makelp[-window:]) 

                    # Define your half‐spread
                    half_spread = 0.65  #0.4 0.6 0.65 0.7 0.5
                    buy_quote = fair_price - half_spread
                    sell_quote = fair_price + half_spread

                    # 4) Position constraints
                    max_position = 50
                    buyable_qty = max_position - sf_position     # how many more we *can* buy
                    sellable_qty = sf_position + max_position    # how many we can sell without going below -max_position

                    # =========== 5) TAKE EXISTING LIQUIDITY ON THE ASK SIDE ===========
                    # If ask price <= buy_quote, we want to buy that liquidity
                    for ask_price, ask_vol in all_asks:
                        if ask_price <= buy_quote and buyable_qty > 0:
                            # ask_vol is typically negative, so the max you can buy is min(buyable_qty, -ask_vol)
                            can_buy = min(buyable_qty, -ask_vol)
                            if can_buy > 0:
                                orders.append(Order('KELP', ask_price, can_buy))
                                buyable_qty -= can_buy
                        else:
                            break  # The asks are sorted ascending; no need to check higher prices if we didn't buy this one

                    # After taking what's available, if we still have capacity, post our own buy limit at buy_quote
                    if buyable_qty > 0:
                        # Example: quote up to 5 lots (or up to buyable_qty)
                        size = buyable_qty
                        orders.append(Order('KELP', int(round(buy_quote)), size))

                    # =========== 6) TAKE EXISTING LIQUIDITY ON THE BID SIDE ===========
                    # If bid price >= sell_quote, we want to sell into it
                    for bid_price, bid_vol in all_bids:
                        if bid_price >= sell_quote and sellable_qty > 0:
                            # bid_vol is typically positive
                            can_sell = min(sellable_qty, bid_vol)
                            if can_sell > 0:
                                orders.append(Order('KELP', bid_price, -can_sell))
                                sellable_qty -= can_sell
                        else:
                            break  # No need to check lower bids if this one wasn't filled

                    # After taking what's available, if we still have capacity, post our own sell limit at sell_quote
                    if sellable_qty > 0:
                        size = round(sellable_qty)
                        orders.append(Order('KELP', int(round(sell_quote)), -size))

                else:
                    # Not enough data yet: for example, just store initial star_prev values or skip
                    self.star_prev_ask = best_ask
                    self.star_prev_bid = best_bid

                # Finally, record the orders
                result[product] = orders
            
            if product == 'RAINFOREST_RESIN':
                rf_orders = []                
                position = state.position.get(product, 0)
                limit = 50
                base_size = 10

                order_depth = state.order_depths[product]
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                best_ask_volume = order_depth.sell_orders[best_ask]

                spread = best_ask - best_bid
                fair_value = 10000

                # === Improved quoting logic ===

                # Quote skew based on inventory
                skew = int(-position * 0.05)

                # Adaptive quoting (inside spread if wide)
                if spread >= 4:
                    my_bid = best_bid + 1 + skew
                    my_ask = best_ask - 1 + skew
                else:
                    my_bid = 9999 + skew
                    my_ask = 10001 + skew

                # Clamp to outer bounds
                my_bid = min(my_bid, 9999)
                my_ask = max(my_ask, 10001)

                # Size adjustment — larger when flat
                size = base_size + max(0, (limit - abs(position)) // 10)

                # === Inventory-safe quoting ===
                buyable = max(0, limit - position)
                sellable = max(0, limit + position)

                if buyable > 0:
                    rf_orders.append(Order(product, my_bid, min(size, buyable)))
                if sellable > 0:
                    rf_orders.append(Order(product, my_ask, -min(size, sellable)))

                # === Light trimming logic (only if within safe bounds)
                if 40 > position >= 30:
                    trim_size = min(5, best_bid_volume, sellable)
                    if trim_size > 0:
                        rf_orders.append(Order(product, best_bid, -trim_size))
                elif -40 < position <= -30:
                    trim_size = min(5, best_ask_volume, buyable)
                    if trim_size > 0:
                        rf_orders.append(Order(product, best_ask, trim_size))

                # === Hard cut at edges (no limit breach)
                if position >= 40:
                    cut_size = min(10, best_bid_volume, sellable)
                    if cut_size > 0:
                        rf_orders.append(Order(product, best_bid, -cut_size))
                elif position <= -40:
                    cut_size = min(10, best_ask_volume, buyable)
                    if cut_size > 0:
                        rf_orders.append(Order(product, best_ask, cut_size))

                # === Finalize output
                result[product] = rf_orders
                self.time += 1


        
        self.stateHistory.append(state)
        traderData = "SANJIAER" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
                            # No idea what this is for yet 
        # print('Conversions is ',conversions)
        print(state.position)
        return result, conversions, traderData