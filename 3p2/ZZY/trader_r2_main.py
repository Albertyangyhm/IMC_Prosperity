from typing import Dict, List
import math
import pandas as pd 
import numpy as np
from collections import defaultdict, deque
from datamodel import OrderDepth, TradingState, Order
from math import erf

class Trader:
    def __init__(self):

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

        # Shared state histories if needed
        self.stateHistory = []
        self.time = 0

        # == Basket1 variables ==
        self.pic1_flag = 0
        self.picnic_basket_1_mid = deque(maxlen=20)
        self.mid_price_diff_b1 = deque(maxlen=100)

        # == Basket2 variables ==
        self.pic2_flag = 0
        self.picnic_basket_2_mid = deque(maxlen=20)
        self.mid_price_diff_b2 = deque(maxlen=100)

        # Example base parameters you can tune
        self.base_vol_b1 = 10
        self.base_vol_b2 = 10

        # For partial internalization
        self.partial_orders_b1 = defaultdict(int)
        self.partial_orders_b2 = defaultdict(int)

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
     
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Single run method that processes both Basket1 and Basket2 in one shot,
        merges orders (internalizes) for net trades.
        """
        result = {}
        conversions = 0  # not used here, but we keep the structure
        traderData = "ZZY"  # you can store any JSON if needed
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
 
        # Initialize empty lists for each product
        for product in ["CROISSANTS", "JAMS", "DJEMBES", 
                        "PICNIC_BASKET1", "PICNIC_BASKET2"]:
            result[product] = []

        # =========================
        # 1) Process Basket1 logic
        # =========================

        # Acquire positions
        crs_pos = state.position.get("CROISSANTS", 0)
        jam_pos = state.position.get("JAMS", 0)
        dje_pos = state.position.get("DJEMBES", 0)
        pic1_pos = state.position.get("PICNIC_BASKET1", 0)

        # Acquire order depths
        crs_depth = state.order_depths["CROISSANTS"]
        jam_depth = state.order_depths["JAMS"]
        dje_depth = state.order_depths["DJEMBES"]
        pic1_depth = state.order_depths["PICNIC_BASKET1"]

        # Compute best bids/asks
        crs_best_ask = min(crs_depth.sell_orders.keys())
        crs_best_ask_vol = abs(crs_depth.sell_orders[crs_best_ask])
        crs_best_bid = max(crs_depth.buy_orders.keys())
        crs_best_bid_vol = abs(crs_depth.buy_orders[crs_best_bid])

        jam_best_ask = min(jam_depth.sell_orders.keys())
        jam_best_ask_vol = abs(jam_depth.sell_orders[jam_best_ask])
        jam_best_bid = max(jam_depth.buy_orders.keys())
        jam_best_bid_vol = abs(jam_depth.buy_orders[jam_best_bid])

        dje_best_ask = min(dje_depth.sell_orders.keys())
        dje_best_ask_vol = abs(dje_depth.sell_orders[dje_best_ask])
        dje_best_bid = max(dje_depth.buy_orders.keys())
        dje_best_bid_vol = abs(dje_depth.buy_orders[dje_best_bid])

        pic1_best_ask = min(pic1_depth.sell_orders.keys())
        pic1_best_ask_vol = abs(pic1_depth.sell_orders[pic1_best_ask])
        pic1_best_bid = max(pic1_depth.buy_orders.keys())
        pic1_best_bid_vol = abs(pic1_depth.buy_orders[pic1_best_bid])

        # Compute mid
        crs_mid = int(0.5 * (crs_best_ask + crs_best_bid))
        jam_mid = int(0.5 * (jam_best_ask + jam_best_bid))
        dje_mid = int(0.5 * (dje_best_ask + dje_best_bid))
        pic1_mid = int(0.5 * (pic1_best_ask + pic1_best_bid))

        # Basket1 synthetic
        synth_b1 = 6*crs_mid + 3*jam_mid + 1*dje_mid
        spread_b1 = pic1_mid - synth_b1

        # Track for rolling stats
        self.mid_price_diff_b1.append(spread_b1)
        self.picnic_basket_1_mid.append(pic1_mid)

        # Basic check
        if len(self.mid_price_diff_b1) < 20:
            # not enough data yet to trade basket1
            pass
        else:
            # Compute metrics
            mean_b1 = np.mean(self.mid_price_diff_b1)
            std_b1 = np.std(self.mid_price_diff_b1) + 1e-8
            z_b1 = (spread_b1 - mean_b1)/std_b1

            # Possibly do a spread cost check:
            spread_cost_b1 = (
                (crs_best_ask - crs_best_bid)
                + (jam_best_ask - jam_best_bid)
                + (dje_best_ask - dje_best_bid)
                + (pic1_best_ask - pic1_best_bid)
            )

            # Vol-based sizing
            vol_b1 = np.std(self.picnic_basket_1_mid)
            vol_b1 = max(2, min(vol_b1, 20))  # clamp 2..20
            scale = (20 - vol_b1)/(20 - 2)
            scale = max(0.2, min(scale, 1.0))
            raw_size_b1 = int(round(self.base_vol_b1 * scale))

            # We'll do a simple threshold logic like your code:
            # Example thresholds
            upper_thresh = 1.645
            lower_thresh = -1.645

            # Overvalued: short basket, long components
            if z_b1 > upper_thresh and abs(spread_b1 - mean_b1) > 1.3*spread_cost_b1:
                # See how many we can buy in the components:
                max_buy_crs = min(crs_best_ask_vol, (250 - crs_pos)//6)
                max_buy_jam = min(jam_best_ask_vol, (350 - jam_pos)//3)
                max_buy_dje = min(dje_best_ask_vol, (60 - dje_pos)//1)
                # short basket:
                max_sell_basket = min(pic1_best_bid_vol, 60 + pic1_pos)
                trade_vol = min(raw_size_b1, max_buy_crs, max_buy_jam, max_buy_dje, max_sell_basket)
                if trade_vol > 0:
                    self.partial_orders_b1["CROISSANTS"] += 6*trade_vol
                    self.partial_orders_b1["JAMS"]      += 3*trade_vol
                    self.partial_orders_b1["DJEMBES"]   += 1*trade_vol
                    self.partial_orders_b1["PICNIC_BASKET1"] -= trade_vol
                    self.pic1_flag = -1

            # Undervalued: long basket, short components
            elif z_b1 < lower_thresh and abs(spread_b1 - mean_b1) > 1.3*spread_cost_b1:
                max_sell_crs = min(crs_best_bid_vol, (250 + crs_pos)//6)
                max_sell_jam = min(jam_best_bid_vol, (350 + jam_pos)//3)
                max_sell_dje = min(dje_best_bid_vol, (60 + dje_pos)//1)
                max_buy_basket = min(pic1_best_ask_vol, 60 - pic1_pos)
                trade_vol = min(raw_size_b1, max_sell_crs, max_sell_jam, max_sell_dje, max_buy_basket)
                if trade_vol > 0:
                    self.partial_orders_b1["CROISSANTS"] -= 6*trade_vol
                    self.partial_orders_b1["JAMS"]       -= 3*trade_vol
                    self.partial_orders_b1["DJEMBES"]    -= 1*trade_vol
                    self.partial_orders_b1["PICNIC_BASKET1"] += trade_vol
                    self.pic1_flag = 1

            # Exit short
            elif self.pic1_flag == -1 and z_b1 < 0.5:
                # close entire short: volume = abs(pic1_pos) etc
                close_vol = min(abs(pic1_pos), pic1_best_ask_vol)
                close_crs = min(crs_best_bid_vol, abs(crs_pos)//6)
                # ...
                # For brevity, just do the min of them all
                trade_vol = min(close_vol, close_crs)
                if trade_vol > 0:
                    self.partial_orders_b1["CROISSANTS"] -= 6*trade_vol
                    self.partial_orders_b1["JAMS"]       -= 3*trade_vol
                    self.partial_orders_b1["DJEMBES"]    -= 1*trade_vol
                    self.partial_orders_b1["PICNIC_BASKET1"] += trade_vol
                    self.pic1_flag = 0

            # Exit long
            elif self.pic1_flag == 1 and z_b1 > -0.5:
                close_vol = min(abs(pic1_pos), pic1_best_bid_vol)
                close_crs = min(crs_best_ask_vol, abs(crs_pos)//6)
                # ...
                trade_vol = min(close_vol, close_crs)
                if trade_vol > 0:
                    self.partial_orders_b1["CROISSANTS"] += 6*trade_vol
                    self.partial_orders_b1["JAMS"]       += 3*trade_vol
                    self.partial_orders_b1["DJEMBES"]    += 1*trade_vol
                    self.partial_orders_b1["PICNIC_BASKET1"] -= trade_vol
                    self.pic1_flag = 0

        # =========================
        # 2) Process Basket2 logic
        # =========================

        crs2_pos = crs_pos  # same positions
        jam2_pos = jam_pos
        pic2_pos = state.position.get("PICNIC_BASKET2", 0)

        # Confirm we have order depths
        crs2_depth = state.order_depths["CROISSANTS"]
        jam2_depth = state.order_depths["JAMS"]
        pic2_depth = state.order_depths["PICNIC_BASKET2"]

        # Best bids/asks for basket2
        crs2_best_ask = min(crs2_depth.sell_orders.keys())
        crs2_best_ask_vol = abs(crs2_depth.sell_orders[crs2_best_ask])
        crs2_best_bid = max(crs2_depth.buy_orders.keys())
        crs2_best_bid_vol = abs(crs2_depth.buy_orders[crs2_best_bid])

        jam2_best_ask = min(jam2_depth.sell_orders.keys())
        jam2_best_ask_vol = abs(jam2_depth.sell_orders[jam2_best_ask])
        jam2_best_bid = max(jam2_depth.buy_orders.keys())
        jam2_best_bid_vol = abs(jam2_depth.buy_orders[jam2_best_bid])

        pic2_best_ask = min(pic2_depth.sell_orders.keys())
        pic2_best_ask_vol = abs(pic2_depth.sell_orders[pic2_best_ask])
        pic2_best_bid = max(pic2_depth.buy_orders.keys())
        pic2_best_bid_vol = abs(pic2_depth.buy_orders[pic2_best_bid])

        crs2_mid = int(0.5*(crs2_best_ask + crs2_best_bid))
        jam2_mid = int(0.5*(jam2_best_ask + jam2_best_bid))
        pic2_mid = int(0.5*(pic2_best_ask + pic2_best_bid))

        synth_b2 = 4*crs2_mid + 2*jam2_mid
        spread_b2 = pic2_mid - synth_b2
        self.mid_price_diff_b2.append(spread_b2)
        self.picnic_basket_2_mid.append(pic2_mid)

        if len(self.mid_price_diff_b2) < 20:
            pass  # not enough data
        else:
            median_b2 = np.median(self.mid_price_diff_b2)
            mad_b2 = np.median(np.abs(self.mid_price_diff_b2 - median_b2)) + 1e-8
            z_b2 = (spread_b2 - median_b2)/mad_b2

            spread_cost_b2 = ((crs2_best_ask - crs2_best_bid) + 
                              (jam2_best_ask - jam2_best_bid) + 
                              (pic2_best_ask - pic2_best_bid))

            vol_b2 = np.std(self.picnic_basket_2_mid)
            vol_b2 = max(2, min(vol_b2, 20))
            scale2 = (20 - vol_b2)/(20 - 2)
            scale2 = max(0.2, min(scale2, 1.0))
            raw_size_b2 = int(round(self.base_vol_b2 * scale2))

            # Asymmetric threshold example
            if z_b2 > 1.5 and abs(spread_b2 - median_b2) > 1.3 * spread_cost_b2:
                max_buy_crs = min(crs2_best_ask_vol, (250 - crs2_pos)//4)
                max_buy_jam = min(jam2_best_ask_vol, (350 - jam2_pos)//2)
                max_sell_pic2 = min(pic2_best_bid_vol, 60 + pic2_pos)
                trade_vol = min(raw_size_b2, max_buy_crs, max_buy_jam, max_sell_pic2)
                if trade_vol > 0:
                    self.partial_orders_b2["CROISSANTS"] += 4*trade_vol
                    self.partial_orders_b2["JAMS"]       += 2*trade_vol
                    self.partial_orders_b2["PICNIC_BASKET2"] -= trade_vol
                    self.pic2_flag = -1

            elif z_b2 < -2.2 and abs(spread_b2 - median_b2) > 1.3 * spread_cost_b2:
                max_sell_crs = min(crs2_best_bid_vol, (250 + crs2_pos)//4)
                max_sell_jam = min(jam2_best_bid_vol, (350 + jam2_pos)//2)
                max_buy_pic2 = min(pic2_best_ask_vol, 60 - pic2_pos)
                trade_vol = min(raw_size_b2, max_sell_crs, max_sell_jam, max_buy_pic2)
                if trade_vol > 0:
                    self.partial_orders_b2["CROISSANTS"] -= 4*trade_vol
                    self.partial_orders_b2["JAMS"]       -= 2*trade_vol
                    self.partial_orders_b2["PICNIC_BASKET2"] += trade_vol
                    self.pic2_flag = 1

            elif self.pic2_flag == -1 and z_b2 < 0.5:
                close_vol = min(abs(pic2_pos), pic2_best_ask_vol)
                close_crs = min(crs2_best_bid_vol, abs(crs2_pos)//4)
                # ...
                trade_vol = min(close_vol, close_crs)
                if trade_vol > 0:
                    self.partial_orders_b2["CROISSANTS"] -= 4*trade_vol
                    self.partial_orders_b2["JAMS"]       -= 2*trade_vol
                    self.partial_orders_b2["PICNIC_BASKET2"] += trade_vol
                    self.pic2_flag = 0

            elif self.pic2_flag == 1 and z_b2 > -0.5:
                close_vol = min(abs(pic2_pos), pic2_best_bid_vol)
                close_crs = min(crs2_best_ask_vol, abs(crs2_pos)//4)
                # ...
                trade_vol = min(close_vol, close_crs)
                if trade_vol > 0:
                    self.partial_orders_b2["CROISSANTS"] += 4*trade_vol
                    self.partial_orders_b2["JAMS"]       += 2*trade_vol
                    self.partial_orders_b2["PICNIC_BASKET2"] -= trade_vol
                    self.pic2_flag = 0

        # ================================
        # 3) Net out partial orders
        # ================================

        # We'll combine partial_orders_b1 + partial_orders_b2
        # So for each product, we sum them up to get final net quantity
        net_orders_dict = defaultdict(int)

        # Merge basket1 partial
        for prod, qty in self.partial_orders_b1.items():
            net_orders_dict[prod] += qty

        # Merge basket2 partial
        for prod, qty in self.partial_orders_b2.items():
            net_orders_dict[prod] += qty

        # Now create final Order objects from net_orders_dict
        # We'll assume mid price for each product to ensure immediate crossing if qty>0 => buy at best_ask, etc.

        # We'll build a small helper to figure out best bid/ask from state
        def best_prices(product):
            od = state.order_depths[product]
            bestA = min(od.sell_orders.keys())
            bestB = max(od.buy_orders.keys())
            return bestA, bestB

        # For each net product
        final_orders = defaultdict(list)

        for product, net_qty in net_orders_dict.items():
            if net_qty == 0:
                continue
            best_ask, best_bid = best_prices(product)
            mid = int(0.5 * (best_ask + best_bid)) 
            if net_qty > 0:
                # net buy => we want to buy from best ask
                final_orders[product].append(Order(product, mid, net_qty))
            else:
                # net sell => we want to sell at best bid
                final_orders[product].append(Order(product, mid, net_qty))

        # Put them into result
        for product in final_orders:
            result[product] = final_orders[product]

        # Clear partial orders after each run
        self.partial_orders_b1.clear()
        self.partial_orders_b2.clear()
        self.stateHistory.append(state)
        self.time += 1
        return result, conversions, traderData
