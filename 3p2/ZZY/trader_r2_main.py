from typing import Dict, List
import math
import numpy as np
from collections import defaultdict, deque
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Single run method that processes both Basket1 and Basket2 in one shot,
        merges orders (internalizes) for net trades.
        """
        result = {}
        conversions = 0  # not used here, but we keep the structure
        traderData = "ZZY"  # you can store any JSON if needed

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

        self.time += 1
        return result, conversions, traderData
