from typing import Dict, List
import math
import statistics
import pandas as pd
from collections import defaultdict,deque 
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
    self.synthetic_mid_prev = None 
    self.picnic_basket_1_mid = deque(maxlen=20)
    self.pic1_flag = 0
    self.mid_price_diff = deque(maxlen=100)

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
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """
    result = {}
    crs_position = state.position.get('CROISSANTS', 0)
    jam_position = state.position.get('JAMS', 0)
    dje_position = state.position.get('DJEMBES', 0)
    pic1_position = state.position.get('PICNIC_BASKET1', 0)

    crs_order_depth = state.order_depths['CROISSANTS']
    jam_order_depth = state.order_depths['JAMS']
    dje_order_depth = state.order_depths['DJEMBES']
    pic1_order_depth = state.order_depths['PICNIC_BASKET1']

    crs_best_ask = min(crs_order_depth.sell_orders.keys())
    crs_best_ask_volume = abs(crs_order_depth.sell_orders[crs_best_ask])
    crs_best_bid = max(crs_order_depth.buy_orders.keys())
    crs_best_bid_volume = abs(crs_order_depth.buy_orders[crs_best_bid])

    jam_best_ask = min(jam_order_depth.sell_orders.keys())
    jam_best_ask_volume = abs(jam_order_depth.sell_orders[jam_best_ask])
    jam_best_bid = max(jam_order_depth.buy_orders.keys())
    jam_best_bid_volume = abs(jam_order_depth.buy_orders[jam_best_bid])

    dje_best_ask = min(dje_order_depth.sell_orders.keys())
    dje_best_ask_volume = abs(dje_order_depth.sell_orders[dje_best_ask])
    dje_best_bid = max(dje_order_depth.buy_orders.keys())
    dje_best_bid_volume = abs(dje_order_depth.buy_orders[dje_best_bid])

    pic1_best_ask = min(pic1_order_depth.sell_orders.keys())
    pic1_best_ask_volume = abs(pic1_order_depth.sell_orders[pic1_best_ask])
    pic1_best_bid = max(pic1_order_depth.buy_orders.keys())
    pic1_best_bid_volume = abs(pic1_order_depth.buy_orders[pic1_best_bid])

    crs_mid = int(0.5 * (crs_best_bid + crs_best_ask))
    jam_mid = int(0.5 * (jam_best_bid + jam_best_ask))
    dje_mid = int(0.5 * (dje_best_bid + dje_best_ask))
    pic1_mid = int(0.5 * (pic1_best_bid + pic1_best_ask))

    synthetic_price = 6 * crs_mid + 3 * jam_mid + 1 * dje_mid
    spread = pic1_mid - synthetic_price
    self.mid_price_diff.append(spread)
    self.picnic_basket_1_mid.append(pic1_mid)

    spread_cost = (
        1 * (crs_best_ask - crs_best_bid) +
        1 * (jam_best_ask - jam_best_bid) +
        1 * (dje_best_ask - dje_best_bid) +
        (pic1_best_ask - pic1_best_bid)
    )

    crs_orders = []
    jam_orders = []
    dje_orders = []
    pic1_orders = []

    buyable = defaultdict(int)
    sellable = defaultdict(int)
    buyable['CROISSANTS'] = max(0, 250 - crs_position)
    sellable['CROISSANTS'] = max(0, crs_position + 250)
    buyable['JAMS'] = max(0, 350 - jam_position)
    sellable['JAMS'] = max(0, jam_position + 350)
    buyable['DJEMBES'] = max(0, 60 - dje_position)
    sellable['DJEMBES'] = max(0, dje_position + 60)
    buyable['PICNIC_BASKET1'] = max(0, 60 - pic1_position)
    sellable['PICNIC_BASKET1'] = max(0, pic1_position + 60)
    max_vol = 20
    min_vol = 2 
    min_multiplier = 0.2 
    base_vol = 10
    
    if crs_position == jam_position == dje_position == pic1_position == 0:
        self.pic1_flag = 0

    if len(self.mid_price_diff) < 20:
       return result,0,'ZZY'
    
    else:
        mean = np.mean(self.mid_price_diff)
        std = np.std(self.mid_price_diff)
        z_score = (spread - mean) / (std + 1e-8)
        vol = np.std(self.picnic_basket_1_mid)
        vol = min(max(vol, min_vol), max_vol)

        size_multiplier = (max_vol - vol) / (max_vol - min_vol)
        size_multiplier = max(min_multiplier, min(size_multiplier, 1.0))
        raw_size = int(round(base_vol * size_multiplier))

        if z_score > 2 and abs(spread - mean) > 1.2 * spread_cost:
            max_buy_vol = min(
                crs_best_ask_volume, buyable['CROISSANTS'] // 6,
                jam_best_ask_volume, buyable['JAMS'] // 3,
                dje_best_ask_volume, buyable['DJEMBES']
            )
            max_sell_vol = min(pic1_best_bid_volume, sellable['PICNIC_BASKET1'])
            trade_vol = min(raw_size,max_buy_vol, max_sell_vol)
            crs_orders.append(Order('CROISSANTS', crs_mid, 6 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, 3 * trade_vol))
            dje_orders.append(Order('DJEMBES', dje_mid, 1 * trade_vol))
            pic1_orders.append(Order('PICNIC_BASKET1', pic1_mid, -trade_vol))
            self.pic1_flag = -1

        elif z_score < -2 and abs(spread - mean) > 1.2 * spread_cost:
            max_sell_vol = min(
                crs_best_bid_volume, sellable['CROISSANTS'] // 6,
                jam_best_bid_volume, sellable['JAMS'] // 3,
                dje_best_bid_volume, sellable['DJEMBES']
            )
            max_buy_vol = min(raw_size,pic1_best_ask_volume, buyable['PICNIC_BASKET1'])
            trade_vol = min(max_buy_vol, max_sell_vol)
            crs_orders.append(Order('CROISSANTS', crs_mid, -6 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, -3 * trade_vol))
            dje_orders.append(Order('DJEMBES', dje_mid, -1 * trade_vol))
            pic1_orders.append(Order('PICNIC_BASKET1', pic1_mid, trade_vol))
            self.pic1_flag = 1

        elif self.pic1_flag == -1 and z_score < 0.5:
            max_sell_vol = min(
                crs_best_bid_volume, sellable['CROISSANTS'] // 6,
                jam_best_bid_volume, sellable['JAMS'] // 3,
                dje_best_bid_volume, sellable['DJEMBES']
            )
            max_buy_vol = min(abs(pic1_best_ask_volume), buyable['PICNIC_BASKET1'])
            trade_vol = min(max_buy_vol, max_sell_vol,pic1_position)
            crs_orders.append(Order('CROISSANTS', crs_mid, -6 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, -3 * trade_vol))
            dje_orders.append(Order('DJEMBES', dje_mid, -1 * trade_vol))
            pic1_orders.append(Order('PICNIC_BASKET1', pic1_mid, trade_vol))
            self.pic1_flag = 0

        elif self.pic1_flag == 1 and z_score > -0.5:
            max_buy_vol = min(
                abs(crs_best_ask_volume), buyable['CROISSANTS'] // 6,
                abs(jam_best_ask_volume), buyable['JAMS'] // 3,
                abs(dje_best_ask_volume), buyable['DJEMBES']
            )
            max_sell_vol = min(pic1_best_bid_volume, sellable['PICNIC_BASKET1'])
            trade_vol = min(max_buy_vol, max_sell_vol,pic1_position)
            crs_orders.append(Order('CROISSANTS', crs_mid, 6 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, 3 * trade_vol))
            dje_orders.append(Order('DJEMBES', dje_mid, 1 * trade_vol))
            pic1_orders.append(Order('PICNIC_BASKET1', pic1_mid, -trade_vol))
            self.pic1_flag = 0

    result['CROISSANTS'] = crs_orders
    result['JAMS'] = jam_orders
    result['DJEMBES'] = dje_orders
    result['PICNIC_BASKET1'] = pic1_orders
    self.time += 1
    traderData = 'ZZY'
    conversions = 0
    return result, conversions, traderData

