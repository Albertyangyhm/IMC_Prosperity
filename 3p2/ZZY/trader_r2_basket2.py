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
    self.picnic_basket_2_mid = deque(maxlen=20)
    self.pic2_flag = 0
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
    pic2_position = state.position.get('PICNIC_BASKET2', 0)

    crs_order_depth = state.order_depths['CROISSANTS']
    jam_order_depth = state.order_depths['JAMS']
    pic2_order_depth = state.order_depths['PICNIC_BASKET2']

    crs_best_ask = min(crs_order_depth.sell_orders.keys())
    crs_best_ask_volume = abs(crs_order_depth.sell_orders[crs_best_ask])
    crs_best_bid = max(crs_order_depth.buy_orders.keys())
    crs_best_bid_volume = abs(crs_order_depth.buy_orders[crs_best_bid])

    jam_best_ask = min(jam_order_depth.sell_orders.keys())
    jam_best_ask_volume = abs(jam_order_depth.sell_orders[jam_best_ask])
    jam_best_bid = max(jam_order_depth.buy_orders.keys())
    jam_best_bid_volume = abs(jam_order_depth.buy_orders[jam_best_bid])

    pic2_best_ask = min(pic2_order_depth.sell_orders.keys())
    pic2_best_ask_volume = abs(pic2_order_depth.sell_orders[pic2_best_ask])
    pic2_best_bid = max(pic2_order_depth.buy_orders.keys())
    pic2_best_bid_volume = abs(pic2_order_depth.buy_orders[pic2_best_bid])

    crs_mid = int(0.5 * (crs_best_bid + crs_best_ask))
    jam_mid = int(0.5 * (jam_best_bid + jam_best_ask))
    pic2_mid = int(0.5 * (pic2_best_bid + pic2_best_ask))

    synthetic_price = 4 * crs_mid + 2 * jam_mid 
    spread = pic2_mid - synthetic_price
    self.mid_price_diff.append(spread)
    self.picnic_basket_2_mid.append(pic2_mid)

    spread_cost = (
        1 * (crs_best_ask - crs_best_bid) +
        1 * (jam_best_ask - jam_best_bid) +
        (pic2_best_ask - pic2_best_bid)
    )

    crs_orders = []
    jam_orders = []
    dje_orders = []
    pic2_orders = []

    buyable = defaultdict(int)
    sellable = defaultdict(int)
    buyable['CROISSANTS'] = max(0, 250 - crs_position)
    sellable['CROISSANTS'] = max(0, crs_position + 250)
    buyable['JAMS'] = max(0, 350 - jam_position)
    sellable['JAMS'] = max(0, jam_position + 350)
    buyable['PICNIC_BASKET2'] = max(0, 60 - pic2_position)
    sellable['PICNIC_BASKET2'] = max(0, pic2_position + 60)
    max_vol = 20
    min_vol = 2 
    min_multiplier = 0.2 
    base_vol = 10
    
    if crs_position == jam_position == pic2_position == 0:
        self.pic2_flag = 0

    if len(self.mid_price_diff) < 20:
       return result,0,'ZZY'
    
    else:
        median = np.median(self.mid_price_diff)
        mad = np.median(np.abs(self.mid_price_diff - median)) + 1e-8
        z_score = (spread - median) / mad
        vol = np.std(self.picnic_basket_2_mid)
        vol = min(max(vol, min_vol), max_vol)

        size_multiplier = (max_vol - vol) / (max_vol - min_vol)
        size_multiplier = max(min_multiplier, min(size_multiplier, 1.0))
        raw_size = int(round(base_vol * size_multiplier))

        if z_score > 1.5 and abs(spread - median) > 1.3 * spread_cost:
            max_buy_vol = min(
                crs_best_ask_volume, buyable['CROISSANTS'] // 4,
                jam_best_ask_volume, buyable['JAMS'] // 2,
            )
            max_sell_vol = min(pic2_best_bid_volume, sellable['PICNIC_BASKET2'])
            trade_vol = min(raw_size,max_buy_vol, max_sell_vol)
            crs_orders.append(Order('CROISSANTS', crs_mid, 4 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, 2 * trade_vol))
            pic2_orders.append(Order('PICNIC_BASKET2', pic2_mid, -trade_vol))
            self.pic2_flag = -1

        elif z_score < -2.2 and abs(spread - median) > 1.3 * spread_cost:
            max_sell_vol = min(
                crs_best_bid_volume, sellable['CROISSANTS'] // 4,
                jam_best_bid_volume, sellable['JAMS'] // 2,
            )
            max_buy_vol = min(raw_size,pic2_best_ask_volume, buyable['PICNIC_BASKET2'])
            trade_vol = min(max_buy_vol, max_sell_vol)
            crs_orders.append(Order('CROISSANTS', crs_mid, -4 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, -2 * trade_vol))
            pic2_orders.append(Order('PICNIC_BASKET2', pic2_mid, trade_vol))
            self.pic2_flag = 1

        elif self.pic2_flag == -1 and z_score < 0.5:
            max_sell_vol = min(
                crs_best_bid_volume, sellable['CROISSANTS'] // 4,
                jam_best_bid_volume, sellable['JAMS'] // 2,
            )
            max_buy_vol = min(abs(pic2_best_ask_volume), buyable['PICNIC_BASKET2'])
            trade_vol = min(max_buy_vol, max_sell_vol,pic2_position)
            crs_orders.append(Order('CROISSANTS', crs_mid, -4 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, -2 * trade_vol))
            pic2_orders.append(Order('PICNIC_BASKET2', pic2_mid, trade_vol))
            self.pic2_flag = 0

        elif self.pic2_flag == 1 and z_score > -0.5:
            max_buy_vol = min(
                abs(crs_best_ask_volume), buyable['CROISSANTS'] // 4,
                abs(jam_best_ask_volume), buyable['JAMS'] // 2,
            )
            max_sell_vol = min(pic2_best_bid_volume, sellable['PICNIC_BASKET2'])
            trade_vol = min(max_buy_vol, max_sell_vol,pic2_position)
            crs_orders.append(Order('CROISSANTS', crs_mid, 4 * trade_vol))
            jam_orders.append(Order('JAMS', jam_mid, 2 * trade_vol))
            pic2_orders.append(Order('PICNIC_BASKET2', pic2_mid, -trade_vol))
            self.pic2_flag = 0

    result['CROISSANTS'] = crs_orders
    result['JAMS'] = jam_orders
    result['PICNIC_BASKET2'] = pic2_orders
    self.time += 1
    traderData = 'ZZY'
    conversions = 0
    return result, conversions, traderData
