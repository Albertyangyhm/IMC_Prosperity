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


  def run(self, state: TradingState):
    """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
    """
    self.stateHistory.append(state)
    result = {}
    sq_orders = []
    product = 'SQUID_INK'
    # Initialize the method output dict as an empty dict
    order_depth = state.order_depths[product]
    bid_prices, bid_vols, ask_prices, ask_vols = self.extract_orderbook_series('SQUID_INK')
    best_bid = max(state.order_depths['SQUID_INK'].buy_orders.keys())
    best_ask = min(state.order_depths['SQUID_INK'].sell_orders.keys())
    mid_price = (best_ask + best_bid) // 2

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

    # Book mid 
    curr = order_depth

    bid_prices_now, bid_vols_now = [], []
    ask_prices_now, ask_vols_now = [], []

    for i in range(1, 4):
        try:
            bid_p = sorted(curr.buy_orders.keys(), reverse=True)[i - 1]
            bid_v = curr.buy_orders[bid_p]
            bid_prices_now.append(bid_p)
            bid_vols_now.append(bid_v)
        except:
            continue

    for i in range(1, 4):
        try:
            ask_p = sorted(curr.sell_orders.keys())[i - 1]
            ask_v = curr.sell_orders[ask_p]
            ask_prices_now.append(ask_p)
            ask_vols_now.append(ask_v)
        except:
            continue

    # Remove NaNs and zeros
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
    mid_diff = current_mid - book_mid

    if mid_diff > 0.05:
        sq_orders.append(Order(product,int(book_mid),buyable))
        sq_orders.append(Order(product,int(current_mid),-sellable))
    
    elif mid_diff < -0.05:
        sq_orders.append(Order(product,int(book_mid),-sellable))
        sq_orders.append(Order(product,int(current_mid),buyable))
    
    else:
        if position > 0:
            sq_orders.append(Order(product,int(current_mid),-position))
        elif position <0:
            sq_orders.append(Order(product,int(current_mid),position))

    # === Finalize output
    result[product] = sq_orders
    self.time += 1
    traderData = 'ZZY'
    conversions = 0
    return result, conversions, traderData
