from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
# from datamodel import Logger
# logger = Logger()

import math

# storing string as const to avoid typos
SUBMISSION = "SUBMISSION"
RR = "RAINFOREST_RESIN"
BANANAS = "SQUID_INK"

PRODUCTS = [
    RR,
    BANANAS,
]

DEFAULT_PRICES = {
    RR : 10_000,
    BANANAS : 5_000,
}

class Trader:

    def __init__(self) -> None:
        
        print("Initializing Trader...")

        self.position_limit = {
            RR : 20,
            BANANAS : 20,
        }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
        
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.ema_param = 0.1


    # utils
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

    def get_mid_price(self, product, state: TradingState, weight_by_volume: bool = False):
        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no ask orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)

        if weight_by_volume:
            bid_volume = sum(volume for price, volume in market_bids.items())
            ask_volume = sum(volume for price, volume in market_asks.items())
            total_volume = bid_volume + ask_volume

            if total_volume == 0:
                return default_price

            weighted_bid_price = sum(price * volume for price, volume in market_bids.items()) / bid_volume
            weighted_ask_price = sum(price * volume for price, volume in market_bids.items()) / ask_volume

            return (weighted_bid_price + weighted_ask_price) / 2
        else:
            return (best_bid + best_ask) / 2

    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state, weight_by_volume=False)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]


    # Algorithm logic
    def rr_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of rr.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_rr = self.get_position(RR, state)

        bid_volume = self.position_limit[RR] - position_rr
        ask_volume = - self.position_limit[RR] - position_rr

        orders = []
        orders.append(Order(RR, DEFAULT_PRICES[RR] - 1, bid_volume))
        orders.append(Order(RR, DEFAULT_PRICES[RR] + 1, ask_volume))

        return orders

    def bananas_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of bananas.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_bananas = self.get_position(BANANAS, state)

        bid_volume = self.position_limit[BANANAS] - position_bananas
        ask_volume = - self.position_limit[BANANAS] - position_bananas

        orders = []

        if position_bananas == 0:
            # Not long nor short
            orders.append(Order(BANANAS, math.floor(self.ema_prices[BANANAS] - 1), bid_volume))
            orders.append(Order(BANANAS, math.ceil(self.ema_prices[BANANAS] + 1), ask_volume))
        
        if position_bananas > 0:
            # Long position
            orders.append(Order(BANANAS, math.floor(self.ema_prices[BANANAS] - 2), bid_volume))
            orders.append(Order(BANANAS, math.ceil(self.ema_prices[BANANAS]), ask_volume))

        if position_bananas < 0:
            # Short position
            orders.append(Order(BANANAS, math.floor(self.ema_prices[BANANAS]), bid_volume))
            orders.append(Order(BANANAS, math.ceil(self.ema_prices[BANANAS] + 2), ask_volume))

        return orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        self.round += 1
        self.update_ema_prices(state)
        
        # Initialize the method output dict as an empty dict
        result = {}

        # PEARL STRATEGY
        # try:
        #     result[RR] = self.rr_strategy(state)
        # except Exception as e:
        #     print("Error in rr strategy")
        #     print(e)

        # BANANA STRATEGY
        try:
            result[BANANAS] = self.bananas_strategy(state)
        except Exception as e:
            print("Error in bananas strategy")
            print(e)

        print("+---------------------------------+")
        conversions = 0
        trader_data = ""
        # logger.flush(state, result, conversions, trader_data)
        return result, 0, ""