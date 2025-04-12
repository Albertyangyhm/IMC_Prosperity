from typing import Dict, List
import numpy as np
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        # State/history tracking and flags for basket arbitrage positions.
        self.stateHistory = []
        self.enteredLong = False
        self.enteredShort = False
        self.b1long = False   # flag for PICNIC_BASKET1 long arbitrage (basket long, underlying short)
        self.b1short = False  # flag for PICNIC_BASKET1 short arbitrage (basket short, underlying long)
        self.b2long = False   # flag for PICNIC_BASKET2 long arbitrage (basket long, underlying short)
        self.b2short = False  # flag for PICNIC_BASKET2 short arbitrage (basket short, underlying long)
        self.prevValue = 0
        self.time = 0
        
        # For debugging or tracking spread history (optional)
        self.b1spread = []
        self.b2spread = []
    
    def updateProductHist(self, productName: str, state: TradingState, ifPop: bool):
        """
        Optional function to update state history and extract historical information.
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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        This method implements a static-threshold index-arbitrage strategy.
        
        Basket compositions:
          - PICNIC_BASKET1: 6 CROISSANTS, 3 JAMS, 1 DJEMBES
          - PICNIC_BASKET2: 4 CROISSANTS, 2 JAMS
        
        New entries for each basket are allowed only if the current positions on the baskets
        and their underlying instruments are less than 2/3 of their position limits.
        This separation of safe-to-trade conditions lets you wager your index arbitrage in Basket 1
        first and then sequentially execute Basket 2 arbitrage.
        
        In the unwind (neutral signal) branches, we now delete the "if pos_X < 0:" conditions.
        Instead, we always send orders to flatten the underlying positions by issuing orders
        with the full quantity needed to flatten that position.
        """
        result = {}

        #########################################################
        # Section 1: Define Position Limits (Explicit Variables)
        #########################################################
        limit_croissant = 250
        limit_jam = 350
        limit_djembe = 60
        limit_basket1 = 60
        limit_basket2 = 100

        #########################################################
        # Section 2: Get Order Depths, Best Prices, and Volumes
        #########################################################
        # Initialize empty lists for orders
        croissant_orders: List[Order] = []
        jam_orders: List[Order] = []
        djembe_orders: List[Order] = []
        basket1_orders: List[Order] = []
        basket2_orders: List[Order] = []
        
        # Underlying: CROISSANTS
        croissant_od: OrderDepth = state.order_depths['CROISSANTS']
        croissant_best_ask = min(croissant_od.sell_orders.keys())
        croissant_best_bid = max(croissant_od.buy_orders.keys())
        mid_croissant = 0.5 * (croissant_best_ask + croissant_best_bid)
        croissant_ask_vol = croissant_od.sell_orders.get(croissant_best_ask, 0)
        croissant_bid_vol = croissant_od.buy_orders.get(croissant_best_bid, 0)
        
        # Underlying: JAMS
        jam_od: OrderDepth = state.order_depths['JAMS']
        jam_best_ask = min(jam_od.sell_orders.keys())
        jam_best_bid = max(jam_od.buy_orders.keys())
        mid_jam = 0.5 * (jam_best_ask + jam_best_bid)
        jam_ask_vol = jam_od.sell_orders.get(jam_best_ask, 0)
        jam_bid_vol = jam_od.buy_orders.get(jam_best_bid, 0)
        
        # Underlying: DJEMBES (for Basket1 only)
        djembe_od: OrderDepth = state.order_depths['DJEMBES']
        djembe_best_ask = min(djembe_od.sell_orders.keys())
        djembe_best_bid = max(djembe_od.buy_orders.keys())
        mid_djembe = 0.5 * (djembe_best_ask + djembe_best_bid)
        djembe_ask_vol = djembe_od.sell_orders.get(djembe_best_ask, 0)
        djembe_bid_vol = djembe_od.buy_orders.get(djembe_best_bid, 0)
        
        # Basket: PICNIC_BASKET1
        basket1_od: OrderDepth = state.order_depths['PICNIC_BASKET1']
        basket1_best_ask = min(basket1_od.sell_orders.keys())
        basket1_best_bid = max(basket1_od.buy_orders.keys())
        mid_basket1 = 0.5 * (basket1_best_ask + basket1_best_bid)
        basket1_ask_vol = basket1_od.sell_orders.get(basket1_best_ask, 0)
        basket1_bid_vol = basket1_od.buy_orders.get(basket1_best_bid, 0)
        
        # Basket: PICNIC_BASKET2
        basket2_od: OrderDepth = state.order_depths['PICNIC_BASKET2']
        basket2_best_ask = min(basket2_od.sell_orders.keys())
        basket2_best_bid = max(basket2_od.buy_orders.keys())
        mid_basket2 = 0.5 * (basket2_best_ask + basket2_best_bid)
        basket2_ask_vol = basket2_od.sell_orders.get(basket2_best_ask, 0)
        basket2_bid_vol = basket2_od.buy_orders.get(basket2_best_bid, 0)

        #########################################################
        # Section 3: Get Current Positions from State
        #########################################################
        pos_croissant = state.position.get('CROISSANTS', 0)
        pos_jam = state.position.get('JAMS', 0)
        pos_djembe = state.position.get('DJEMBES', 0)
        current_b1_pos = state.position.get('PICNIC_BASKET1', 0)
        current_b2_pos = state.position.get('PICNIC_BASKET2', 0)

        #########################################################
        # Section 3.5: Define Safe-to-Trade Conditions Separately
        #########################################################
        # For Basket1, check positions for Basket1, CROISSANTS, JAMS and DJEMBES.
        safe_to_trade_b1 = (
            abs(current_b1_pos) < (2/3) * limit_basket1 and
            abs(pos_croissant) < (2/3) * limit_croissant and
            abs(pos_jam) < (2/3) * limit_jam and
            abs(pos_djembe) < (2/3) * limit_djembe
        )
        # For Basket2, check positions for Basket2, CROISSANTS and JAMS.
        safe_to_trade_b2 = (
            abs(current_b2_pos) < (2/3) * limit_basket2 and
            abs(pos_croissant) < (2/3) * limit_croissant and
            abs(pos_jam) < (2/3) * limit_jam
        )

        #########################################################
        # Section 4: Compute Synthetic Prices and Spreads for Baskets
        #########################################################
        # Basket1 Synthetic Price: 6 CROISSANTS, 3 JAMS, 1 DJEMBES
        synthetic_b1 = 6 * mid_croissant + 3 * mid_jam + 1 * mid_djembe
        spread1 = mid_basket1 - synthetic_b1
        self.b1spread.append(spread1)
        
        # Basket2 Synthetic Price: 4 CROISSANTS, 2 JAMS
        synthetic_b2 = 4 * mid_croissant + 2 * mid_jam
        spread2 = mid_basket2 - synthetic_b2
        self.b2spread.append(spread2)

        #########################################################
        # Section 5: Calculate Tradeable Volumes
        #########################################################
        volume_b1_buy = min(
            (limit_basket1 - current_b1_pos),
            basket1_ask_vol,
            croissant_ask_vol // 6,
            jam_ask_vol // 3,
            djembe_ask_vol
        )
        volume_b1_sell = min(
            (limit_basket1 + current_b1_pos),
            basket1_bid_vol,
            croissant_bid_vol // 6,
            jam_bid_vol // 3,
            djembe_bid_vol
        )
        volume_b2_buy = min(
            (limit_basket2 - current_b2_pos),
            basket2_ask_vol,
            croissant_ask_vol // 4,
            jam_ask_vol // 2
        )
        volume_b2_sell = min(
            (limit_basket2 + current_b2_pos),
            basket2_bid_vol,
            croissant_bid_vol // 4,
            jam_bid_vol // 2
        )
        
        window = 3 
        rolling_spread1 = np.mean(self.b1spread[-window:]) if len(self.b1spread) >= window else spread1
        rolling_spread2 = np.mean(self.b1spread[-window:]) if len(self.b2spread) >= window else spread2

        #########################################################
        # Section 6: Define Static Thresholds (Hard Cutoffs)
        #########################################################
        # Basket1 thresholds (tunable)
        upper_threshold_b1 = 80   # if spread1 > this, basket1 is overpriced
        lower_threshold_b1 = -60  # if spread1 < this, basket1 is underpriced

        # Basket2 thresholds (tunable)
        upper_threshold_b2 = 65   # if spread2 > this, basket2 is overpriced
        lower_threshold_b2 = -50   # if spread2 < this, basket2 is underpriced

        #########################################################
        # Section 7: Entry/Exit Logic for Basket1 and its Underlyings
        #########################################################
        if safe_to_trade_b1:
            if rolling_spread1 > upper_threshold_b1:
                # Basket1 is overpriced: short basket1 and buy underlying.
                self.b1short = True
                basket1_orders.append(Order('PICNIC_BASKET1', basket1_best_bid, -volume_b1_sell))
                croissant_orders.append(Order('CROISSANTS', croissant_best_ask, 6 * volume_b1_sell))
                jam_orders.append(Order('JAMS', jam_best_ask, 3 * volume_b1_sell))
                djembe_orders.append(Order('DJEMBES', djembe_best_ask, 1 * volume_b1_sell))
            elif rolling_spread1 < lower_threshold_b1:
                # Basket1 is underpriced: buy basket1 and sell underlying.
                self.b1long = True
                basket1_orders.append(Order('PICNIC_BASKET1', basket1_best_ask, volume_b1_buy))
                croissant_orders.append(Order('CROISSANTS', croissant_best_bid, -6 * volume_b1_buy))
                jam_orders.append(Order('JAMS', jam_best_bid, -3 * volume_b1_buy))
                djembe_orders.append(Order('DJEMBES', djembe_best_bid, -1 * volume_b1_buy))
            else:
                # Neutral signal: unwind any open positions for Basket1 and its underlyings.
                if self.b1long:
                    basket1_orders.append(Order('PICNIC_BASKET1', basket1_best_bid, -current_b1_pos))
                    # Flatten underlying positions regardless of sign.
                    croissant_orders.append(Order('CROISSANTS', croissant_best_ask, pos_croissant))
                    jam_orders.append(Order('JAMS', jam_best_ask, pos_jam))
                    djembe_orders.append(Order('DJEMBES', djembe_best_ask, pos_djembe))
                    self.b1long = False
                if self.b1short:
                    basket1_orders.append(Order('PICNIC_BASKET1', basket1_best_ask, current_b1_pos))
                    croissant_orders.append(Order('CROISSANTS', croissant_best_bid, -pos_croissant))
                    jam_orders.append(Order('JAMS', jam_best_bid, -pos_jam))
                    djembe_orders.append(Order('DJEMBES', djembe_best_bid, -pos_djembe))
                    self.b1short = False
        else:
            # Not safe to trade new Basket1 entries; only unwind if needed.
            if self.b1long:
                basket1_orders.append(Order('PICNIC_BASKET1', basket1_best_bid, -current_b1_pos))
                croissant_orders.append(Order('CROISSANTS', croissant_best_ask, pos_croissant))
                jam_orders.append(Order('JAMS', jam_best_ask, pos_jam))
                djembe_orders.append(Order('DJEMBES', djembe_best_ask, pos_djembe))
                self.b1long = False
            if self.b1short:
                basket1_orders.append(Order('PICNIC_BASKET1', basket1_best_ask, current_b1_pos))
                croissant_orders.append(Order('CROISSANTS', croissant_best_bid, -pos_croissant))
                jam_orders.append(Order('JAMS', jam_best_bid, -pos_jam))
                djembe_orders.append(Order('DJEMBES', djembe_best_bid, -pos_djembe))
                self.b1short = False

        #########################################################
        # Section 8: Entry/Exit Logic for Basket2 and its Underlyings
        #########################################################
        if safe_to_trade_b2:
            if rolling_spread2 > upper_threshold_b2:
                # Basket2 is overpriced: short basket2 and buy underlying.
                self.b2short = True
                basket2_orders.append(Order('PICNIC_BASKET2', basket2_best_bid, -volume_b2_sell))
                croissant_orders.append(Order('CROISSANTS', croissant_best_ask, 4 * volume_b2_sell))
                jam_orders.append(Order('JAMS', jam_best_ask, 2 * volume_b2_sell))
            elif rolling_spread2 < lower_threshold_b2:
                # Basket2 is underpriced: buy basket2 and sell underlying.
                self.b2long = True
                basket2_orders.append(Order('PICNIC_BASKET2', basket2_best_ask, volume_b2_buy))
                croissant_orders.append(Order('CROISSANTS', croissant_best_bid, -4 * volume_b2_buy))
                jam_orders.append(Order('JAMS', jam_best_bid, -2 * volume_b2_buy))
            else:
                # Neutral signal: unwind any open positions for Basket2 and its underlyings.
                if self.b2long:
                    basket2_orders.append(Order('PICNIC_BASKET2', basket2_best_bid, -current_b2_pos))
                    croissant_orders.append(Order('CROISSANTS', croissant_best_ask, pos_croissant))
                    jam_orders.append(Order('JAMS', jam_best_ask, pos_jam))
                    self.b2long = False
                if self.b2short:
                    basket2_orders.append(Order('PICNIC_BASKET2', basket2_best_ask, current_b2_pos))
                    croissant_orders.append(Order('CROISSANTS', croissant_best_bid, -pos_croissant))
                    jam_orders.append(Order('JAMS', jam_best_bid, -pos_jam))
                    self.b2short = False
        else:
            # Not safe to trade new Basket2 entries; only unwind if needed.
            if self.b2long:
                basket2_orders.append(Order('PICNIC_BASKET2', basket2_best_bid, -current_b2_pos))
                croissant_orders.append(Order('CROISSANTS', croissant_best_ask, pos_croissant))
                jam_orders.append(Order('JAMS', jam_best_ask, pos_jam))
                self.b2long = False
            if self.b2short:
                basket2_orders.append(Order('PICNIC_BASKET2', basket2_best_ask, current_b2_pos))
                croissant_orders.append(Order('CROISSANTS', croissant_best_bid, -pos_croissant))
                jam_orders.append(Order('JAMS', jam_best_bid, -pos_jam))
                self.b2short = False

        #########################################################
        # Section 9: Assemble Final Orders
        #########################################################
        result['CROISSANTS'] = croissant_orders
        result['JAMS'] = jam_orders
        result['DJEMBES'] = djembe_orders
        result['PICNIC_BASKET1'] = basket1_orders
        result['PICNIC_BASKET2'] = basket2_orders

        self.time += 1
        traderData = "PicnicTrader_SGC_v2_HardCutoffs_Unwind_Sequential_NoPosChecks"
        conversions = 0
        
        return result, conversions, traderData
