from typing import Dict, List
import numpy as np
import math
from math import log, sqrt, exp
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        # For diagnostics only.
        self.stateHistory: List[TradingState] = []
        # Rolling list of underlying mid-prices for realized volatility estimation.
        self.underlying_mid_prices: List[float] = []
        # If order depth for underlying is incomplete, fallback to last known mid-price.
        self.last_mid_price: float = None
        self.time = 0  # Tick counter.
        
        # Field to track the unhedged net delta (accumulated over time).
        self.net_option_delta = 0.0
        
        # Time conversion constants.
        self.TICKS_PER_DAY = 10000.0
        self.TRADING_DAYS_PER_YEAR = 252.0
        self.TICKS_PER_YEAR = self.TICKS_PER_DAY * self.TRADING_DAYS_PER_YEAR  # 2,520,000 ticks/year
        
        # Black–Scholes parameters.
        self.r = 0.00           # risk-free rate.
        self.default_sigma = 0.1  # default annual volatility.
        
        # Rolling window length for estimating volatility.
        self.realized_vol_window = 200
        
        # Define available call options (vouchers) with strikes.
        self.voucher_strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        
        # Position limits.
        self.pos_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        
        # Market making parameters.
        self.base_order_size = 10
        self.spread_offset = 5.0  # fixed half-spread around the fair value.
        # Optional: widen spread if inventory is high.
        self.inventory_spread_factor = 0.02
        
    def updateProductHist(self, productName, state, ifPop):
        self.stateHistory.append(state)
        startDay = 0 if not ifPop else len(self.stateHistory) - 6
        order_depth_hist = [i.order_depths[productName] for i in self.stateHistory[startDay:-1]]
        sell_orders_hist = [i.sell_orders for i in order_depth_hist]
        buy_orders_hist = [i.buy_orders for i in order_depth_hist]

        best_ask_hist = [[min(orders.keys()), orders[min(orders.keys())]] for orders in sell_orders_hist]
        best_bid_hist = [[max(orders.keys()), orders[max(orders.keys())]] for orders in buy_orders_hist]

        best_ask_vol_hist = [i[1] for i in best_ask_hist]
        best_bid_vol_hist = [i[1] for i in best_bid_hist]
        best_ask_hist = [i[0] for i in best_bid_hist]
        best_bid_hist = [i[0] for i in best_bid_hist]

        return best_ask_hist, best_ask_vol_hist, best_bid_hist, best_bid_vol_hist

    def norm_cdf(self, x: float) -> float:
        """Custom implementation of the standard normal CDF using math.erf."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
    
    def black_scholes_call_and_delta(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Compute the Black–Scholes call price and delta.
        For T<=0, returns intrinsic value and an immediate exercise delta.
        """
        if T <= 0:
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
            return price, delta
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        price = S * self.norm_cdf(d1) - K * exp(-r * T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        return price, delta
    
    def compute_realized_vol(self) -> float:
        """
        Estimate the annualized realized volatility from tick-level log returns of the
        underlying mid-prices over the last 'realized_vol_window' samples.
        Uses TICKS_PER_YEAR for annualization.
        """
        if len(self.underlying_mid_prices) < 2:
            return self.default_sigma
        
        # Use the most recent realized_vol_window prices.
        prices = self.underlying_mid_prices[-self.realized_vol_window:]
        if len(prices) < 2:
            return self.default_sigma
        
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                log_returns.append(math.log(prices[i] / prices[i - 1]))
        
        if len(log_returns) < 2:
            return self.default_sigma
        
        sample_std = np.std(log_returns, ddof=1)
        # Annualize: each tick represents 1/TICKS_PER_DAY of a day.
        realized_vol_annual = sample_std * math.sqrt(self.TICKS_PER_YEAR)
        return max(realized_vol_annual, 0.01)  # impose a minimum vol if needed
    
    def run(self, state: TradingState):
        """
        Delta-neutral market making strategy:
          1. Compute or update the underlying mid-price.
          2. Estimate the annualized volatility from recent tick data.
          3. For each option, compute fair value using Black–Scholes and quote bid/ask with a fixed
             plus inventory-adjusted spread. Sizing is capped by both your base order size and the
             maximum volume available at the top of the corresponding order book.
          4. Hedge net option delta by trading the underlying. If available volume prevents full hedging,
             update self.net_option_delta with the residual unhedged delta to carry forward.
          
        Time-to-maturity T is computed as:
            T = (days_remaining) / TRADING_DAYS_PER_YEAR
        where days_remaining = 7 - (self.time / TICKS_PER_DAY).
        """
        result: Dict[str, List[Order]] = {}
        traderData = "DeltaNeutralMarketMaker_v3"
        conversions = 0
        
        # 1. Update state.
        self.stateHistory.append(state)
        
        # --- Underlying Price Computation ---
        underlying = "VOLCANIC_ROCK"
        if (underlying not in state.order_depths or
            not state.order_depths[underlying].buy_orders or
            not state.order_depths[underlying].sell_orders):
            if self.last_mid_price is not None:
                S = self.last_mid_price
                print(f"Incomplete order depth for {underlying}, using last mid price: {S:.2f}")
            else:
                S = 10000.0
                print(f"Incomplete order depth for {underlying}, fallback S = {S:.2f}")
        else:
            order_depth_under = state.order_depths[underlying]
            best_bid_under = max(order_depth_under.buy_orders.keys())
            best_ask_under = min(order_depth_under.sell_orders.keys())
            S = (best_bid_under + best_ask_under) / 2.0
            self.last_mid_price = S
        
        # Update rolling mid-price history.
        self.underlying_mid_prices.append(S)
        
        # 2. Estimate realized (annualized) volatility from tick data.
        sigma_est = self.compute_realized_vol()
        
        # 3. Quote for each call option (voucher).
        voucher_products = list(self.voucher_strikes.keys())
        for voucher in voucher_products:
            voucher_orders: List[Order] = []
            strike = self.voucher_strikes[voucher]
            pos_voucher = state.position.get(voucher, 0)
            limit_voucher = self.pos_limits[voucher]
            
            if (voucher not in state.order_depths or
                not state.order_depths[voucher].buy_orders or
                not state.order_depths[voucher].sell_orders):
                print(f"Incomplete order depth for {voucher}")
                result[voucher] = voucher_orders
                continue
            
            order_depth_voucher: OrderDepth = state.order_depths[voucher]
            # Compute time to expiry.
            days_remaining = 7.0 - (self.time / self.TICKS_PER_DAY)
            if days_remaining <= 0:
                days_remaining = 0.0001  # avoid zero expiry.
            T = days_remaining / self.TRADING_DAYS_PER_YEAR  # time to expiry in years.
            
            # Compute fair value and delta using Black–Scholes with sigma_est.
            theo_price, _ = self.black_scholes_call_and_delta(S, strike, T, self.r, sigma_est)
            
            # Inventory-adjusted half-spread.
            inv_skew = 1.0 + self.inventory_spread_factor * (abs(pos_voucher) / float(limit_voucher))
            half_spread = self.spread_offset * inv_skew
            
            # Construct bid/ask quotes.
            bid_price = int(theo_price - half_spread)
            ask_price = int(theo_price + half_spread)
            
            # --- Check available volume on the order book ---
            best_ask_voucher = min(order_depth_voucher.sell_orders.keys())
            available_to_buy = abs(order_depth_voucher.sell_orders[best_ask_voucher])
            best_bid_voucher = max(order_depth_voucher.buy_orders.keys())
            available_to_sell = order_depth_voucher.buy_orders[best_bid_voucher]
            
            order_size_buy = min(self.base_order_size, available_to_buy)
            order_size_sell = min(self.base_order_size, available_to_sell)
            
            print(f"[{voucher}] exchange_best_bid: {best_bid_voucher} ({available_to_sell}), exchange_best_ask: {best_ask_voucher} ({available_to_buy})")
            print()
    
            if pos_voucher + order_size_buy <= limit_voucher and order_size_buy > 0:
                voucher_orders.append(Order(voucher, bid_price, order_size_buy))
                print(f"[{voucher}] BID {order_size_buy} @ {bid_price:.2f} (theo={theo_price:.2f}, sigma={sigma_est:.3f}, T={T:.5f})")
                print()
            if pos_voucher - order_size_sell >= -limit_voucher and order_size_sell > 0:
                voucher_orders.append(Order(voucher, ask_price, -order_size_sell))
                print(f"[{voucher}] ASK {order_size_sell} @ {ask_price:.2f} (theo={theo_price:.2f}, sigma={sigma_est:.3f}, T={T:.5f})")
                print()
            
            result[voucher] = voucher_orders
        
        # 4. Delta hedging with the underlying.
        # When computing hedge, we want to include the new orders we just sent in the current tick.
        # New orders are not settled in state.position. So, for each voucher, add the net new quantity from result.
        computed_option_delta = 0.0
        # Use a minimum T for hedging as well.
        T_for_hedge = max((7.0 - (self.time / self.TICKS_PER_DAY)) / self.TRADING_DAYS_PER_YEAR, 1.0 / self.TRADING_DAYS_PER_YEAR)
        for voucher in voucher_products:
            new_orders = result.get(voucher, [])
            new_voucher_pos = sum(order.quantity for order in new_orders)
            strike = self.voucher_strikes[voucher]
            _, delta = self.black_scholes_call_and_delta(S, strike, T_for_hedge, self.r, sigma_est)
            computed_option_delta += new_voucher_pos * delta
        
        underlying_pos = state.position.get(underlying, 0)
        # Combine the underlying's settled position, computed option delta (including new orders),
        # and any residual unhedged delta from previous ticks.
        total_net_delta = computed_option_delta + self.net_option_delta
        desired_hedge = - total_net_delta  # This is the hedge we want to execute.
        
        underlying_orders: List[Order] = []
        limit_under = self.pos_limits[underlying]
        
        executed_hedge = 0  # Track the actual hedge executed.
        
        if abs(desired_hedge) >= 1:
            if (underlying not in state.order_depths or
                not state.order_depths[underlying].buy_orders or
                not state.order_depths[underlying].sell_orders):
                print("Incomplete order depth for underlying hedge")
            else:
                order_depth_under = state.order_depths[underlying]
                best_bid_under = max(order_depth_under.buy_orders.keys())
                best_ask_under = min(order_depth_under.sell_orders.keys())
                if desired_hedge > 0:
                    # Need to buy underlying.
                    available_to_buy = limit_under - underlying_pos
                    hedge_qty = min(int(round(desired_hedge)), available_to_buy)
                    if hedge_qty > 0:
                        underlying_orders.append(Order(underlying, best_ask_under, hedge_qty))
                        executed_hedge = hedge_qty  # Positive hedge order.
                        print(f"[HEDGE] BUY {hedge_qty} {underlying} @ {best_ask_under}")
                        print()
                else:
                    # Need to sell underlying.
                    available_to_sell = limit_under + underlying_pos
                    hedge_qty = min(int(round(abs(desired_hedge))), available_to_sell)
                    if hedge_qty > 0:
                        underlying_orders.append(Order(underlying, best_bid_under, -hedge_qty))
                        executed_hedge = -hedge_qty  # Negative hedge order.
                        print(f"[HEDGE] SELL {hedge_qty} {underlying} @ {best_bid_under}")
                        print()
        result[underlying] = underlying_orders
        
        self.net_option_delta = total_net_delta + executed_hedge
        
        print(f"Net delta after hedging: {self.net_option_delta:.2f} (executed hedge: {executed_hedge})")
        print()
        
        # Increment tick counter.
        self.time += 1
        return result, conversions, traderData
