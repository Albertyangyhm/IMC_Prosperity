import pandas as pd
import matplotlib.pyplot as plt
import os

class MarketData:
    def __init__(self, data_prefix, product=None):
        """
        Initialize with a given data directory prefix and an optional product.
        If product is None, all products in the data will be used.
        :param data_prefix: Directory path where the CSV files are located.
        :param product: If provided, filter for this product (e.g. 'KELP'); otherwise, use all products.
        """
        self.data_prefix = data_prefix
        self.product = product  # If None, no filtering is applied.
        self.price_df = None    # Price snapshots DataFrame
        self.trade_df = None    # Trade events DataFrame
        self.merged_df = None   # Merged trade and price DataFrame
        self.product_df = None  # Final DataFrame (across all products) with key columns

    def load_price_data(self, days=[-1, 0, 1]):
        """
        Reads price CSV files for the given days, optionally filters for the specified product,
        computes composite time (in seconds), and calculates the total trading volume.
        """
        price_files = [f'{self.data_prefix}prices_round_2_day_{d}.csv' for d in days]
        self.price_df = pd.concat([
            pd.read_csv(file, sep=';').assign(day=d)
            for file, d in zip(price_files, days)
        ], ignore_index=True)

        # If a single product is specified, filter for it. Otherwise, keep all products.
        if self.product is not None:
            self.price_df = self.price_df[self.price_df['product'] == self.product].copy()

        # Create composite time (assume each day has 1,000,000 timestamp units; convert to seconds)
        self.price_df['composite_time'] = ((self.price_df['day'] + 2) * 1_000_000 +
                                           self.price_df['timestamp']) / 1000
        
    def load_trade_data(self, days=[-1, 0, 1]):
        """
        Reads trade CSV files for the given days, assigns a column 'day' from the loop variable,
        optionally filters for trades with the specified product, and computes composite time (in seconds).
        The column 'symbol' is renamed to 'product' for consistency.
        """
        trade_files = [f'{self.data_prefix}trades_round_2_day_{d}.csv' for d in days]
        self.trade_df = pd.concat([
            pd.read_csv(file, sep=';').assign(day=d)
            for file, d in zip(trade_files, days)
        ], ignore_index=True)

        # Optionally filter for a specific product if provided (the trade file uses 'symbol' for product)
        if self.product is not None:
            self.trade_df = self.trade_df[self.trade_df['symbol'] == self.product].copy()

        # Rename 'symbol' to 'product' for consistency with price_df
        self.trade_df.rename(columns={'symbol': 'product'}, inplace=True)

        # Create composite time for trades (in seconds)
        self.trade_df['composite_time'] = ((self.trade_df['day'] + 2) * 1_000_000 +
                                           self.trade_df['timestamp']) / 1000

    def merge_data(self):
        """
        Merges trade and price data using a merge_asof operation.
        Each trade is matched with the most recent price snapshot within a 1-second tolerance,
        matching on the 'product' as well as 'composite_time'. The final DataFrame is then filtered 
        to select key columns (including profit_and_loss).
        """
        # Both DataFrames are sorted by product and composite_time.
        self.merged_df = pd.merge_asof(
            self.price_df.sort_values(['composite_time']),
            self.trade_df.sort_values(['composite_time']),
            on='composite_time',
            by='product',
            direction='backward',
            tolerance=1  # Matches trades to the latest price snapshot within 1 second
        )

        # Select key columns into self.product_df; note that profit_and_loss is now included.
        self.product_df = self.merged_df[[
            'composite_time', 'product', 'price', 'quantity', 'mid_price',
            'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2',
            'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1',
            'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3']].copy()
        
    def create_synthetic_baskets(self):
        """
        Filters the product_df for the following products:
        PICNIC_BASKET1, CROISSANTS, JAMS, DJEMBE, PICNIC_BASKET2.
        It pivots the DataFrame so that each product's price is aligned by composite_time
        and creates two new columns:
        - SYN_PICNIC_BASKET1 = 6 * CROISSANTS + 3 * JAMS + 1 * PICNIC_BASKET2
        - SYN_PICNIC_BASKET2 = 4 * CROISSANTS + 2 * JAMS
        Returns the pivoted DataFrame with the synthetic basket columns.
        """
        # Define the basket products of interest
        basket_products = ['PICNIC_BASKET1', 'CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET2']
        df_baskets = self.product_df[self.product_df['product'].isin(basket_products)].copy()
        
        # Use pivot_table to aggregate duplicate composite_time entries (aggregating by the mean price)
        baskets_pivot = df_baskets.pivot_table(
            index='composite_time', 
            columns='product', 
            values='mid_price', 
            aggfunc='mean'
        )
        
        # Compute synthetic basket prices using the provided formulas
        if all(col in baskets_pivot.columns for col in ['CROISSANTS', 'JAMS', 'DJEMBES']):
            baskets_pivot['SYN_PICNIC_BASKET1'] = (6 * baskets_pivot['CROISSANTS'] +
                                                3 * baskets_pivot['JAMS'] +
                                                baskets_pivot['DJEMBES'])
        else:
            print("Missing one or more columns needed for SYN_PICNIC_BASKET1")
            
        if all(col in baskets_pivot.columns for col in ['CROISSANTS', 'JAMS']):
            baskets_pivot['SYN_PICNIC_BASKET2'] = (4 * baskets_pivot['CROISSANTS'] +
                                                2 * baskets_pivot['JAMS'])
        else:
            print("Missing one or more columns needed for SYN_PICNIC_BASKET2")
        
        return baskets_pivot


    def plot_synthetic_baskets(self):
        """
        Creates two separate plots:
          1. A plot comparing SYN_PICNIC_BASKET1 and PICNIC_BASKET1.
          2. A plot comparing SYN_PICNIC_BASKET2 and PICNIC_BASKET2.
        Uses composite_time as the x-axis.
        """
        baskets_df = self.create_synthetic_baskets()
        
        # Check if the required columns exist before plotting.
        if 'SYN_PICNIC_BASKET1' in baskets_df.columns and 'PICNIC_BASKET1' in baskets_df.columns:
            plt.figure(figsize=(14, 7))
            plt.plot(baskets_df.index, baskets_df['SYN_PICNIC_BASKET1'], label='SYN_PICNIC_BASKET1', color='blue')
            plt.plot(baskets_df.index, baskets_df['PICNIC_BASKET1'], label='PICNIC_BASKET1', color='red')
            plt.xlabel('Composite Time')
            plt.ylabel('Price')
            plt.title('SYN_PICNIC_BASKET1 vs PICNIC_BASKET1')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("One of the columns (SYN_PICNIC_BASKET1 or PICNIC_BASKET1) is missing in the data.")

        if 'SYN_PICNIC_BASKET2' in baskets_df.columns and 'PICNIC_BASKET2' in baskets_df.columns:
            plt.figure(figsize=(14, 7))
            plt.plot(baskets_df.index, baskets_df['SYN_PICNIC_BASKET2'], label='SYN_PICNIC_BASKET2', color='blue')
            plt.plot(baskets_df.index, baskets_df['PICNIC_BASKET2'], label='PICNIC_BASKET2', color='red')
            plt.xlabel('Composite Time')
            plt.ylabel('Price')
            plt.title('SYN_PICNIC_BASKET2 vs PICNIC_BASKET2')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("One of the columns (SYN_PICNIC_BASKET2 or PICNIC_BASKET2) is missing in the data.")
            
    def plot_synthetic_baskets(self):
        """
        Creates two separate plots:
        1. A plot comparing SYN_PICNIC_BASKET1 and PICNIC_BASKET1.
        2. A plot comparing SYN_PICNIC_BASKET2 and PICNIC_BASKET2.
        Uses composite_time as the x-axis.
        """
        baskets_df = self.create_synthetic_baskets()
        
        # Plot Basket 1 and its Synthetic counterpart
        if 'SYN_PICNIC_BASKET1' in baskets_df.columns and 'PICNIC_BASKET1' in baskets_df.columns:
            plt.figure(figsize=(14, 7))
            plt.plot(baskets_df.index, baskets_df['SYN_PICNIC_BASKET1'], label='SYN_PICNIC_BASKET1', color='blue')
            plt.plot(baskets_df.index, baskets_df['PICNIC_BASKET1'], label='PICNIC_BASKET1', color='red')
            plt.xlabel('Composite Time')
            plt.ylabel('Price')
            plt.title('SYN_PICNIC_BASKET1 vs PICNIC_BASKET1')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("One of the columns (SYN_PICNIC_BASKET1 or PICNIC_BASKET1) is missing in the data.")

        # Plot Basket 2 and its Synthetic counterpart
        if 'SYN_PICNIC_BASKET2' in baskets_df.columns and 'PICNIC_BASKET2' in baskets_df.columns:
            plt.figure(figsize=(14, 7))
            plt.plot(baskets_df.index, baskets_df['SYN_PICNIC_BASKET2'], label='SYN_PICNIC_BASKET2', color='blue')
            plt.plot(baskets_df.index, baskets_df['PICNIC_BASKET2'], label='PICNIC_BASKET2', color='red')
            plt.xlabel('Composite Time')
            plt.ylabel('Price')
            plt.title('SYN_PICNIC_BASKET2 vs PICNIC_BASKET2')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("One of the columns (SYN_PICNIC_BASKET2 or PICNIC_BASKET2) is missing in the data.")

    def plot_basket_differences(self):
        """
        Computes the difference between the actual basket prices and the synthetic basket prices,
        then produces two separate plots:
          1. The difference between PICNIC_BASKET1 and SYN_PICNIC_BASKET1.
          2. The difference between PICNIC_BASKET2 and SYN_PICNIC_BASKET2.
        Uses composite_time as the x-axis and returns the mean difference for both comparisons.
        """
        baskets_df = self.create_synthetic_baskets()
        
        # Initialize variables to hold mean differences
        mean_diff1 = None
        mean_diff2 = None
        
        # Compute and plot differences for Basket 1
        if 'PICNIC_BASKET1' in baskets_df.columns and 'SYN_PICNIC_BASKET1' in baskets_df.columns:
            diff_basket1 = baskets_df['PICNIC_BASKET1'] - baskets_df['SYN_PICNIC_BASKET1']
            mean_diff1 = abs(diff_basket1).mean()
            plt.figure(figsize=(14, 7))
            plt.plot(baskets_df.index, diff_basket1, label="Difference: PICNIC_BASKET1 - SYN_PICNIC_BASKET1", color='purple')
            plt.xlabel('Composite Time')
            plt.ylabel('Price Difference')
            plt.title('Difference between PICNIC_BASKET1 and SYN_PICNIC_BASKET1')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("Missing columns for Basket 1 difference computation.")

        # Compute and plot differences for Basket 2
        if 'PICNIC_BASKET2' in baskets_df.columns and 'SYN_PICNIC_BASKET2' in baskets_df.columns:
            diff_basket2 = baskets_df['PICNIC_BASKET2'] - baskets_df['SYN_PICNIC_BASKET2']
            mean_diff2 = abs(diff_basket2).mean()
            plt.figure(figsize=(14, 7))
            plt.plot(baskets_df.index, diff_basket2, label="Difference: PICNIC_BASKET2 - SYN_PICNIC_BASKET2", color='green')
            plt.xlabel('Composite Time')
            plt.ylabel('Price Difference')
            plt.title('Difference between PICNIC_BASKET2 and SYN_PICNIC_BASKET2')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("Missing columns for Basket 2 difference computation.")
            
        return mean_diff1, mean_diff2


# ------------------- MAIN DRIVER -------------------
if __name__ == '__main__':
    # Define your data directory prefix
    data_prefix = '/Users/halamadrid/Desktop/IMC Prosperity/imc-prosperity-3/round-2-island-data-bottle/'

    # Create an instance. Set product to None to use all products.
    market_data = MarketData(data_prefix, product=None)

    # Load price and trade data; merge them (the merged DataFrame now includes profit_and_loss)
    market_data.load_price_data()
    market_data.load_trade_data()
    market_data.merge_data()

    # Create synthetic basket columns and produce the plots.
    # This will filter the five basket products and compute the synthetic baskets,
    # then plot:
    #   (1) SYN_PICNIC_BASKET1 vs PICNIC_BASKET1, and
    #   (2) SYN_PICNIC_BASKET2 vs PICNIC_BASKET2.
    market_data.plot_synthetic_baskets()
    
    # Plot differences between actual and synthetic baskets and capture the mean differences.
    mean_diff_basket1, mean_diff_basket2 = market_data.plot_basket_differences()
    print("Mean difference for Basket 1 (PICNIC_BASKET1 - SYN_PICNIC_BASKET1):", mean_diff_basket1)
    print("Mean difference for Basket 2 (PICNIC_BASKET2 - SYN_PICNIC_BASKET2):", mean_diff_basket2)