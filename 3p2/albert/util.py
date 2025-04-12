import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def get_column(df, column_name):
  """
  Gets a single column from a DataFrame by its name.

  Args:
    df (pd.DataFrame): The input pandas DataFrame.
    column_name (str): The name of the column to retrieve.

  Returns:
    pd.Series: The requested column as a pandas Series, or None if the column doesn't exist.
  """
  if not isinstance(df, pd.DataFrame):
      print("Error: Input 'df' must be a pandas DataFrame.")
      return None
  if column_name not in df.columns:
    print(f"Error: Column '{column_name}' not found in the DataFrame.")
    return None
  return df[column_name]

def plot_xy(x_data, y_data, y2_data=None, x_label="X-axis", y_label="Y-axis", y2_label=None, title="Simple Plot", color_threshold=None):
  try:
    # Check for equal length
    if len(x_data) != len(y_data):
      print(f"Error: Input data arrays must have the same length ({len(x_data)} != {len(y_data)}).")
      return
    if y2_data is not None and len(x_data) != len(y2_data):
      print(f"Error: Input data arrays must have the same length ({len(x_data)} != {len(y2_data)}).")
      return
  except TypeError:
    print("Error: Input data must be array-like (e.g., list, pandas Series) that supports len().")
    return

  fig = go.Figure()

  if color_threshold is not None:
    color = ['red' if y > color_threshold else 'blue' for y in y_data]
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=y_label, marker=dict(color=color)))
  else:
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=y_label))

  if y2_data is not None:
    fig.add_trace(go.Scatter(x=x_data, y=y2_data, mode='markers', name=y2_label))

  fig.update_layout(
    title=title,
    xaxis_title=x_label,
    yaxis_title=y_label,
    showlegend=True
  )

  fig.show()

import numpy as np # Needed for handling potential division by zero

def calculate_weighted_mid(df):
  """
  Calculates the weighted mid-price for each row in a DataFrame.

  The weighted mid-price is calculated using the best bid/ask prices and volumes:
  Weighted Mid = (BidPrice1 * AskVolume1 + AskPrice1 * BidVolume1) / (BidVolume1 + AskVolume1)

  Handles cases where the total volume (BidVolume1 + AskVolume1) is zero by returning NaN.

  Args:
    df (pd.DataFrame): Input DataFrame containing market data.
                       Must include columns: 'bid_price_1', 'bid_volume_1',
                       'ask_price_1', 'ask_volume_1'.

  Returns:
    pd.Series: A pandas Series containing the calculated weighted mid-price
               for each row, or None if required columns are missing. Returns NaN
               for rows where total level 1 volume is zero or necessary inputs are NaN.
  """
  required_columns = ['bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1']

  # --- Input Validation ---
  if not isinstance(df, pd.DataFrame):
    print("Error: Input must be a pandas DataFrame.")
    return None

  missing_cols = [col for col in required_columns if col not in df.columns]
  if missing_cols:
    print(f"Error: DataFrame is missing required columns: {missing_cols}")
    return None
  # --- End Validation ---

  # Extract necessary columns (creates copies, safe to modify if needed)
  bid_price = df['bid_price_1']
  bid_volume = df['bid_volume_1']
  ask_price = df['ask_price_1']
  ask_volume = df['ask_volume_1']

  # Calculate numerator and denominator
  # Ensure inputs are numeric, coercing errors to NaN if necessary
  # (This helps if columns were read as object type)
  bid_price = pd.to_numeric(bid_price, errors='coerce')
  bid_volume = pd.to_numeric(bid_volume, errors='coerce')
  ask_price = pd.to_numeric(ask_price, errors='coerce')
  ask_volume = pd.to_numeric(ask_volume, errors='coerce')


  numerator = (bid_price * ask_volume) + (ask_price * bid_volume)
  denominator = bid_volume + ask_volume

  # Calculate weighted mid-price, handling division by zero or NaN denominator
  # np.where(condition, value_if_true, value_if_false)
  weighted_mid = np.where(
      (denominator == 0) | denominator.isna(), # Condition: denominator is zero or NaN
      np.nan,                                  # Value if true: return NaN
      numerator / denominator                  # Value if false: perform calculation
  )

  # Return as a pandas Series
  return pd.Series(weighted_mid, index=df.index)


def plot_xyxy(x1_data, y1_data, x2_data, y2_data=None, x_label="X-axis", y_label="Y-axis", y2_label=None, title="Simple Plot"):
  try:
    # Check for equal length
    if len(x1_data) != len(y1_data):
      print(f"Error: Input data arrays must have the same length ({len(x1_data)} != {len(y1_data)}).")
      return
    if y2_data is not None and len(x2_data) != len(y2_data):
      print(f"Error: Input data arrays must have the same length ({len(x2_data)} != {len(y2_data)}).")
      return
  except TypeError:
    print("Error: Input data must be array-like (e.g., list, pandas Series) that supports len().")
    return

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x1_data, y=y1_data, mode='markers', name=y_label))

  if y2_data is not None:
    fig.add_trace(go.Scatter(x=x2_data, y=y2_data, mode='markers', name=y2_label))

  fig.update_layout(
    title=title,
    xaxis_title=x_label,
    yaxis_title=y_label,
    showlegend=True
  )

  fig.show()

import warnings
def smooth_series_fixed_length(series, num_indexes):
  """
  Smooths a pandas Series by averaging chunks to a fixed number of points.

  Divides the Series into 'num_indexes' approximately equal chunks and
  calculates the mean of each chunk. Returns a new Series of length
  'num_indexes'.

  Args:
    series (pd.Series): The input pandas Series (e.g., from get_column).
                         Should contain numeric data.
    num_indexes (int): The desired number of data points (indexes) in the
                       smoothed output Series. Must be a positive integer
                       less than or equal to the length of the input series.

  Returns:
    pd.Series: A new pandas Series of length 'num_indexes' containing the
               smoothed (averaged) data, or None if an error occurs.
               The index of the returned Series will be a simple RangeIndex.
  """
  # --- Input Validation ---
  if not isinstance(series, pd.Series):
    print("Error: Input 'series' must be a pandas Series.")
    return None

  if not pd.api.types.is_numeric_dtype(series.dtype):
      # Allow object dtype only if all elements *can* be numeric
      # (e.g. object containing ints/floats, possibly with NaNs)
      if series.dtype == 'object':
          try:
              pd.to_numeric(series) # Test conversion
          except (ValueError, TypeError):
              print(f"Error: Series dtype '{series.dtype}' is not numeric and cannot be converted.")
              return None
      else:
            print(f"Error: Series dtype '{series.dtype}' is not numeric.")
            return None

  if not isinstance(num_indexes, int) or num_indexes <= 0:
    print("Error: 'num_indexes' must be a positive integer.")
    return None

  n = len(series)
  if n == 0:
      print("Warning: Input series is empty. Returning an empty Series.")
      return pd.Series([], dtype=series.dtype) # Return empty series of same type

  if num_indexes > n:
    print(f"Error: 'num_indexes' ({num_indexes}) cannot be greater than the series length ({n}).")
    return None

  # --- Smoothing Calculation ---
  # Convert series to numpy array for efficient splitting
  values = series.to_numpy()

  # Split the array into 'num_indexes' chunks. Handles non-even splits.
  # Note: np.array_split might produce empty arrays if num_indexes > n,
  # but we already checked for that.
  chunks = np.array_split(values, num_indexes)

  # Calculate the mean of each chunk.
  # Use np.nanmean to handle potential NaN values within chunks gracefully.
  # If a whole chunk consists of NaNs, nanmean returns NaN.
  smoothed_values = []
  with warnings.catch_warnings():
      # Suppress RuntimeWarning: Mean of empty slice (can happen with NaNs)
      warnings.simplefilter("ignore", category=RuntimeWarning)
      for chunk in chunks:
          # Check if chunk is empty *after* dropping NaNs for nanmean
          # Useful if a chunk *only* contains NaN
          if np.all(np.isnan(chunk)):
               smoothed_values.append(np.nan)
          else:
               smoothed_values.append(np.nanmean(chunk))


  # --- Output ---
  # Create a new Series with the smoothed values and a simple integer index
  smoothed_series = pd.Series(smoothed_values, name=f"{series.name}_smoothed" if series.name else "smoothed")

  return smoothed_series

def plot_histogram_from_series(series, bins='auto', title=None, xlabel=None, ylabel="Frequency", figsize=(10, 6)):
  """
  Plots a histogram of the frequency distribution of values in a pandas Series.

  Args:
    series (pd.Series): The input pandas Series (e.g., from get_column).
                         Should ideally contain numerical data.
    bins (int or str or sequence, optional): The number of bins for the histogram,
        or a strategy string (e.g., 'auto', 'sqrt', 'fd'), or a sequence
        specifying bin edges. Defaults to 'auto'.
    title (str, optional): The title for the plot. Defaults to
        "Histogram of [Series Name]".
    xlabel (str, optional): The label for the x-axis. Defaults to the
        Series name.
    ylabel (str, optional): The label for the y-axis. Defaults to "Frequency".
    figsize (tuple, optional): The figure size (width, height) in inches.
        Defaults to (10, 6).

  Returns:
    matplotlib.axes.Axes: The Axes object containing the plot, or None if an error
                          occurs or the series is unsuitable for a histogram.
  """
  # --- Input Validation ---
  if not isinstance(series, pd.Series):
    print("Error: Input 'series' must be a pandas Series.")
    return None

  if series.empty:
      print("Warning: Input series is empty. Cannot plot histogram.")
      return None

  # Drop NaN values for plotting, warn if many NaNs exist
  original_length = len(series)
  series_cleaned = series.dropna()
  if len(series_cleaned) < original_length:
      print(f"Warning: {original_length - len(series_cleaned)} NaN value(s) were dropped before plotting.")

  if series_cleaned.empty:
      print("Warning: Series contains only NaN values after dropping them. Cannot plot histogram.")
      return None

  # Check if data is numeric - histograms are best for numerical data
  # Allow objects only if they can be coerced to numeric
  is_numeric = pd.api.types.is_numeric_dtype(series_cleaned.dtype)
  can_be_numeric = False
  if series_cleaned.dtype == 'object':
        try:
            pd.to_numeric(series_cleaned)
            can_be_numeric = True
            series_cleaned = pd.to_numeric(series_cleaned) # Convert for plotting
            print("Warning: Series dtype is 'object', converted to numeric for histogram.")
        except (ValueError, TypeError):
            pass # Will be caught by the next check

  if not is_numeric and not can_be_numeric:
      warning_msg = (f"Warning: Series dtype '{series.dtype}' is not numeric. "
                     "A histogram might not be the most appropriate plot. "
                     "Consider using series.value_counts().plot(kind='bar') for categorical data.")
      print(warning_msg)
      # Decide whether to proceed or stop. Let's proceed with a warning for now,
      # as matplotlib might handle some non-numeric types (like Timestamps),
      # but it's less ideal. Alternatively, return None here.

  # --- Plotting ---
  try:
      fig, ax = plt.subplots(figsize=figsize)

      # Plot the histogram
      ax.hist(series_cleaned, bins=bins, edgecolor='black', alpha=0.75)

      # --- Set Labels and Title ---
      # Default title
      if title is None:
          plot_title = f"Histogram of {series.name}" if series.name else "Histogram of Series"
      else:
          plot_title = title
      ax.set_title(plot_title, fontsize=14)

      # Default xlabel
      if xlabel is None:
          plot_xlabel = series.name if series.name else "Values"
      else:
          plot_xlabel = xlabel
      ax.set_xlabel(plot_xlabel, fontsize=12)

      # Set ylabel (default is already set in function args)
      ax.set_ylabel(ylabel, fontsize=12)

      # Add a grid for better readability
      ax.grid(axis='y', linestyle='--', alpha=0.7)

      # Adjust layout
      plt.tight_layout()

      # Display the plot
      plt.show()

      return ax # Return the axes object for further customization if needed

  except Exception as e:
      print(f"An error occurred during plotting: {e}")
      return None
  

def aggregate_trades_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
  """
  Aggregates trades in a DataFrame occurring at the same timestamp.

  Combines rows with identical timestamps by calculating the average
  of their 'price'. Other columns will retain the value from the
  first row within each timestamp group.

  Args:
    df (pd.DataFrame): Input DataFrame assumed to have at least
                       'timestamp' and 'price' columns.

  Returns:
    pd.DataFrame: A new DataFrame with trades aggregated by timestamp,
                  including the calculated average 'price'. Rows are
                  sorted by timestamp implicitly by the groupby operation.

  Raises:
    TypeError: If the input is not a pandas DataFrame.
    ValueError: If the DataFrame does not contain 'timestamp' or 'price' columns.
  """
  if not isinstance(df, pd.DataFrame):
      raise TypeError("Input must be a pandas DataFrame.")
  if 'timestamp' not in df.columns:
      raise ValueError("DataFrame must contain a 'timestamp' column.")
  if 'price' not in df.columns:
      raise ValueError("DataFrame must contain a 'price' column.")

  # --- Aggregation Logic ---
  # Create a dictionary to define how each column should be aggregated.
  # Default to taking the 'first' value for columns other than 'price'.
  agg_dict = {col: 'first' for col in df.columns if col != 'timestamp'}

  # Specifically set 'price' to be averaged ('mean').
  agg_dict['price'] = 'mean'

  # If you have a 'quantity' or 'volume' column you want to sum, add it:
  # if 'quantity' in agg_dict:
  #   agg_dict['quantity'] = 'sum'

  # Group by timestamp and apply the aggregation rules
  aggregated_df = df.groupby('timestamp', as_index=False).agg(agg_dict)
  # Using as_index=False keeps 'timestamp' as a regular column.
  # Alternatively, use .reset_index() after the agg() call.

  return aggregated_df

import pandas as pd

def filter_df_by_timestamps(df_reference: pd.DataFrame,
                            df_to_filter: pd.DataFrame,
                            timestamp_col: str = 'timestamp') -> pd.DataFrame:
  """
  Filters a DataFrame (df_to_filter) based on timestamps present in another
  DataFrame (df_reference).

  Keeps only the rows in df_to_filter where the value in the specified
  timestamp column exists in the timestamp column of df_reference.

  Args:
    df_reference (pd.DataFrame): The DataFrame containing the reference timestamps.
    df_to_filter (pd.DataFrame): The DataFrame to be filtered.
    timestamp_col (str): The name of the column containing timestamps
                         in both DataFrames. Defaults to 'timestamp'.

  Returns:
    pd.DataFrame: A new DataFrame containing only the rows from df_to_filter
                  whose timestamp value exists in df_reference.

  Raises:
    TypeError: If inputs are not pandas DataFrames.
    ValueError: If the timestamp column is not found in either DataFrame.
  """
  if not isinstance(df_reference, pd.DataFrame) or not isinstance(df_to_filter, pd.DataFrame):
      raise TypeError("Both inputs must be pandas DataFrames.")
  if timestamp_col not in df_reference.columns:
      raise ValueError(f"Timestamp column '{timestamp_col}' not found in df_reference (df1).")
  if timestamp_col not in df_to_filter.columns:
      raise ValueError(f"Timestamp column '{timestamp_col}' not found in df_to_filter (df2).")

  # 1. Get the unique timestamps from the reference DataFrame (df1)
  # Using unique() is efficient and handles potential duplicates in df1
  valid_timestamps = df_reference[timestamp_col].unique()

  # 2. Use boolean indexing with `.isin()` to filter the second DataFrame (df2)
  # This checks for each timestamp in df_to_filter if it is present in the valid_timestamps array
  filtered_df = df_to_filter[df_to_filter[timestamp_col].isin(valid_timestamps)].copy()
  # Using .copy() prevents SettingWithCopyWarning if you modify the result later

  # 3. Return the filtered DataFrame
  return filtered_df