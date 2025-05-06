import pandas as pd
import os
import glob
from math import ceil
import numpy as np
import re # Import re for position parsing

# === Wall and Corner Constants (from notebooks) ===
XMIN = -5.5
XMAX = 5.5
ZMIN = -16.5
ZMAX = 4.5
CORNER_THRESHOLD = 1.0 # How close to two walls to be considered a corner

groups = os.listdir("Data/Unity Data")
# Filter out non-directory items or hidden files like .DS_Store if present
groups = [g for g in groups if os.path.isdir(os.path.join("Data/Unity Data", g)) and not g.startswith('.')]
# The original data_files dictionary seems unused later, but keeping structure if needed elsewhere
# data_files = {group : glob.glob(f"Data/Unity Data/{group}/*") for group in groups }

#region : Data Pre-Cleaning
def load_and_clean_data(file_path):
    """Loads and cleans the data from the given file path. Keeps Position."""
    try:
        df = pd.read_csv(file_path)
        # Keep 'Position' but drop others
        cols_to_drop = ["Score", "Penalty", "ID", "RollNumber", "Group"]
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
        # Ensure the first row exists before dropping (often header/metadata)
        if not df.empty and df.index[0] == 0:
             # Check if the first row looks like headers repeated or metadata
             # A simple check could be if EventType is exactly "EventType"
             if "EventType" in df.columns and df.iloc[0]["EventType"] == "EventType":
                df.drop(index=0, inplace=True, errors='ignore')
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty file encountered: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading basic data for {file_path}: {e}")
        return pd.DataFrame()
    return df

def parse_position(pos_str):
    """ Parses a position string like 'x y z' into a tuple of floats. """
    if pd.isna(pos_str) or not isinstance(pos_str, str):
        return (np.nan, np.nan, np.nan)
    parts = pos_str.strip().split() # Split by space
    if len(parts) == 3:
        try:
            # Convert each part to float
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            # Handle cases where parts are not valid numbers
            # print(f"Warning: Could not convert position parts to float: {parts} in string '{pos_str}'")
            return (np.nan, np.nan, np.nan)
    else:
        # Handle cases where split doesn't result in 3 parts
        # print(f"Warning: Could not parse position string (expected 3 parts): {pos_str}")
        return (np.nan, np.nan, np.nan)


def filter_simulation_data(df):
    """Filters the DataFrame to keep only data between the last SimulationStarted and the first SimulationEnded after it."""
    if df.empty:
        return df

    mask_start = df[df["EventType"] == "SimulationStarted"]
    mask_end = df[df["EventType"] == "SimulationEnded"]

    if mask_start.empty:
        print("Warning: SimulationStarted event not found. Cannot reliably filter simulation duration.")
        # Option 1: Return empty - Saftest if start is crucial
        # return pd.DataFrame(columns=df.columns)
        # Option 2: Use the first record's index (less safe)
        start_index = df.index[0]
        print("Using first record index as start.")
    else:
        start_index = mask_start.index[-1] # Use the *last* start event

    if mask_end.empty:
        print("Warning: SimulationEnded event not found.")
        # Option 1: Return empty if end is crucial
        # return pd.DataFrame(columns=df.columns)
         # Option 2: Use the last record's index (less safe)
        end_index = df.index[-1]
        print("Using last record index as end.")
    else:
        # Find the *first* end event *after* the chosen start event
        valid_end_indices = mask_end.index[mask_end.index >= start_index]
        if not valid_end_indices.empty:
            end_index = valid_end_indices[0]
        else:
            # If all ends are before the last start (unlikely but possible), use last record
            print(f"Warning: All SimulationEnded events are before the last SimulationStarted event (index {start_index}). Using last record index.")
            end_index = df.index[-1]

    # Ensure start_index is not greater than end_index (can happen with warnings above)
    if start_index > end_index:
        print(f"Warning: Determined start_index ({start_index}) is after end_index ({end_index}). Returning empty DataFrame.")
        # Decide on handling: return empty or maybe full df? Empty is safer.
        return pd.DataFrame(columns=df.columns)

    return df.loc[start_index:end_index].copy() # Return a copy


def remove_fake_sitting_indications(df):
    """Removes short Entry/Exit UnderTable pairs from the DataFrame."""
    if df.empty or "EventType" not in df.columns or "Time" not in df.columns:
        return df

    entry_under_table = df[df["EventType"] == "EntryUnderTable"]
    exit_under_table = df[df["EventType"] == "ExitUnderTable"]

    # Ensure Time is numeric before proceeding
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    if df['Time'].isnull().any():
         print("Warning: Null values found in 'Time' column during fake sitting removal. Dropping rows with null Time.")
         df.dropna(subset=['Time'], inplace=True)
         # Re-filter after dropping NaNs
         entry_under_table = df[df["EventType"] == "EntryUnderTable"]
         exit_under_table = df[df["EventType"] == "ExitUnderTable"]


    table_interaction = pd.concat([entry_under_table, exit_under_table])
    table_interaction.sort_values(by="Time", inplace=True) # Sort by Time is crucial

    minimum_time_with_table = 500  # milliseconds
    indices_to_drop = []
    i = 0
    while i < len(table_interaction) - 1:
        current_event = table_interaction.iloc[i]
        next_event = table_interaction.iloc[i+1]

        # Ensure we are looking at an Entry followed immediately by an Exit in the sorted list
        if current_event["EventType"] == "EntryUnderTable" and next_event["EventType"] == "ExitUnderTable":
            time_diff = next_event["Time"] - current_event["Time"]
            if time_diff < minimum_time_with_table:
                # Use original DataFrame indices stored in the interaction df
                indices_to_drop.extend([current_event.name, next_event.name])
            # Whether dropped or not, move past this pair
            i += 2
        else:
            # Move to the next event if it's not a valid Entry-Exit pair start
            i += 1

    if indices_to_drop:
        # Use .index access on the original df
        df.drop(index=indices_to_drop, inplace=True, errors='ignore')
    return df


def remove_initial_books_placement(df):
    """Removes BookPlaced events occurring shortly after the last SimulationStarted."""
    if df.empty or "EventType" not in df.columns or "Time" not in df.columns:
        return df

    sim_start_events = df[df["EventType"] == "SimulationStarted"]
    if not sim_start_events.empty:
         # Ensure Time is numeric
         df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
         if df['Time'].isnull().any():
             print("Warning: Null values found in 'Time' column during initial book removal. Dropping rows with null Time.")
             df.dropna(subset=['Time'], inplace=True)
             # Re-filter after dropping NaNs
             sim_start_events = df[df["EventType"] == "SimulationStarted"]
             if sim_start_events.empty: # Check again if dropping removed the start event
                  return df


         filter_time = sim_start_events["Time"].iloc[-1] + 1000 # milliseconds Use last SimulationStarted time
         entry_books = df[df["EventType"] == "BookPlaced"]
         # Ensure we compare times correctly
         rows_to_remove_indices = entry_books[entry_books["Time"] < filter_time].index
         if not rows_to_remove_indices.empty:
             df.drop(index=rows_to_remove_indices, inplace=True, errors='ignore')
    return df

def get_cleaned_data(file_path):
    """Main function to perform data pre-cleaning and parsing."""
    try:
        df = load_and_clean_data(file_path)
        if df is None or df.empty:
            print(f"Warning: Empty DataFrame after loading {file_path}")
            return pd.DataFrame() # Return empty DataFrame explicitly

        # --- Essential Column Checks ---
        if "EventType" not in df.columns or "Time" not in df.columns:
             print(f"Error: Missing essential 'EventType' or 'Time' column in {file_path}. Cannot proceed.")
             return pd.DataFrame()

        # --- Convert Time early ---
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['Time'], inplace=True)
        if len(df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(df)} rows with non-numeric Time in {file_path}")
        if df.empty:
            print(f"Warning: DataFrame empty after dropping rows with invalid Time in {file_path}")
            return df

        df['Position'] = df['EventPosition(X Y Z)']

        # --- Filter Simulation Time ---
        df = filter_simulation_data(df)
        if df.empty:
            print(f"Warning: Empty DataFrame after filtering simulation time {file_path}")
            return df

        # --- Parse Position Data ---
        if 'Position' in df.columns:
            pos_tuples = df['Position'].apply(parse_position)
            df['PosX'] = pos_tuples.apply(lambda x: x[0])
            df['PosY'] = pos_tuples.apply(lambda x: x[1])
            df['PosZ'] = pos_tuples.apply(lambda x: x[2])
            # Optionally drop original Position string column
            # df.drop(columns=['Position'], inplace=True)
            # Check if parsing failed significantly
            if df['PosX'].isnull().sum() > len(df) * 0.5: # Example threshold: >50% failed
                  print(f"Warning: High rate of Position parsing failures in {file_path}. PosX/Y/Z might be unreliable.")
        else:
             print(f"Warning: 'Position' column missing in {file_path}. Near wall/corner stats will be zero.")
             # Add NaN columns so downstream functions don't break
             df['PosX'] = np.nan
             df['PosY'] = np.nan
             df['PosZ'] = np.nan


        # --- Apply Other Cleaning Steps ---
        df = remove_fake_sitting_indications(df)
        df = remove_initial_books_placement(df) # Apply this after time filtering

        # --- Ensure PlayerSeated exists and is boolean-like ---
        if 'PlayerSeated' in df.columns:
             # Handle various representations of boolean
             df['PlayerSeated'] = df['PlayerSeated'].astype(str).str.lower().map(
                 {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False, 'nan': False} # Added more cases
             ).fillna(False) # Default to False if mapping fails or value is missing
        else:
             print(f"Warning: 'PlayerSeated' column missing in {file_path}. Assuming False.")
             df['PlayerSeated'] = False # Add the column defaulted to False


        # --- Final Checks ---
        if df.empty:
            print(f"Warning: DataFrame became empty during cleaning for {file_path}")

    except Exception as e:
        print(f"Error during comprehensive cleaning for {file_path}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return pd.DataFrame() # Return empty DataFrame on error
    return df
#endregion

#region : Common Helper Functions
def _get_earthquake_times(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Extracts the start and end times of the earthquake from the DataFrame."""
    if df.empty or 'Time' not in df.columns or 'EventType' not in df.columns:
        return None, None

    # Ensure Time is numeric
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df.dropna(subset=['Time'], inplace=True) # Drop rows where time isn't valid


    earthquake_start_events = df[df["EventType"] == "EarthquakeStart"].sort_values("Time")
    earthquake_end_events = df[df["EventType"] == "EarthquakeEnd"].sort_values("Time")

    start_time = earthquake_start_events["Time"].iloc[0] if not earthquake_start_events.empty else None
    # Find the first end time *after* the first start time
    end_time = None
    if start_time is not None and not earthquake_end_events.empty:
        valid_end_times = earthquake_end_events[earthquake_end_events["Time"] >= start_time]
        if not valid_end_times.empty:
            end_time = valid_end_times["Time"].iloc[0]

    if start_time is None:
        # print(f"Warning: EarthquakeStart event time not found.") # Reduce verbosity maybe
        pass
    if end_time is None and start_time is not None:
         # print(f"Warning: Corresponding EarthquakeEnd event time not found after start time {start_time}.")
         pass

    # If times found, convert them to float explicitly, otherwise keep as None
    start_time_float = float(start_time) if start_time is not None else None
    end_time_float = float(end_time) if end_time is not None else None

    return start_time_float, end_time_float

def _calculate_duration(start_time: float | pd.Timestamp, end_time: float | pd.Timestamp) -> float:
    """Calculates the duration between two time points (numeric or timestamp)."""
    # Convert Timestamps to numeric (e.g., milliseconds since epoch) if necessary,
    # but the input 'Time' column is already converted to numeric.
    if isinstance(start_time, pd.Timestamp): start_time = start_time.value / 1_000_000 # To milliseconds
    if isinstance(end_time, pd.Timestamp): end_time = end_time.value / 1_000_000 # To milliseconds

    # Ensure they are floats after potential conversion
    start_time = float(start_time)
    end_time = float(end_time)

    return max(0.0, end_time - start_time) # Ensure non-negative

def _calculate_earthquake_overlap_duration(event_start_time: float, event_end_time: float, earthquake_start_time: float | None, earthquake_end_time: float | None) -> float:
    """Calculates the duration of overlap between an event interval and the earthquake interval."""
    if earthquake_start_time is None or earthquake_end_time is None or event_start_time is None or event_end_time is None:
        return 0.0 # No valid earthquake or event interval

    # Ensure earthquake times are ordered
    eq_start = min(earthquake_start_time, earthquake_end_time)
    eq_end = max(earthquake_start_time, earthquake_end_time)

    overlap_start = max(event_start_time, eq_start)
    overlap_end = min(event_end_time, eq_end)

    return max(0.0, overlap_end - overlap_start)

def _calculate_before_earthquake_duration(event_start_time: float, event_end_time: float, earthquake_start_time: float | None) -> float:
    """Calculates the duration of the event period before the earthquake starts."""
    if event_start_time is None or event_end_time is None:
        return 0.0

    if earthquake_start_time is None:
        # If no earthquake start defined, the entire duration is 'before'
        return _calculate_duration(event_start_time, event_end_time)

    effective_end_time = min(event_end_time, earthquake_start_time)
    return max(0.0, effective_end_time - event_start_time)

def _calculate_after_earthquake_duration(event_start_time: float, event_end_time: float, earthquake_end_time: float | None) -> float:
    """Calculates the duration of the event period after the earthquake ends."""
    if event_start_time is None or event_end_time is None or earthquake_end_time is None:
        return 0.0 # If no earthquake end, no duration is 'after'

    effective_start_time = max(event_start_time, earthquake_end_time)
    return max(0.0, event_end_time - effective_start_time)

#endregion

#region : Books Placement
def get_books_placed_stats(df, participant_id, group):
    """Calculates books placed before, during, and after the earthquake using time."""
    if df.empty or 'EventType' not in df.columns or 'Time' not in df.columns:
        return [participant_id, group, 0, 0, 0]

    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    books_placed_events = df[df["EventType"] == "BookPlaced"].copy() # Work on a copy

    if books_placed_events.empty:
        return [participant_id, group, 0, 0, 0]

    # Ensure Time is numeric
    books_placed_events['Time'] = pd.to_numeric(books_placed_events['Time'], errors='coerce')
    books_placed_events.dropna(subset=['Time'], inplace=True)

    if earthquake_start_time is None: # Treat all as 'before' if no earthquake start
        books_before = books_placed_events.shape[0]
        books_during = 0
        books_after = 0
    elif earthquake_end_time is None: # Treat all after start as 'during' if no end
         books_before = books_placed_events[books_placed_events["Time"] < earthquake_start_time].shape[0]
         books_during = books_placed_events[books_placed_events["Time"] >= earthquake_start_time].shape[0]
         books_after = 0
    else: # Normal case with start and end
        books_before = books_placed_events[books_placed_events["Time"] < earthquake_start_time].shape[0]
        books_during = books_placed_events[
            (books_placed_events["Time"] >= earthquake_start_time) &
            (books_placed_events["Time"] <= earthquake_end_time)
        ].shape[0]
        books_after = books_placed_events[books_placed_events["Time"] > earthquake_end_time].shape[0]

    return [participant_id, group, books_before, books_during, books_after]
#endregion

#region : Item Observation
def get_items_observed_stats(df, participant_id, group):
    """Calculates items observed before, during, and after the earthquake using time."""
    if df.empty or 'EventType' not in df.columns or 'Time' not in df.columns:
        return [participant_id, group, 0, 0, 0]

    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    item_observed_events = df[df["EventType"] == "ItemObserved"].copy()

    if item_observed_events.empty:
        return [participant_id, group, 0, 0, 0]

    # Ensure Time is numeric
    item_observed_events['Time'] = pd.to_numeric(item_observed_events['Time'], errors='coerce')
    item_observed_events.dropna(subset=['Time'], inplace=True)


    if earthquake_start_time is None:
        observed_before = item_observed_events.shape[0]
        observed_during = 0
        observed_after = 0
    elif earthquake_end_time is None:
        observed_before = item_observed_events[item_observed_events["Time"] < earthquake_start_time].shape[0]
        observed_during = item_observed_events[item_observed_events["Time"] >= earthquake_start_time].shape[0]
        observed_after = 0
    else:
        observed_before = item_observed_events[item_observed_events["Time"] < earthquake_start_time].shape[0]
        observed_during = item_observed_events[
            (item_observed_events["Time"] >= earthquake_start_time) &
            (item_observed_events["Time"] <= earthquake_end_time)
        ].shape[0]
        observed_after = item_observed_events[item_observed_events["Time"] > earthquake_end_time].shape[0]

    return [participant_id, group, observed_before, observed_during, observed_after]
#endregion

#region : Items picked
def get_items_picked_stats(df, participant_id, group):
    """Calculates items picked before, during, and after an earthquake using time."""
    if df.empty or 'EventType' not in df.columns or 'Time' not in df.columns:
        return [participant_id, group, 0, 0, 0]

    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    items_picked_events = df[df["EventType"] == "ItemPicked"].copy()

    if items_picked_events.empty:
        return [participant_id, group, 0, 0, 0]

    # Ensure Time is numeric
    items_picked_events['Time'] = pd.to_numeric(items_picked_events['Time'], errors='coerce')
    items_picked_events.dropna(subset=['Time'], inplace=True)

    if earthquake_start_time is None:
        picked_before = items_picked_events.shape[0]
        picked_during = 0
        picked_after = 0
    elif earthquake_end_time is None:
        picked_before = items_picked_events[items_picked_events["Time"] < earthquake_start_time].shape[0]
        picked_during = items_picked_events[items_picked_events["Time"] >= earthquake_start_time].shape[0]
        picked_after = 0
    else:
        picked_before = items_picked_events[items_picked_events["Time"] < earthquake_start_time].shape[0]
        picked_during = items_picked_events[
            (items_picked_events["Time"] >= earthquake_start_time) &
            (items_picked_events["Time"] <= earthquake_end_time)
        ].shape[0]
        picked_after = items_picked_events[items_picked_events["Time"] > earthquake_end_time].shape[0]

    return [participant_id, group, picked_before, picked_during, picked_after]
#endregion

#region : Table cover analysis
def _prepare_table_cover_events(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares a DataFrame containing only table cover entry and exit events, sorted by time."""
    if df.empty or "EventType" not in df.columns or "Time" not in df.columns:
         return pd.DataFrame()

    table_cover_taken = df[df["EventType"] == "EntryUnderTable"].copy()
    table_cover_removed = df[df["EventType"] == "ExitUnderTable"].copy()
    df_t = pd.concat([table_cover_taken, table_cover_removed])

    # Ensure Time is numeric before sorting
    df_t['Time'] = pd.to_numeric(df_t['Time'], errors='coerce')
    df_t.dropna(subset=['Time'], inplace=True)

    df_t.sort_values(by="Time", inplace=True) # Sort by Time is crucial
    return df_t

def get_table_cover_stats(df: pd.DataFrame, participant_id: str, group: str) -> list:
    """Calculates statistics related to taking cover under a table."""
    if df.empty or 'Time' not in df.columns or 'EventType' not in df.columns:
        # print(f"Error: DataFrame empty or missing required columns for table cover stats for participant {participant_id} in group {group}")
        return [participant_id, group, 0, 0.0, 0.0, 0.0, 0.0, 0.0] # Match expected stats, use 0.0 for floats

    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    df_t = _prepare_table_cover_events(df)

    if df_t.empty:
        return [participant_id, group, 0, 0.0, 0.0, 0.0, 0.0, 0.0]

    cover_attempts = 0
    total_duration_in_table_cover = 0.0
    total_duration_in_table_cover_before_earthquake = 0.0
    total_duration_in_table_cover_during_earthquake = 0.0
    total_duration_in_table_cover_after_earthquake = 0.0

    i = 0
    while i < len(df_t):
        current_event = df_t.iloc[i]
        # Look for an Entry event
        if current_event["EventType"] == "EntryUnderTable":
            entry_time = current_event["Time"]
            # Find the next event, which should be an Exit for a valid pair
            if i + 1 < len(df_t):
                next_event = df_t.iloc[i + 1]
                if next_event["EventType"] == "ExitUnderTable":
                    exit_time = next_event["Time"]
                    cover_attempts += 1

                    duration_in_table = _calculate_duration(entry_time, exit_time)
                    total_duration_in_table_cover += duration_in_table

                    duration_before = _calculate_before_earthquake_duration(entry_time, exit_time, earthquake_start_time)
                    duration_during = _calculate_earthquake_overlap_duration(entry_time, exit_time, earthquake_start_time, earthquake_end_time)
                    duration_after = _calculate_after_earthquake_duration(entry_time, exit_time, earthquake_end_time)

                    total_duration_in_table_cover_before_earthquake += duration_before
                    total_duration_in_table_cover_during_earthquake += duration_during
                    total_duration_in_table_cover_after_earthquake += duration_after

                    i += 2 # Move past the processed pair
                else:
                    # Entry followed by another Entry or unrelated event - skip this entry
                    # print(f"Warning: Unmatched 'EntryUnderTable' (followed by {next_event['EventType']}) at time {entry_time} for participant {participant_id}")
                    i += 1
            else:
                # Entry event is the last event in the log - ignore
                # print(f"Warning: Unmatched 'EntryUnderTable' at the end of events for participant {participant_id}")
                i += 1 # Move past the last event
        else:
             # Current event is an Exit without a preceding Entry (shouldn't happen with sorting, but safety)
             # print(f"Warning: Unmatched 'ExitUnderTable' at time {current_event['Time']} for participant {participant_id}")
             i += 1 # Move to the next event


    # Calculate averages safely
    average_duration_in_table_cover = (total_duration_in_table_cover / cover_attempts) if cover_attempts > 0 else 0.0

    user_stats = [
        participant_id,
        group,
        cover_attempts,
        average_duration_in_table_cover / 1000.0, # Average total duration per attempt in seconds
        total_duration_in_table_cover / 1000.0, # Grand total duration in seconds
        total_duration_in_table_cover_before_earthquake / 1000.0, # Total before duration in seconds
        total_duration_in_table_cover_during_earthquake / 1000.0, # Total during duration in seconds
        total_duration_in_table_cover_after_earthquake / 1000.0,  # Total after duration in seconds
    ]
    return user_stats
#endregion

#region : Sitting behaviour analysis
def get_seated_stats(df: pd.DataFrame, participant_id: str, group: str) -> list:
    """Calculates statistics related to player sitting behavior."""
    if df.empty or 'Time' not in df.columns or 'PlayerSeated' not in df.columns:
        # print(f"Error: DataFrame empty or missing required columns for seated stats for participant {participant_id} in group {group}")
        # Match the expected number of stats (8)
        return [participant_id, group, 0, 0.0, 0.0, 0.0, 0.0, 0.0]

    try:
        earthquake_start_time, earthquake_end_time = _get_earthquake_times(df.copy()) # Use copy to be safe
        # Handle None return values if necessary by setting non-overlapping bounds
        # Note: _calculate functions already handle None inputs
    except Exception as e:
        print(f"Error getting earthquake times for sitting stats (participant {participant_id} in group {group}): {e}")
        earthquake_start_time = None
        earthquake_end_time = None

    # Ensure Time is numeric and sorted, PlayerSeated is boolean
    df_sorted = df.copy()
    df_sorted['Time'] = pd.to_numeric(df_sorted['Time'], errors='coerce')
    df_sorted.dropna(subset=['Time'], inplace=True)
    # PlayerSeated should be boolean from cleaning, but double-check
    if 'PlayerSeated' in df_sorted.columns and df_sorted['PlayerSeated'].dtype != bool:
         df_sorted['PlayerSeated'] = df_sorted['PlayerSeated'].astype(str).str.lower().map(
             {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False, 'nan': False}
         ).fillna(False)


    df_sorted.sort_values(by='Time', inplace=True)

    if df_sorted.empty:
         return [participant_id, group, 0, 0.0, 0.0, 0.0, 0.0, 0.0]


    seated_transitions = 0 # Count of times player *sat down*
    num_seated_periods = 0 # Count of distinct seated periods (start-end)
    total_seated_duration = 0.0
    total_seated_duration_before_earthquake = 0.0
    total_seated_duration_during_earthquake = 0.0
    total_seated_duration_after_earthquake = 0.0

    current_seated_start_time = None
    last_state = None # Track the previous state

    for index, row in df_sorted.iterrows():
        current_time = row['Time']
        is_seated = row['PlayerSeated']

        # Initialize last_state on the first row
        if last_state is None:
            last_state = is_seated
            if is_seated: # If starting seated
                current_seated_start_time = current_time
                # Don't count this as a transition *to* sitting yet
            continue # Move to next row

        # Detect state change from the previous row
        if is_seated != last_state:
            if is_seated:
                # Transitioned TO seated (False -> True)
                if current_seated_start_time is None: # Should always be None here if logic is right
                    current_seated_start_time = current_time
                    seated_transitions += 1
            else:
                # Transitioned FROM seated (True -> False)
                if current_seated_start_time is not None: # End of a seated period
                    start_of_seated_period = current_seated_start_time
                    end_of_seated_period = current_time # Use current row's time as end

                    duration = _calculate_duration(start_of_seated_period, end_of_seated_period)
                    total_seated_duration += duration
                    num_seated_periods += 1

                    # Calculate durations relative to earthquake
                    duration_before = _calculate_before_earthquake_duration(start_of_seated_period, end_of_seated_period, earthquake_start_time)
                    duration_during = _calculate_earthquake_overlap_duration(start_of_seated_period, end_of_seated_period, earthquake_start_time, earthquake_end_time)
                    duration_after = _calculate_after_earthquake_duration(start_of_seated_period, end_of_seated_period, earthquake_end_time)

                    total_seated_duration_before_earthquake += duration_before
                    total_seated_duration_during_earthquake += duration_during
                    total_seated_duration_after_earthquake += duration_after

                    current_seated_start_time = None # Reset start time

        last_state = is_seated # Update state for the next iteration


    # Handle if seated at the very end of the simulation data
    if current_seated_start_time is not None:
        start_of_seated_period = current_seated_start_time
        # Use the time of the last record as the end of the period
        end_of_seated_period = df_sorted['Time'].iloc[-1]

        duration = _calculate_duration(start_of_seated_period, end_of_seated_period)
        total_seated_duration += duration
        num_seated_periods += 1 # This counts as a period

        # Calculate durations relative to earthquake for the final period
        duration_before = _calculate_before_earthquake_duration(start_of_seated_period, end_of_seated_period, earthquake_start_time)
        duration_during = _calculate_earthquake_overlap_duration(start_of_seated_period, end_of_seated_period, earthquake_start_time, earthquake_end_time)
        duration_after = _calculate_after_earthquake_duration(start_of_seated_period, end_of_seated_period, earthquake_end_time)

        total_seated_duration_before_earthquake += duration_before
        total_seated_duration_during_earthquake += duration_during
        total_seated_duration_after_earthquake += duration_after


    # Calculate averages safely
    average_seated_duration = (total_seated_duration / num_seated_periods) if num_seated_periods > 0 else 0.0

    user_stats = [
        participant_id,
        group,
        seated_transitions, # Number of times sat down
        average_seated_duration / 1000.0, # Average duration per seated period in seconds
        total_seated_duration / 1000.0, # Grand total seated duration in seconds
        total_seated_duration_before_earthquake / 1000.0, # Total before duration in seconds
        total_seated_duration_during_earthquake / 1000.0, # Total during duration in seconds
        total_seated_duration_after_earthquake / 1000.0,  # Total after duration in seconds
    ]
    return user_stats
#endregion

#region : Near Wall Analysis (Adapted from nearWall.ipynb)
def get_near_wall_stats(df: pd.DataFrame, participant_id: str, group: str) -> list:
    """Calculates time spent near walls before, during, and after the earthquake."""
    cols = ["ID", "Group", "TotalTimeNearWall", "TimeNearWallBeforeEarthquake", "TimeNearWallDuringEarthquake", "TimeNearWallAfterEarthquake"]
    default_return = [participant_id, group, 0.0, 0.0, 0.0, 0.0]

    if df.empty or not all(c in df.columns for c in ['Time', 'PosX', 'PosZ']):
        # print(f"Warning: Missing Time or Position data for near wall stats (ID: {participant_id})")
        return default_return

    df_wall = df[['Time', 'PosX', 'PosZ']].copy()
    df_wall.dropna(subset=['Time', 'PosX', 'PosZ'], inplace=True)
    df_wall.sort_values(by='Time', inplace=True)

    if len(df_wall) < 2: # Need at least two points to calculate duration
        return default_return

    # Calculate time difference between consecutive rows
    df_wall['Duration'] = df_wall['Time'].diff()
    # The first duration will be NaN, fill with 0 or decide how to handle the first point's time interval
    df_wall["Duration"] = df_wall['Duration'].fillna(0)

    # --- Determine if near a wall ---
    is_near_xmin = df_wall['PosX'] <= XMIN + CORNER_THRESHOLD # Use corner threshold consistent with notebook? Or define a separate wall threshold? Let's use corner_threshold for now.
    is_near_xmax = df_wall['PosX'] >= XMAX - CORNER_THRESHOLD
    is_near_zmin = df_wall['PosZ'] <= ZMIN + CORNER_THRESHOLD
    is_near_zmax = df_wall['PosZ'] >= ZMAX - CORNER_THRESHOLD
    df_wall['IsNearWall'] = is_near_xmin | is_near_xmax | is_near_zmin | is_near_zmax

    # --- Get Earthquake Times ---
    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df) # Use the original full df

    # --- Calculate time spent near wall in different phases ---
    total_near_wall_duration = 0.0
    before_eq_near_wall_duration = 0.0
    during_eq_near_wall_duration = 0.0
    after_eq_near_wall_duration = 0.0

    # Iterate through the time intervals (Duration represents time *until* the current row's timestamp)
    for i in range(1, len(df_wall)): # Start from the second row as Duration[0] is NaN/0
        # The state (near wall or not) applies to the interval *leading up to* this row's time
        # So we check the state of the *previous* row usually, but the notebook logic sums duration where *current* row is near wall. Let's stick to notebook logic for adaptation.
        is_near = df_wall.iloc[i]['IsNearWall']
        duration = df_wall.iloc[i]['Duration']
        interval_end_time = df_wall.iloc[i]['Time']
        interval_start_time = interval_end_time - duration # Approx start of interval

        if is_near and duration > 0:
             total_near_wall_duration += duration

             # Determine phase based on the interval's *end* time (or midpoint?) - using end time for simplicity
             if earthquake_start_time is None: # No earthquake start
                 before_eq_near_wall_duration += duration
             elif interval_end_time < earthquake_start_time: # Before start
                 before_eq_near_wall_duration += duration
             elif earthquake_end_time is None: # After start, no end
                 during_eq_near_wall_duration += duration
             elif interval_end_time <= earthquake_end_time: # During (inclusive end)
                 during_eq_near_wall_duration += duration
             else: # After end
                 after_eq_near_wall_duration += duration


    # Return stats in seconds
    return [
        participant_id,
        group,
        total_near_wall_duration / 1000.0,
        before_eq_near_wall_duration / 1000.0,
        during_eq_near_wall_duration / 1000.0,
        after_eq_near_wall_duration / 1000.0
    ]

#endregion

#region : Near Corner Analysis (Adapted from nearCorner.ipynb)
def get_near_corner_stats(df: pd.DataFrame, participant_id: str, group: str) -> list:
    """Calculates time spent near corners before, during, and after the earthquake."""
    cols = ["ID", "Group", "TotalTimeNearCorner", "TimeNearCornerBeforeEarthquake", "TimeNearCornerDuringEarthquake", "TimeNearCornerAfterEarthquake"]
    default_return = [participant_id, group, 0.0, 0.0, 0.0, 0.0]

    if df.empty or not all(c in df.columns for c in ['Time', 'PosX', 'PosZ']):
        # print(f"Warning: Missing Time or Position data for near corner stats (ID: {participant_id})")
        return default_return

    df_corner = df[['Time', 'PosX', 'PosZ']].copy()
    df_corner.dropna(subset=['Time', 'PosX', 'PosZ'], inplace=True)
    df_corner.sort_values(by='Time', inplace=True)

    if len(df_corner) < 2: # Need at least two points to calculate duration
        return default_return

    # Calculate time difference between consecutive rows
    df_corner['Duration'] = df_corner['Time'].diff()
    df_corner['Duration'] = df_corner['Duration'].fillna(0)

    # --- Determine if near a corner ---
    near_xmin = df_corner['PosX'] <= XMIN + CORNER_THRESHOLD
    near_xmax = df_corner['PosX'] >= XMAX - CORNER_THRESHOLD
    near_zmin = df_corner['PosZ'] <= ZMIN + CORNER_THRESHOLD
    near_zmax = df_corner['PosZ'] >= ZMAX - CORNER_THRESHOLD

    # Check for combinations of two walls
    is_near_corner_1 = near_xmin & near_zmin # Bottom-left
    is_near_corner_2 = near_xmin & near_zmax # Top-left
    is_near_corner_3 = near_xmax & near_zmin # Bottom-right
    is_near_corner_4 = near_xmax & near_zmax # Top-right

    df_corner['IsNearCorner'] = is_near_corner_1 | is_near_corner_2 | is_near_corner_3 | is_near_corner_4

    # --- Get Earthquake Times ---
    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df) # Use original full df

    # --- Calculate time spent near corner in different phases ---
    total_near_corner_duration = 0.0
    before_eq_near_corner_duration = 0.0
    during_eq_near_corner_duration = 0.0
    after_eq_near_corner_duration = 0.0

    # Iterate through the time intervals
    for i in range(1, len(df_corner)):
        is_near = df_corner.iloc[i]['IsNearCorner']
        duration = df_corner.iloc[i]['Duration']
        interval_end_time = df_corner.iloc[i]['Time']

        if is_near and duration > 0:
            total_near_corner_duration += duration

            # Determine phase based on the interval's end time
            if earthquake_start_time is None:
                before_eq_near_corner_duration += duration
            elif interval_end_time < earthquake_start_time:
                before_eq_near_corner_duration += duration
            elif earthquake_end_time is None:
                 during_eq_near_corner_duration += duration
            elif interval_end_time <= earthquake_end_time:
                 during_eq_near_corner_duration += duration
            else:
                 after_eq_near_corner_duration += duration

    # Return stats in seconds
    return [
        participant_id,
        group,
        total_near_corner_duration / 1000.0,
        before_eq_near_corner_duration / 1000.0,
        during_eq_near_corner_duration / 1000.0,
        after_eq_near_corner_duration / 1000.0
    ]

#endregion


def main():
    # Create Results directory if it doesn't exist
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # Initialize lists for all statistics
    items_picked_stats = []
    books_placed_stats = []
    items_observed_stats = []
    table_cover_stats_list = []
    sitting_behaviour_stats_list = []
    near_wall_stats_list = []       # New list for wall stats
    near_corner_stats_list = []     # New list for corner stats


    all_valid_ids = set() # Keep track of IDs for which we successfully processed data

    for group in groups:
        group_path = os.path.join("Data/Unity Data", group)
        print(f"Processing group: {group} in path: {group_path}")

        # Use glob to find CSV files directly
        files = glob.glob(os.path.join(group_path, "*.csv"))
        if not files:
            print(f"   No CSV files found in {group_path}")
            continue

        for file_path in files:
            # Extract participant ID robustly (handles potential variations)
            base_name = os.path.basename(file_path)
            participant_id = os.path.splitext(base_name)[0] # Removes .csv extension

            print(f"   Processing file: {base_name} (ID: {participant_id})")
            df = get_cleaned_data(file_path)

            if df is None or df.empty:
                print(f"   Skipping {participant_id} due to data cleaning issues or empty data.")
                continue

            # --- Check for essential columns required by *all* analysis functions ---
            # Time, EventType are checked in get_cleaned_data
            # PlayerSeated is added if missing
            # PosX, PosZ are added if Position exists, otherwise filled with NaN
            required_cols = ['Time', 'EventType', 'PlayerSeated', 'PosX', 'PosZ'] # Add PosX/Z
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   Skipping {participant_id} due to missing essential columns after cleaning: {missing_cols}.")
                continue

            # If cleaning succeeded and essential columns are present, add to valid IDs
            all_valid_ids.add((participant_id, group))

            # --- Run all analysis functions ---
            # Use df.copy() if functions might modify df inplace, although current versions try not to
            df_copy = df.copy() # Use a single copy for all functions for this file
            items_picked_stats.append(get_items_picked_stats(df_copy, participant_id, group))
            table_cover_stats_list.append(get_table_cover_stats(df_copy, participant_id, group))
            sitting_behaviour_stats_list.append(get_seated_stats(df_copy, participant_id, group))
            near_wall_stats_list.append(get_near_wall_stats(df_copy, participant_id, group)) # Call new function
            near_corner_stats_list.append(get_near_corner_stats(df_copy, participant_id, group)) # Call new function


            # Conditional analysis based on group
            if group in ['Group 1', 'Group 2']:
                books_placed_stats.append(get_books_placed_stats(df_copy, participant_id, group))
            else: # Assume Groups 3 and 4 (or others) do item observation
                # Append default empty stats for book placing for non-book groups? Or handle merge later. Let merge handle.
                 items_observed_stats.append(get_items_observed_stats(df_copy, participant_id, group))

    # --- Define column names explicitly for all statistics ---
    items_picked_cols = ["ID", "Group", "ItemsPickedBeforeEarthquake", "ItemsPickedDuringEarthquake", "ItemsPickedAfterEarthquake"]
    books_placed_cols = ["ID", "Group", "BooksPlacedBeforeEarthquake", "BooksPlacedDuringEarthquake", "BooksPlacedAfterEarthquake"]
    items_observed_cols = ["ID", "Group", "ItemsObservedBeforeEarthquake", "ItemsObservedDuringEarthquake", "ItemsObservedAfterEarthquake"]
    table_cover_cols = [
        "ID", "Group", "CoverAttempts", "AverageDurationInTableCover", # Renamed for clarity and units
        "TotalDurationInTableCover",                               # Renamed for clarity and units
        "TotalDurationInTableCoverBeforeEarthquake",
        "TotalDurationInTableCoverDuringEarthquake",
        "TotalDurationInTableCoverAfterEarthquake"
    ]
    sitting_behaviour_cols = [
        "ID", "Group", "SittingTransitions", "AverageSeatedDuration",   # Renamed for clarity and units
        "TotalSeatedDuration",                                     # Renamed for clarity and units
        "TotalSeatedDurationBeforeEarthquake",
        "TotalSeatedDurationDuringEarthquake",
        "TotalSeatedDurationAfterEarthquake"
    ]
    near_wall_cols = [ # New column names
         "ID", "Group", "TotalTimeNearWall", "TimeNearWallBeforeEarthquake",
         "TimeNearWallDuringEarthquake", "TimeNearWallAfterEarthquake"
    ]
    near_corner_cols = [ # New column names
         "ID", "Group", "TotalTimeNearCorner", "TimeNearCornerBeforeEarthquake",
         "TimeNearCornerDuringEarthquake", "TimeNearCornerAfterEarthquake"
    ]


    # --- Create DataFrames for all statistics ---
    # Use try-except blocks or check list emptiness before creating DataFrames
    def create_df_safely(data_list, columns, name):
        if not data_list:
             print(f"Warning: No data collected for {name}. Creating empty DataFrame.")
             return pd.DataFrame(columns=columns)
        try:
            df = pd.DataFrame(data_list, columns=columns)
            # Optional: Check if dimensions match
            if df.shape[1] != len(columns):
                 print(f"Error: Column count mismatch for {name}. Expected {len(columns)}, got {df.shape[1]}. Check function return values.")
                 # Decide how to handle: return empty df or raise error?
                 return pd.DataFrame(columns=columns)
            return df
        except ValueError as e:
            print(f"Error creating DataFrame for {name}: {e}. Check function return values.")
            # Print first few elements of list to help debug
            print("First few data rows:", data_list[:2])
            return pd.DataFrame(columns=columns) # Return empty DF on error


    items_picked_df = create_df_safely(items_picked_stats, items_picked_cols, "items_picked")
    books_placed_df = create_df_safely(books_placed_stats, books_placed_cols, "books_placed")
    items_observed_df = create_df_safely(items_observed_stats, items_observed_cols, "items_observed")
    table_cover_df = create_df_safely(table_cover_stats_list, table_cover_cols, "table_cover")
    sitting_behaviour_df = create_df_safely(sitting_behaviour_stats_list, sitting_behaviour_cols, "sitting_behaviour")
    near_wall_df = create_df_safely(near_wall_stats_list, near_wall_cols, "near_wall") # Create new DF
    near_corner_df = create_df_safely(near_corner_stats_list, near_corner_cols, "near_corner") # Create new DF


    # --- Create a base DataFrame with all processed IDs ---
    if all_valid_ids:
        base_df = pd.DataFrame(list(all_valid_ids), columns=["ID", "Group"])
        print(f"Base DataFrame created with {len(base_df)} valid participants.")
    else:
        print("Error: No valid participant data was successfully processed. Final CSV will likely be empty or incomplete.")
        base_df = pd.DataFrame(columns=["ID", "Group"])


    # --- Merge all DataFrames onto the base DataFrame ---
    final = base_df
    # Add the new DFs to the list
    dfs_to_merge = [
        items_picked_df, books_placed_df, items_observed_df, table_cover_df,
        sitting_behaviour_df, near_wall_df, near_corner_df
    ]

    for i, df_merge in enumerate(dfs_to_merge):
        df_name = ["items_picked", "books_placed", "items_observed", "table_cover", "sitting", "near_wall", "near_corner"][i]
        if not df_merge.empty:
            # Check for duplicate columns before merge (excluding keys 'ID', 'Group')
            merge_cols = df_merge.columns.difference(final.columns).tolist()
            key_cols = ['ID', 'Group']
            cols_to_use = key_cols + merge_cols
            if not all(k in df_merge.columns for k in key_cols):
                 print(f"Warning: Key columns ('ID', 'Group') missing in DataFrame {df_name}. Skipping merge.")
                 continue

            # Perform the merge
            final = pd.merge(final, df_merge[cols_to_use], on=["ID", "Group"], how="left")
            print(f"Merged {df_name} data. Final shape: {final.shape}")
        else:
             print(f"Skipping merge for empty DataFrame: {df_name}")


    # --- Add Task and Information columns based on Group ---
    # Ensure 'Group' column exists before applying functions
    if 'Group' in final.columns:
        final["Task"] = final["Group"].apply(lambda x: "Book Task" if x in ["Group 1", "Group 2"] else "No Task")
        final["Information"] = final["Group"].apply(lambda x: "Given" if x in ["Group 1", "Group 3"] else "Not Given")
    else:
        print("Warning: 'Group' column not found in final DataFrame. Cannot add 'Task' and 'Information'.")
        final["Task"] = "Unknown"
        final["Information"] = "Unknown"


    # --- Reorder columns ---
    # Put ID, Group, Task, Information first, then the rest alphabetically? Or specific order?
    # Let's try a specific, logical order
    id_group_cols = ["ID", "Group", "Task", "Information"]
    # Get remaining columns, exclude the ones already listed
    other_cols = [col for col in final.columns if col not in id_group_cols]
    # Sort other columns alphabetically for consistency
    other_cols.sort()

    # Combine the lists
    final_cols_order = id_group_cols + other_cols

    # Ensure all expected columns are present before reordering
    final_cols_order = [col for col in final_cols_order if col in final.columns]
    final = final[final_cols_order]

    # --- Fill NaN values ---
    # NaNs likely occur where a participant exists (in base_df) but had no data for a specific metric (e.g., no book task group has book stats)
    # Filling with 0 is reasonable for counts and durations.
    final.fillna(0, inplace=True)
    print("Filled NaN values with 0.")

    # --- Save the final DataFrame ---
    output_file = "CovertedUnityData_Combined.csv" # New name
    output_path = os.path.join(results_dir, output_file)
    try:
        final.to_csv(output_path, index=False)
        print(f"Successfully saved combined analysis data to {output_path}")
    except Exception as e:
        print(f"Error saving final data to {output_path}: {e}")


if __name__ == "__main__":
    main()