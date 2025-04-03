import pandas as pd
import os
import glob
from math import ceil

groups = os.listdir("Data/Unity Data")
data_files = {group : glob.glob(f"Data/Unity Data/{group}/*") for group in groups }

#region : Data Pre-Cleaning
def load_and_clean_data(file_path):
    """Loads and cleans the data from the given file path."""
    df = pd.read_csv(file_path)
    df.drop(columns=["Score", "Penalty", "ID", "RollNumber", "Group"], inplace=True, errors='ignore') # Added errors='ignore'
    # Ensure the first row exists before dropping
    if not df.empty:
        df.drop(index=0, inplace=True, errors='ignore') # Added errors='ignore'
    return df

def filter_simulation_data(df):
    """Filters out the simulation data from the DataFrame."""
    mask_start = df[df["EventType"] == "SimulationStarted"]
    mask_end = df[df["EventType"] == "SimulationEnded"]

    # Handle cases where SimulationStarted or SimulationEnded might be missing
    if mask_start.empty:
        start_index = df.index[0] if not df.empty else 0
    else:
        start_index = mask_start.index[-1]

    if mask_end.empty:
        end_index = df.index[-1] if not df.empty else start_index
    else:
        # Ensure end_index is not before start_index
        valid_end_indices = mask_end.index[mask_end.index >= start_index]
        if not valid_end_indices.empty:
            end_index = valid_end_indices[0]
        else: # If all SimulationEnded are before SimulationStarted, take the last row
            end_index = df.index[-1] if not df.empty else start_index

    # Ensure start_index is not greater than end_index
    if start_index > end_index and not df.empty:
       print(f"Warning: SimulationStarted index ({start_index}) is after SimulationEnded index ({end_index}). Taking full range.")
       start_index = df.index[0]
       end_index = df.index[-1]
    elif df.empty:
       return df # Return empty df if input is empty

    return df.loc[start_index:end_index]


def remove_fake_sitting_indications(df):
    """Removes fake sitting indications from the DataFrame."""
    entry_under_table = df[df["EventType"] == "EntryUnderTable"]
    exit_under_table = df[df["EventType"] == "ExitUnderTable"]
    table_interaction = pd.concat([entry_under_table, exit_under_table])
    table_interaction.sort_values(by="Time", inplace=True)
    minimum_time_with_table = 500  # milliseconds

    indices_to_drop = []
    i = 0
    while i < len(table_interaction) - 1:
        # Ensure we are looking at an Entry followed by an Exit
        if table_interaction.iloc[i]["EventType"] == "EntryUnderTable" and table_interaction.iloc[i+1]["EventType"] == "ExitUnderTable":
            if table_interaction.iloc[i + 1]["Time"] - table_interaction.iloc[i]["Time"] < minimum_time_with_table:
                indices_to_drop.extend([table_interaction.index[i], table_interaction.index[i + 1]])
            i += 2 # Move to the next potential pair
        else:
            # Skip the current event if it's not part of a valid Entry-Exit pair start
            i += 1

    if indices_to_drop:
        df.drop(index=indices_to_drop, inplace=True, errors='ignore') # Added errors='ignore'
    return df

def remove_initial_books_placement(df):
    """Removes initial books placement from the DataFrame."""
    sim_start_events = df[df["EventType"] == "SimulationStarted"]
    if not sim_start_events.empty:
        filter_time = sim_start_events["Time"].values[-1] + 1000 # milliseconds Use last SimulationStarted
        entry_books = df[df["EventType"] == "BookPlaced"]
        rows_to_remove = entry_books[entry_books["Time"] < filter_time]
        if not rows_to_remove.empty:
            df.drop(index=rows_to_remove.index, inplace=True, errors='ignore') # Added errors='ignore'
    return df

def get_cleaned_data(file_path):
    """Main function to perform data pre-cleaning."""
    try:
        df = load_and_clean_data(file_path)
        if df.empty:
             print(f"Warning: Empty DataFrame after loading {file_path}")
             return df
        df = filter_simulation_data(df)
        if df.empty:
             print(f"Warning: Empty DataFrame after filtering simulation {file_path}")
             return df
        df = remove_fake_sitting_indications(df)
        df = remove_initial_books_placement(df)
        # Convert Time column to numeric, coercing errors
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        # Drop rows where Time could not be converted
        df.dropna(subset=['Time'], inplace=True)
        # Ensure PlayerSeated exists and is boolean-like
        if 'PlayerSeated' in df.columns:
             df['PlayerSeated'] = df['PlayerSeated'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False)
        else:
             print(f"Warning: 'PlayerSeated' column missing in {file_path}. Assuming False.")
             df['PlayerSeated'] = False

    except Exception as e:
        print(f"Error cleaning data for {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    return df
#endregion

#region : Books Placement
def get_books_placed_stats(df, id, group):
    if df.empty or 'EventType' not in df.columns:
         return [id, group, 0, 0, 0]
    earthquake_start_events = df[df["EventType"] == "EarthquakeStart"]
    earthquake_end_events = df[df["EventType"] == "EarthquakeEnd"]

    if earthquake_start_events.empty or earthquake_end_events.empty:
        print(f"Warning: Missing earthquake events for {id} in {group}. Cannot calculate timed book placements.")
        total_books = df[df["EventType"] == "BookPlaced"].shape[0]
        return [id, group, total_books, 0, 0] # Attribute all books to 'before' if events missing

    earthquake_start_index = earthquake_start_events.index[0]
    earthquake_end_index = earthquake_end_events.index[-1]

    # Ensure indices are valid and in order
    if earthquake_start_index > earthquake_end_index:
       print(f"Warning: Earthquake start index after end index for {id} in {group}. Using full range for book placements.")
       earthquake_start_index = df.index.min()
       earthquake_end_index = df.index.max()


    before_earthquake_data = df.loc[:earthquake_start_index] # Inclusive of start event row itself? Check logic. Usually exclusive.
    # Let's make it exclusive for 'before' and inclusive start/end for 'during'
    before_earthquake_data = df.loc[df.index < earthquake_start_index]
    during_earthquake_data = df.loc[earthquake_start_index:earthquake_end_index]
    after_earthquake_data = df.loc[df.index > earthquake_end_index]


    # Calculate number of books placed before, during and after earthquake
    books_before_earthquake = before_earthquake_data[before_earthquake_data["EventType"] == "BookPlaced"].shape[0]
    books_during_earthquake = during_earthquake_data[during_earthquake_data["EventType"] == "BookPlaced"].shape[0]
    books_after_earthquake = after_earthquake_data[after_earthquake_data["EventType"] == "BookPlaced"].shape[0]

    return [id, group, books_before_earthquake, books_during_earthquake, books_after_earthquake]
#endregion

#region : Item Observation
def get_items_observed_stats(df, id, group):
    if df.empty or 'EventType' not in df.columns:
        return [id, group, 0, 0, 0]
    earthquake_start_events = df[df["EventType"] == "EarthquakeStart"]
    earthquake_end_events = df[df["EventType"] == "EarthquakeEnd"]

    if earthquake_start_events.empty or earthquake_end_events.empty:
        print(f"Warning: Missing earthquake events for {id} in {group}. Cannot calculate timed item observations.")
        total_observed = df[df["EventType"] == "ItemObserved"].shape[0]
        return [id, group, total_observed, 0, 0] # Attribute all to 'before'

    earthquake_start_index = earthquake_start_events.index[0]
    earthquake_end_index = earthquake_end_events.index[-1]

    if earthquake_start_index > earthquake_end_index:
       print(f"Warning: Earthquake start index after end index for {id} in {group}. Using full range for item observations.")
       earthquake_start_index = df.index.min()
       earthquake_end_index = df.index.max()

    before_earthquake_data = df.loc[df.index < earthquake_start_index]
    during_earthquake_data = df.loc[earthquake_start_index:earthquake_end_index]
    after_earthquake_data = df.loc[df.index > earthquake_end_index]

    # Calcaulate number of items observed before, during and after earthquake
    items_observed_before_earthquake = before_earthquake_data[before_earthquake_data["EventType"] == "ItemObserved"].shape[0]
    items_observed_during_earthquake = during_earthquake_data[during_earthquake_data["EventType"] == "ItemObserved"].shape[0]
    items_observed_after_earthquake = after_earthquake_data[after_earthquake_data["EventType"] == "ItemObserved"].shape[0]

    return [id, group, items_observed_before_earthquake, items_observed_during_earthquake, items_observed_after_earthquake]
#endregion

#region : Items picked
def get_items_picked_stats(df, id, group):
    """Calculates the number of items picked before, during, and after an earthquake."""
    if df.empty or 'EventType' not in df.columns:
        return [id, group, 0, 0, 0]
    earthquake_start_events = df[df["EventType"] == "EarthquakeStart"]
    earthquake_end_events = df[df["EventType"] == "EarthquakeEnd"]

    if earthquake_start_events.empty or earthquake_end_events.empty:
        print(f"Warning: Missing earthquake events for {id} in {group}. Cannot calculate timed item picks.")
        total_picked = df[df["EventType"] == "ItemPicked"].shape[0]
        return [id, group, total_picked, 0, 0] # Attribute all to 'before'

    earthquake_start_index = earthquake_start_events.index[0]
    earthquake_end_index = earthquake_end_events.index[-1]

    if earthquake_start_index > earthquake_end_index:
       print(f"Warning: Earthquake start index after end index for {id} in {group}. Using full range for item picks.")
       earthquake_start_index = df.index.min()
       earthquake_end_index = df.index.max()


    before_earthquake_data = df.loc[df.index < earthquake_start_index]
    during_earthquake_data = df.loc[earthquake_start_index:earthquake_end_index]
    after_earthquake_data = df.loc[df.index > earthquake_end_index]

    items_picked_before_earthquake = before_earthquake_data[before_earthquake_data["EventType"] == "ItemPicked"].shape[0]
    items_picked_during_earthquake = during_earthquake_data[during_earthquake_data["EventType"] == "ItemPicked"].shape[0]
    items_picked_after_earthquake = after_earthquake_data[after_earthquake_data["EventType"] == "ItemPicked"].shape[0]

    return [id, group, items_picked_before_earthquake, items_picked_during_earthquake, items_picked_after_earthquake]
#endregion

#region : Table cover analysis
def _get_earthquake_times(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Extracts the start and end times of the earthquake from the DataFrame."""
    earthquake_start_events = df[df["EventType"] == "EarthquakeStart"]
    earthquake_end_events = df[df["EventType"] == "EarthquakeEnd"]

    start_time = earthquake_start_events.iloc[0]["Time"] if not earthquake_start_events.empty else None
    end_time = earthquake_end_events.iloc[-1]["Time"] if not earthquake_end_events.empty else None

    # Handle case where end time might be before start time if multiple earthquakes occurred
    if start_time is not None and end_time is not None and end_time < start_time:
         # Find the first end time that is after the first start time
         valid_end_times = earthquake_end_events[earthquake_end_events["Time"] >= start_time]
         end_time = valid_end_times.iloc[0]["Time"] if not valid_end_times.empty else None


    if start_time is None or end_time is None:
         print(f"Warning: EarthquakeStart or EarthquakeEnd event time not found or invalid.")
         # Decide how to handle this - maybe use simulation start/end? For now, return None.
         # sim_start = df[df["EventType"] == "SimulationStarted"].iloc[-1]["Time"] if not df[df["EventType"] == "SimulationStarted"].empty else df['Time'].min()
         # sim_end = df[df["EventType"] == "SimulationEnded"].iloc[0]["Time"] if not df[df["EventType"] == "SimulationEnded"].empty else df['Time'].max()
         # return sim_start, sim_end # Alternative: use simulation boundaries
         return None, None # Indicate failure

    return start_time, end_time


def _prepare_table_cover_events(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares a DataFrame containing only table cover entry and exit events, sorted by index."""
    table_cover_taken = df[df["EventType"] == "EntryUnderTable"]
    table_cover_removed = df[df["EventType"] == "ExitUnderTable"]
    df_t = pd.concat([table_cover_taken, table_cover_removed])
    # Sort by time first, then index to handle simultaneous events if any
    df_t.sort_values(by=["Time"], inplace=True)
    return df_t

def _calculate_duration(start_time: float, end_time: float) -> float:
    """Calculates the duration between two time points."""
    return max(0, end_time - start_time) # Ensure non-negative

def _calculate_earthquake_overlap_duration(entry_time: float, exit_time: float, earthquake_start_time: float | None, earthquake_end_time: float | None) -> float:
    """Calculates the duration of overlap between a table cover event and the earthquake."""
    if earthquake_start_time is None or earthquake_end_time is None:
        return 0 # No earthquake defined, so no overlap
    return max(0, min(exit_time, earthquake_end_time) - max(entry_time, earthquake_start_time))

def _calculate_before_earthquake_duration(entry_time: float, exit_time: float, earthquake_start_time: float | None) -> float:
    """Calculates the duration of the cover period before the earthquake starts."""
    if earthquake_start_time is None:
        return _calculate_duration(entry_time, exit_time) # If no earthquake start, all duration is 'before'
    return max(0, min(exit_time, earthquake_start_time) - entry_time)

def _calculate_after_earthquake_duration(entry_time: float, exit_time: float, earthquake_end_time: float | None) -> float:
    """Calculates the duration of the cover period after the earthquake ends."""
    if earthquake_end_time is None:
        return 0 # If no earthquake end, no duration is 'after'
    return max(0, exit_time - max(entry_time, earthquake_end_time))


def get_table_cover_stats(df: pd.DataFrame, participant_id: str, group: str) -> list:
    """Calculates statistics related to taking cover under a table."""
    if df.empty or 'Time' not in df.columns or 'EventType' not in df.columns:
         print(f"Error: DataFrame empty or missing required columns for participant {participant_id} in group {group}")
         # Return list with Nones or Zeros matching the expected output structure
         return [participant_id, group, 0, 0, 0, 0, 0, 0] # Match the final number of expected stats

    earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    # If earthquake times are None, we can't calculate during/after, handle appropriately

    df_t = _prepare_table_cover_events(df)
    cover_attempts = 0
    total_duration_in_table_cover = 0
    total_duration_in_table_cover_before_earthquake = 0
    total_duration_in_table_cover_during_earthquake = 0
    total_duration_in_table_cover_after_earthquake = 0

    i = 0
    while i < len(df_t):
        # Look for an Entry event
        if df_t.iloc[i]["EventType"] == "EntryUnderTable":
            entry_time = df_t.iloc[i]["Time"]
            # Find the next corresponding Exit event
            if i + 1 < len(df_t) and df_t.iloc[i + 1]["EventType"] == "ExitUnderTable":
                exit_time = df_t.iloc[i + 1]["Time"]
                cover_attempts += 1

                duration_in_table = _calculate_duration(entry_time, exit_time)
                total_duration_in_table_cover += duration_in_table

                duration_before = _calculate_before_earthquake_duration(entry_time, exit_time, earthquake_start_time)
                duration_during = _calculate_earthquake_overlap_duration(entry_time, exit_time, earthquake_start_time, earthquake_end_time)
                duration_after = _calculate_after_earthquake_duration(entry_time, exit_time, earthquake_end_time)

                # Sanity check: duration = before + during + after (within floating point error)
                # if not abs(duration_in_table - (duration_before + duration_during + duration_after)) < 1e-6:
                #      print(f"Warning: Duration mismatch for {participant_id} cover event {i//2}: Total={duration_in_table}, B={duration_before}, D={duration_during}, A={duration_after}")


                total_duration_in_table_cover_before_earthquake += duration_before
                total_duration_in_table_cover_during_earthquake += duration_during
                total_duration_in_table_cover_after_earthquake += duration_after

                i += 2 # Move past the processed pair
            else:
                # Entry event without a subsequent Exit event - ignore or log?
                print(f"Warning: Unmatched 'EntryUnderTable' at time {entry_time} for participant {participant_id} in group {group}")
                i += 1 # Move to the next event
        else:
            # Exit event without a preceding Entry event (shouldn't happen with sorted data, but safety check)
            print(f"Warning: Unmatched 'ExitUnderTable' at time {df_t.iloc[i]['Time']} for participant {participant_id} in group {group}")
            i += 1 # Move to the next event


    # Calculate averages safely
    average_duration_in_table_cover = (total_duration_in_table_cover / cover_attempts) if cover_attempts > 0 else 0
    # Note: Average durations for before/during/after might be less meaningful if events span boundaries. Totals are more robust.

    user_stats = [
        participant_id,
        group,
        cover_attempts,
        average_duration_in_table_cover / 1000, # Average total duration per attempt
        total_duration_in_table_cover / 1000, # Grand total duration
        total_duration_in_table_cover_before_earthquake / 1000, # NEW
        total_duration_in_table_cover_during_earthquake / 1000, # Already existed, renamed for clarity
        total_duration_in_table_cover_after_earthquake / 1000,  # NEW
    ]
    return user_stats
#endregion

#region : Sitting behaviour analysis
def get_seated_stats(df: pd.DataFrame, participant_id: str, group: str) -> list:
    """Calculates statistics related to player sitting behavior."""
    if df.empty or 'Time' not in df.columns or 'PlayerSeated' not in df.columns:
         print(f"Error: DataFrame empty or missing required columns for participant {participant_id} in group {group}")
         return [participant_id, group, 0, 0, 0, 0, 0, 0, 0, 0] # Match the final number of expected stats

    try:
        earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
        # Handle None return values if necessary, e.g., assign infinity or simulation bounds
        if earthquake_start_time is None: earthquake_start_time = float('inf') # No 'during' or 'after' possible
        if earthquake_end_time is None: earthquake_end_time = float('inf') # No 'after' possible (if start was valid)

    except Exception as e: # Catch potential errors in _get_earthquake_times if not handled inside
        print(f"Error getting earthquake times for participant {participant_id} in group {group}: {e}")
        # Assign values that effectively disable during/after calculations
        earthquake_start_time = float('inf')
        earthquake_end_time = float('inf')


    seated_transitions = 0 # Count of times player *sat down*
    num_seated_periods = 0 # Count of distinct seated periods (start-end)
    total_seated_duration = 0
    total_seated_duration_before_earthquake = 0 # NEW
    total_seated_duration_during_earthquake = 0
    total_seated_duration_after_earthquake = 0 # NEW

    current_seated_start_time = None
    last_state = None # Track the previous state

    df_sorted = df.sort_values(by='Time') # Ensure data is time-sorted

    for _, row in df_sorted.iterrows():
        current_time = row['Time']
        is_seated = row['PlayerSeated']

        # Detect state change
        if is_seated != last_state:
            if is_seated:
                # Transitioned to seated
                if current_seated_start_time is None: # Start of a new seated period
                    current_seated_start_time = current_time
                    seated_transitions += 1
            else:
                # Transitioned to not seated
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

        last_state = is_seated


    # Handle if seated at the very end of the simulation data
    if current_seated_start_time is not None:
        start_of_seated_period = current_seated_start_time
        end_of_seated_period = df_sorted['Time'].iloc[-1] # End time is the time of the last record

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
    average_seated_duration = (total_seated_duration / num_seated_periods) if num_seated_periods > 0 else 0
    # Averages for before/during/after might not be as useful as totals here either

    user_stats = [
        participant_id,
        group,
        seated_transitions, # N_S (Number of times sat down) - Renamed variable for clarity
        average_seated_duration / 1000, # Average duration per seated period
        total_seated_duration / 1000, # Grand total seated duration
        total_seated_duration_before_earthquake / 1000, # NEW
        total_seated_duration_during_earthquake / 1000, # Existing, clarified
        total_seated_duration_after_earthquake / 1000,  # NEW
    ]
    return user_stats
#endregion

def main():
    # Create Results directory if it doesn't exist
    if not os.path.exists("Results"):
        os.makedirs("Results")

    items_picked_stats = []
    books_placed_stats = []
    items_observed_stats = []
    table_cover_stats_list = [] # Renamed to avoid conflict
    sitting_behaviour_stats_list = [] # Renamed to avoid conflict

    all_valid_ids = set() # Keep track of IDs for which we successfully processed data

    for group in groups:
        # Skip hidden files/directories like .DS_Store
        if group.startswith('.'):
            continue
        group_path = f"Data/Unity Data/{group}"
        if not os.path.isdir(group_path):
            print(f"Skipping non-directory item: {group_path}")
            continue

        print(f"Processing group: {group}")
        files = glob.glob(f"{group_path}/*.csv") # Explicitly look for CSVs
        if not files:
             print(f"  No CSV files found in {group_path}")
             continue

        for file_path in files:
            participant_id = os.path.basename(file_path).split(".")[0]
            print(f"  Processing file: {os.path.basename(file_path)}")
            df = get_cleaned_data(file_path)

            if df is None or df.empty:
                 print(f"  Skipping {participant_id} due to data cleaning issues or empty data.")
                 continue

            # Check if essential columns exist after cleaning
            required_cols = ['Time', 'EventType', 'PlayerSeated']
            if not all(col in df.columns for col in required_cols):
                 print(f"  Skipping {participant_id} due to missing required columns after cleaning.")
                 continue

            all_valid_ids.add((participant_id, group)) # Add valid ID/Group pair

            # Safely append stats, functions should handle internal errors and return default lists
            items_picked_stats.append(get_items_picked_stats(df.copy(), participant_id, group)) # Use df.copy() if functions modify df inplace unexpectedly
            table_cover_stats_list.append(get_table_cover_stats(df.copy(), participant_id, group))
            sitting_behaviour_stats_list.append(get_seated_stats(df.copy(), participant_id, group))

            if group in ['Group 1', 'Group 2']:
                books_placed_stats.append(get_books_placed_stats(df.copy(), participant_id, group))
            else: # Assume Groups 3 and 4 (or others) do item observation
                items_observed_stats.append(get_items_observed_stats(df.copy(), participant_id, group))

    # Define column names explicitly, including the new ones
    items_picked_cols = ["ID", "Group", "ItemsPickedBeforeEarthquake", "ItemsPickedDuringEarthquake", "ItemsPickedAfterEarthquake"]
    books_placed_cols = ["ID", "Group", "BooksPlacedBeforeEarthquake", "BooksPlacedDuringEarthquake", "BooksPlacedAfterEarthquake"]
    items_observed_cols = ["ID", "Group", "ItemsObservedBeforeEarthquake", "ItemsObservedDuringEarthquake", "ItemsObservedAfterEarthquake"]
    table_cover_cols = [
        "ID", "Group", "CoverAttempts", "AverageDurationInTableCover",
        "Total Duration In table Cover", # CHANGED to match request
        "Total Duration In table Cover Before Earthquake", # NEW
        "Total Duration In table Cover During Earthquake", # NEW (was TotalDurationInTableCoverDuringEarthquake)
        "Total Duration In table Cover After Earthquake"  # NEW
    ]
    sitting_behaviour_cols = [
        "ID", "Group", "SittingTransitions", "AverageSeatedDuration",
        "Total Seated Duration", # CHANGED to match request
        "Total Seated Duration Before Earthquake", # NEW
        "Total Seated Duration During Earthquake", # NEW (was TotalSeatedDurationDuringEarthquake)
        "Total Seated Duration After Earthquake"  # NEW
    ]

    # Create DataFrames
    # Use try-except blocks for DataFrame creation in case lists are empty
    try:
        items_picked_df = pd.DataFrame(items_picked_stats, columns=items_picked_cols)
    except ValueError:
        print("Warning: No data for items_picked_stats.")
        items_picked_df = pd.DataFrame(columns=items_picked_cols)

    try:
        books_placed_df = pd.DataFrame(books_placed_stats, columns=books_placed_cols)
    except ValueError:
        print("Warning: No data for books_placed_stats.")
        books_placed_df = pd.DataFrame(columns=books_placed_cols)

    try:
        items_observed_df = pd.DataFrame(items_observed_stats, columns=items_observed_cols)
    except ValueError:
        print("Warning: No data for items_observed_stats.")
        items_observed_df = pd.DataFrame(columns=items_observed_cols)

    try:
        table_cover_df = pd.DataFrame(table_cover_stats_list, columns=table_cover_cols)
    except ValueError:
        print("Warning: No data for table_cover_stats_list.")
        table_cover_df = pd.DataFrame(columns=table_cover_cols)

    try:
        sitting_behaviour_df = pd.DataFrame(sitting_behaviour_stats_list, columns=sitting_behaviour_cols)
    except ValueError:
        print("Warning: No data for sitting_behaviour_stats_list.")
        sitting_behaviour_df = pd.DataFrame(columns=sitting_behaviour_cols)


    # Create a base DataFrame with all processed IDs to ensure all participants are included
    if all_valid_ids:
         base_df = pd.DataFrame(list(all_valid_ids), columns=["ID", "Group"])
    else:
         print("Error: No valid participant data was processed. Final CSV will be empty.")
         base_df = pd.DataFrame(columns=["ID", "Group"])


    # Merge all DataFrames onto the base DataFrame
    final = base_df
    dfs_to_merge = [items_picked_df, books_placed_df, items_observed_df, table_cover_df, sitting_behaviour_df]
    for df_merge in dfs_to_merge:
        if not df_merge.empty:
             # Ensure ID and Group columns are suitable for merging (e.g., string type if needed)
             # final['ID'] = final['ID'].astype(str)
             # final['Group'] = final['Group'].astype(str)
             # df_merge['ID'] = df_merge['ID'].astype(str)
             # df_merge['Group'] = df_merge['Group'].astype(str)
             final = pd.merge(final, df_merge, on=["ID", "Group"], how="left") # Use left merge to keep all participants


    # Add Task and Information columns
    final["Task"] = final["Group"].apply(lambda x: "Book Task" if x in ["Group 1", "Group 2"] else "No Task")
    final["Information"] = final["Group"].apply(lambda x: "Given" if x in ["Group 1", "Group 3"] else "Not Given")

    # Reorder columns to make "Task" and "Information" the 3rd and 4th columns
    cols = final.columns.tolist()
    # Remove Task and Information from their current position
    cols.remove("Task")
    cols.remove("Information")
    # Insert them after ID and Group
    final_cols_order = cols[:2] + ["Task", "Information"] + cols[2:]
    final = final[final_cols_order]

    # Optional: Drop the original 'Group' column if desired (usually kept for reference)
    # final.drop(columns=["Group"], inplace=True)

    # Fill NaN values that resulted from merges (e.g., book stats for non-book groups) with 0 or appropriate value
    final.fillna(0, inplace=True)

    # Save the final DataFrame
    output_path = "Results/CovertedUnityData.csv"
    try:
        final.to_csv(output_path, index=False)
        print(f"Successfully saved final data to {output_path}")
    except Exception as e:
        print(f"Error saving final data to {output_path}: {e}")


if __name__ == "__main__":
    main()