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
   df.drop(columns=["Score", "Penalty", "ID", "RollNumber", "Group"], inplace=True)
   df.drop(index=0, inplace=True)
   return df

def filter_simulation_data(df):
   """Filters out the simulation data from the DataFrame."""
   mask_start = df[df["EventType"] == "SimulationStarted"]
   mask_end = df[df["EventType"] == "SimulationEnded"]
   if mask_end.empty:
      mask_end = df.iloc[-1, :]
   return df.loc[mask_start.index[-1]:mask_end.index[0]]

def remove_fake_sitting_indications(df):
    """Removes fake sitting indications from the DataFrame."""
    entry_under_table = df[df["EventType"] == "EntryUnderTable"]
    exit_under_table = df[df["EventType"] == "ExitUnderTable"]
    table_interaction = pd.concat([entry_under_table, exit_under_table])
    table_interaction.sort_values(by="Time", inplace=True)
    minimum_time_with_table = 500  # milliseconds

    for i in range(0, len(table_interaction) - 1, 2):  # Ensure we only process valid pairs
        if i + 1 < len(table_interaction):  # Check if the next index is within bounds
            if table_interaction.iloc[i + 1]["Time"] - table_interaction.iloc[i]["Time"] < minimum_time_with_table:
                df.drop(index=[table_interaction.index[i], table_interaction.index[i + 1]], inplace=True)
    return df

def remove_initial_books_placement(df):
   """Removes initial books placement from the DataFrame."""
   filter_time = df[df["EventType"] == "SimulationStarted"]["Time"].values[0] + 1000 # milliseconds
   entry_books = df[df["EventType"] == "BookPlaced"]
   rows_to_remove = entry_books[entry_books["Time"] < filter_time]
   df.drop(index=rows_to_remove.index, inplace=True)
   return df

def get_cleaned_data(file_path):
   """Main function to perform data pre-cleaning."""
   df = load_and_clean_data(file_path)
   df = filter_simulation_data(df)
   df = remove_fake_sitting_indications(df)
   df = remove_initial_books_placement(df)
   return df
#endregion

#region : Books Placement
def get_books_placed_stats(df, id, group):
   earthquake_start_index = df[df["EventType"] == "EarthquakeStart"].index[0]
   earthquake_end_index = df[df["EventType"] == "EarthquakeEnd"].index[-1]
   before_earthquake_data = df.loc[:earthquake_start_index]
   after_earthquake_data = df.loc[earthquake_end_index:]
   during_earthquake_data = df.loc[earthquake_start_index:earthquake_end_index]

   # Calcaulate number of books placed before, during and after earthquake
   books_before_earthquake = before_earthquake_data[before_earthquake_data["EventType"] == "BookPlaced"].shape[0]
   books_during_earthquake = during_earthquake_data[during_earthquake_data["EventType"] == "BookPlaced"].shape[0]
   books_after_earthquake = after_earthquake_data[after_earthquake_data["EventType"] == "BookPlaced"].shape[0]

   return [id, group, books_before_earthquake, books_during_earthquake, books_after_earthquake]
#endregion 

#region : Item Observation
def get_items_observed_stats(df, id, group):
   earthquake_start_index = df[df["EventType"] == "EarthquakeStart"].index[0]
   earthquake_end_index = df[df["EventType"] == "EarthquakeEnd"].index[-1]
   before_earthquake_data = df.loc[:earthquake_start_index]
   after_earthquake_data = df.loc[earthquake_end_index:]
   during_earthquake_data = df.loc[earthquake_start_index:earthquake_end_index]

   # Calcaulate number of books placed before, during and after earthquake
   items_observed_before_earthquake = before_earthquake_data[before_earthquake_data["EventType"] == "ItemObserved"].shape[0]
   items_observed_during_earthquake = during_earthquake_data[during_earthquake_data["EventType"] == "ItemObserved"].shape[0]
   items_observed_after_earthquake = after_earthquake_data[after_earthquake_data["EventType"] == "ItemObserved"].shape[0]

   return [id, group, items_observed_before_earthquake, items_observed_during_earthquake, items_observed_after_earthquake]
#endregion

#region : Items picked
def get_items_picked_stats(df, id, group):
   """Calculates the number of items picked before, during, and after an earthquake."""
   earthquake_start_index = df[df["EventType"] == "EarthquakeStart"].index[0]
   earthquake_end_index = df[df["EventType"] == "EarthquakeEnd"].index[-1]
   before_earthquake_data = df.loc[:earthquake_start_index]
   after_earthquake_data = df.loc[earthquake_end_index:]
   during_earthquake_data = df.loc[earthquake_start_index:earthquake_end_index]

   items_picked_before_earthquake = before_earthquake_data[before_earthquake_data["EventType"] == "ItemPicked"].shape[0]
   items_picked_during_earthquake = during_earthquake_data[during_earthquake_data["EventType"] == "ItemPicked"].shape[0]
   items_picked_after_earthquake = after_earthquake_data[after_earthquake_data["EventType"] == "ItemPicked"].shape[0]

   return [id, group, items_picked_before_earthquake, items_picked_during_earthquake, items_picked_after_earthquake]
#endregion

#region : Table cover analysis
def _get_earthquake_times(df: pd.DataFrame) -> tuple[float, float]:
    """Extracts the start and end times of the earthquake from the DataFrame."""

    earthquake_start_events = df[df["EventType"] == "EarthquakeStart"]
    earthquake_end_events = df[df["EventType"] == "EarthquakeEnd"]
    if not earthquake_start_events.empty and not earthquake_end_events.empty:
        start_time = earthquake_start_events.iloc[0]["Time"]
        end_time = earthquake_end_events.iloc[-1]["Time"]
        return start_time, end_time
    else:
        raise ValueError("EarthquakeStart or EarthquakeEnd event not found in the DataFrame.")

def _prepare_table_cover_events(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares a DataFrame containing only table cover entry and exit events, sorted by index."""

    table_cover_taken = df[df["EventType"] == "EntryUnderTable"]
    table_cover_removed = df[df["EventType"] == "ExitUnderTable"]
    df_t = pd.concat([table_cover_taken, table_cover_removed])
    df_t.sort_index(inplace=True)
    return df_t

def _calculate_duration(start_time: float, end_time: float) -> float:
    """Calculates the duration between two time points."""
    return end_time - start_time

def _calculate_earthquake_overlap_duration(entry_time: float, exit_time: float, earthquake_start_time: float, earthquake_end_time: float) -> float:
    """Calculates the duration of overlap between a table cover event and the earthquake."""

    return max(0, min(exit_time, earthquake_end_time) - max(entry_time, earthquake_start_time))

def get_table_cover_stats(df: pd.DataFrame, participant_id: str, group: str) -> list[str | float]:
    """Calculates statistics related to taking cover under a table during an earthquake for a single participant."""
    
    try:
        earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    except ValueError as e:
        print(f"Error for participant {participant_id} in group {group}: {e}")
        return [participant_id, group, 0, 0, 0, 0, 0]

    df_t = _prepare_table_cover_events(df)
    cover_attempts = ceil(df_t.shape[0] / 2)
    total_duration_in_table_cover = 0
    total_duration_in_table_cover_during_earthquake = 0

    for i in range(0, len(df_t), 2):
        try:
            entry_time = df_t.iloc[i]["Time"]
            exit_time = df_t.iloc[i + 1]["Time"]

            duration_in_table = _calculate_duration(entry_time, exit_time)
            total_duration_in_table_cover += duration_in_table

            overlap_duration = _calculate_earthquake_overlap_duration(
                entry_time, exit_time, earthquake_start_time, earthquake_end_time
            )
            total_duration_in_table_cover_during_earthquake += overlap_duration

        except IndexError:
            print(f"Error: Mismatched entry and exit under table events for participant {participant_id} in group {group}")
            continue

    if cover_attempts > 0:
        average_duration_in_table_cover = total_duration_in_table_cover / cover_attempts
        average_duration_in_table_cover_during_earthquake = total_duration_in_table_cover_during_earthquake / cover_attempts
    else:
        average_duration_in_table_cover = 0
        average_duration_in_table_cover_during_earthquake = 0

    user_stats = [
        participant_id,
        group,
        cover_attempts,
        average_duration_in_table_cover / 1000,
        total_duration_in_table_cover / 1000,
        total_duration_in_table_cover_during_earthquake / 1000,
        average_duration_in_table_cover_during_earthquake / 1000,
    ]
    return user_stats
#endregion

#region : Sitting behaviour analysis
def get_seated_stats(df: pd.DataFrame, participant_id: str, group: str) -> list[str | float]:
    try:
        earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    except ValueError as e:
        print(f"Error for participant {participant_id} in group {group}: {e}")
        return [participant_id, group, 0, 0, 0, 0, 0, 0, 0]  # Added more zeros for new metrics

    seated_count = 0
    unseated_count = 0
    num_seated_periods = 0
    total_seated_duration = 0
    total_seated_duration_during_earthquake = 0
    current_seated_start_time = None

    for _, row in df.iterrows():
        current_time = row['Time']
        is_seated = row['PlayerSeated']

        if is_seated and current_seated_start_time is None:
            current_seated_start_time = current_time
            seated_count += 1
        elif not is_seated and current_seated_start_time is not None:
            duration = current_time - current_seated_start_time
            total_seated_duration += duration
            num_seated_periods += 1
            overlap = _calculate_earthquake_overlap_duration(
                current_seated_start_time, current_time, earthquake_start_time, earthquake_end_time
            )
            total_seated_duration_during_earthquake += overlap
            current_seated_start_time = None
            unseated_count += 1

    # Handle if seated at the end
    if current_seated_start_time is not None:
        duration = df['Time'].iloc[-1] - current_seated_start_time
        total_seated_duration += duration
        num_seated_periods += 1
        overlap = _calculate_earthquake_overlap_duration(
            current_seated_start_time, df['Time'].iloc[-1], earthquake_start_time, earthquake_end_time
        )
        total_seated_duration_during_earthquake += overlap

    average_seated_duration = total_seated_duration / num_seated_periods if num_seated_periods > 0 else 0
    average_seated_duration_during_earthquake = total_seated_duration_during_earthquake / num_seated_periods if num_seated_periods > 0 else 0

    user_stats = [
        participant_id,
        group,
        seated_count,  # N_S
        average_seated_duration / 1000,
        total_seated_duration / 1000,
        total_seated_duration_during_earthquake / 1000,
        average_seated_duration_during_earthquake / 1000,
    ]
    return user_stats
#endregion

def main():
    items_picked_stats = []
    books_placed_stats = []
    items_observed_stats = []
    table_cover_stats = []
    sitting_behaviour_stats = []
    for group in groups:
        for file_path in data_files[group]:
            df = get_cleaned_data(file_path)
            items_picked_stats.append(get_items_picked_stats(df, os.path.basename(file_path).split(".")[0], group))
            table_cover_stats.append(get_table_cover_stats(df, os.path.basename(file_path).split(".")[0], group))
            sitting_behaviour_stats.append(get_seated_stats(df, os.path.basename(file_path).split(".")[0], group))
            if group in ['Group 1', "Group 2"]:
                books_placed_stats.append(get_books_placed_stats(df, os.path.basename(file_path).split(".")[0], group))
            else:
                items_observed_stats.append(get_items_observed_stats(df, os.path.basename(file_path).split(".")[0], group))
    
    items_picked_stats = pd.DataFrame(items_picked_stats, columns=["ID", "Group", "ItemsPickedBeforeEarthquake", "ItemsPickedDuringEarthquake", "ItemsPickedAfterEarthquake"])
    books_placed_stats = pd.DataFrame(books_placed_stats, columns=["ID", "Group", "BooksPlacedBeforeEarthquake", "BooksPlacedDuringEarthquake", "BooksPlacedAfterEarthquake"])
    items_observed_stats = pd.DataFrame(items_observed_stats, columns=["ID", "Group", "ItemsObservedBeforeEarthquake", "ItemsObservedDuringEarthquake", "ItemsObservedAfterEarthquake"])
    table_cover_stats = pd.DataFrame(table_cover_stats, columns=["ID", "Group", "CoverAttempts", "AverageDurationInTableCover", "TotalDurationInTableCover", "TotalDurationInTableCoverDuringEarthquake", "AverageDurationInTableCoverDuringEarthquake"])
    sitting_behaviour_stats = pd.DataFrame(sitting_behaviour_stats, columns=["ID", "Group", "SittingCount", "AverageSeatedDuration", "TotalSeatedDuration", "TotalSeatedDurationDuringEarthquake", "AverageSeatedDurationDuringEarthquake"])
    
    final = pd.merge(items_picked_stats, books_placed_stats, on=["ID", "Group"], how="outer")
    final = pd.merge(final, items_observed_stats, on=["ID", "Group"], how="outer")
    final = pd.merge(final, table_cover_stats, on=["ID", "Group"], how="outer")
    final = pd.merge(final, sitting_behaviour_stats, on=["ID", "Group"], how="outer")

    final["Task"] = final["Group"].apply(lambda x: "Book Task" if x in ["Group 1", "Group 2"] else "No Task")
    final["Information"] = final["Group"].apply(lambda x: "Given" if x in ["Group 1", "Group 3"] else "Not Given")
    # Reorder columns to make "Task" and "Information" the 3rd and 4th columns
    columns_order = ["ID", "Group", "Task", "Information"] + [col for col in final.columns if col not in ["ID", "Group", "Task", "Information"]]
    final = final[columns_order]
    # final.drop(columns=["Group"], inplace=True)

    final.to_csv(f"Results/CovertedUnityData.csv", index=False)

if __name__ == "__main__":
   main()