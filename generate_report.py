import pandas as pd
import os
import glob
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt
from io import BytesIO

# Define the base directory for the data
data_base_dir = "Unity Data"

# Preparation
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
    i = 0
    while i < len(table_interaction) - 1:
        if table_interaction.iloc[i + 1]["Time"] - table_interaction.iloc[i]["Time"] < minimum_time_with_table:
            df.drop(index=[table_interaction.index[i], table_interaction.index[i + 1]], inplace=True)
            table_interaction.drop(index=[table_interaction.index[i], table_interaction.index[i + 1]], inplace=True)
        else:
            i += 2
    return df

def remove_initial_books_placement(df):
    """Removes initial books placement from the DataFrame."""
    simulation_start_time = df[df["EventType"] == "SimulationStarted"]["Time"].values
    if simulation_start_time.size > 0:
        filter_time = simulation_start_time[0] + 1000  # milliseconds
        entry_books = df[df["EventType"] == "BookPlaced"]
        rows_to_remove = entry_books[entry_books["Time"] < filter_time]
        df.drop(index=rows_to_remove.index, inplace=True)
    return df

def get_cleaned_data(file_path, groups):
    """Main function to perform data pre-cleaning."""
    df = load_and_clean_data(file_path)
    df = filter_simulation_data(df)
    df = remove_fake_sitting_indications(df)
    df = remove_initial_books_placement(df)
    return df

# Task Specific Analysis

### Book placement (Group 1, 2)
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

def get_books_placed_stats_for_group(group, data_file_paths, groups):
    """Returns the number of books placed before, during and after earthquake for the given group."""
    books_placed_stats = []
    for file_path in data_file_paths[group]:
        df = get_cleaned_data(file_path, groups)
        books_placed_stats.append(get_books_placed_stats(df, os.path.basename(file_path).split(".")[0], group))
    books_placed_stats = pd.DataFrame(books_placed_stats, columns=["ID", "Group", "BooksPlacedBeforeEarthquake", "BooksPlacedDuringEarthquake", "BooksPlacedAfterEarthquake"])
    return books_placed_stats

def get_average_book_placed_stats_for_all_groups(data_file_paths, groups):
    """Returns the average number of books placed before, during and after earthquake for all groups."""
    books_placed_stats = []
    for group in data_file_paths:
        if group in ["Group 1", "Group 2"]:
            books_placed_stats.append(get_books_placed_stats_for_group(group, data_file_paths, groups))
    if books_placed_stats:
        books_placed_stats = pd.concat(books_placed_stats)
        # Updated column names
        books_placed_stats = books_placed_stats.rename(columns={
            "Group": "Group",
            "BooksPlacedBeforeEarthquake": "Books Placed Before Earthquake",
            "BooksPlacedDuringEarthquake": "Books Placed During Earthquake",
            "BooksPlacedAfterEarthquake": "Books Placed After Earthquake"
        })
        return books_placed_stats.drop(columns='ID').groupby('Group').mean().reset_index()
    else:
        return pd.DataFrame(columns=["Group", "BooksPlacedBeforeEarthquake", "BooksPlacedDuringEarthquake", "BooksPlacedAfterEarthquake"])

### Item Observation Task (Group 3, 4)
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

def get_items_observed_stats_for_group(group, data_file_paths, groups):
    """Returns the number of books placed before, during and after earthquake for the given group."""
    items_observed_stats = []
    for file_path in data_file_paths[group]:
        df = get_cleaned_data(file_path, groups)
        items_observed_stats.append(get_items_observed_stats(df, os.path.basename(file_path).split(".")[0], group))
    items_observed_stats = pd.DataFrame(items_observed_stats, columns=["ID", "Group", "ItemsObservedBeforeEarthquake", "ItemsObservedDuringEarthquake", "ItemsObservedAfterEarthquake"])
    return items_observed_stats

def get_average_items_observed_stats_for_all_groups(data_file_paths, groups):
    """Returns the average number of books placed before, during and after earthquake for all groups."""
    items_observed_stats = []
    for group in data_file_paths:
        if group in ["Group 3", "Group 4"]:
            items_observed_stats.append(get_items_observed_stats_for_group(group, data_file_paths, groups))
    if items_observed_stats:
        items_observed_stats = pd.concat(items_observed_stats)
        # Updated column names
        items_observed_stats = items_observed_stats.rename(columns={
            "Group": "Group",
            "ItemsObservedBeforeEarthquake": "Items Observed Before Earthquake",
            "ItemsObservedDuringEarthquake": "Items Observed During Earthquake",
            "ItemsObservedAfterEarthquake": "Items Observed After Earthquake"
        })
        return items_observed_stats.drop(columns='ID').groupby('Group').mean().reset_index()
    else:
        return pd.DataFrame(columns=["Group", "ItemsObservedBeforeEarthquake", "ItemsObservedDuringEarthquake", "ItemsObservedAfterEarthquake"])

# Participant Actions

### Number of items picked
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

def get_items_picked_stats_for_group(group, data_file_paths, get_cleaned_data, groups):
    """Returns the number of items picked before, during and after earthquake for the given group."""
    items_picked_stats = []
    for file_path in data_file_paths[group]:
        df = get_cleaned_data(file_path, groups)
        items_picked_stats.append(get_items_picked_stats(df, os.path.basename(file_path).split(".")[0], group))
    items_picked_stats = pd.DataFrame(items_picked_stats, columns=["ID", "Group", "ItemsPickedBeforeEarthquake", "ItemsPickedDuringEarthquake", "ItemsPickedAfterEarthquake"])
    return items_picked_stats

def get_average_items_picked_stats_for_all_groups(data_file_paths, get_cleaned_data, groups):
    """Returns the average number of items picked before, during and after earthquake for all groups."""
    items_picked_stats = []
    for group in data_file_paths:
        items_picked_stats.append(get_items_picked_stats_for_group(group, data_file_paths, get_cleaned_data, groups))
    if items_picked_stats:
        items_picked_stats = pd.concat(items_picked_stats)
        # Updated column names
        items_picked_stats = items_picked_stats.rename(columns={
            "Group": "Group",
            "ItemsPickedBeforeEarthquake": "Items Picked Before Earthquake",
            "ItemsPickedDuringEarthquake": "Items Picked During Earthquake",
            "ItemsPickedAfterEarthquake": "Items Picked After Earthquake"
        })
        return items_picked_stats.drop(columns='ID').groupby('Group').mean().reset_index()
    else:
        return pd.DataFrame(columns=["Group", "ItemsPickedBeforeEarthquake", "ItemsPickedDuringEarthquake", "ItemsPickedAfterEarthquake"])

### Took table cover during earthquake
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
    cover_attempts = df_t.shape[0] // 2
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


def get_table_cover_stats_for_group(group, data_file_paths, groups):
    """Calculates statistics related to taking cover under a table during an earthquake for all participants in a group."""
    table_cover_stats = []
    for file_path in data_file_paths[group]:
        df = get_cleaned_data(file_path, groups)
        table_cover_stats.append(get_table_cover_stats(df, os.path.basename(file_path).split(".")[0], group))
    table_cover_stats = pd.DataFrame(
        table_cover_stats,
        columns=[
            "ID",
            "Group",
            "CoverAttempts",
            "AverageDurationInTableCover",
            "TotalDurationInTableCover",
            "TotalDurationInTableCoverDuringEarthquake",
            "AverageDurationInTableCoverDuringEarthquake",
        ],
    )
    return table_cover_stats

def get_table_cover_stats_for_all_groups(data_file_paths, groups):
    """Calculates statistics related to taking cover under a table during an earthquake for all participants in all groups."""
    table_cover_stats = []
    for group in data_file_paths:
        table_cover_stats.append(get_table_cover_stats_for_group(group, data_file_paths, groups))
    if table_cover_stats:
        table_cover_stats = pd.concat(table_cover_stats)
        cover_attempt_summary = table_cover_stats.groupby(['Group', table_cover_stats["CoverAttempts"] > 0]).size().unstack(fill_value=0)
        cover_attempt_summary.rename(columns={False: 'No Cover Attempts', True: 'Cover Attempts'}, inplace=True)
        average_stats = table_cover_stats.drop(columns='ID').groupby('Group').mean().reset_index()
        # Update column names
        average_stats = average_stats.rename(columns={
            "Group": "Group",
            "CoverAttempts": "Cover Attempts",
            "AverageDurationInTableCover": "Average Duration\nIn Table Cover",
            "TotalDurationInTableCover": "Total Duration\nIn Table\nCover",
            "TotalDurationInTableCoverDuringEarthquake": "Total Duration\nIn Table\nCover During\nEarthquake",
            "AverageDurationInTableCoverDuringEarthquake": "Average Duration\nIn Table\nCover During\nEarthquake"
        })
        return average_stats, cover_attempt_summary
    else:
        return pd.DataFrame(), pd.DataFrame()

### Player sitting behaviour
def get_seated_stats(df: pd.DataFrame, participant_id: str, group: str) -> list[str | float]:
    try:
        earthquake_start_time, earthquake_end_time = _get_earthquake_times(df)
    except ValueError as e:
        print(f"Error for participant {participant_id} in group {group}: {e}")
        return [participant_id, group, 0, 0, 0, 0, 0]  # Added more zeros for new metrics

    seated_count = 0
    unseated_count = 0
    num_seated_periods = 0
    total_seated_duration = 0
    total_seated_duration_during_earthquake = 0
    current_seated_start_time = None
    was_seated = False

    for index, row in df.iterrows():
        current_time = row['Time']
        is_seated = row['PlayerSeated']

        if is_seated and not was_seated:
            seated_count += 1
            current_seated_start_time = current_time
        elif not is_seated and was_seated and current_seated_start_time is not None:
            duration = current_time - current_seated_start_time
            total_seated_duration += duration
            num_seated_periods += 1
            overlap = _calculate_earthquake_overlap_duration(
                current_seated_start_time, current_time, earthquake_start_time, earthquake_end_time
            )
            total_seated_duration_during_earthquake += overlap
            current_seated_start_time = None
            unseated_count += 1

        was_seated = is_seated

    # Handle if seated at the end
    if was_seated and current_seated_start_time is not None:
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

def get_seated_stats_for_group(group, data_file_paths, groups):
    seated_stats = []
    for file_path in data_file_paths[group]:
        df = get_cleaned_data(file_path, groups)
        seated_stats.append(get_seated_stats(df, os.path.basename(file_path).split(".")[0], group))
    seated_stats = pd.DataFrame(
        seated_stats,
        columns=[
            "ID",
            "Group",
            "SeatedCount",
            "AverageSeatedDuration",
            "TotalSeatedDuration",
            "TotalSeatedDurationDuringEarthquake",
            "AverageSeatedDurationDuringEarthquake",
        ],
    )
    return seated_stats

def get_seated_stats_for_all_groups(data_file_paths, groups):
    seated_stats = []
    for group in data_file_paths:
        seated_stats.append(get_seated_stats_for_group(group, data_file_paths, groups))
    if seated_stats:
        seated_stats = pd.concat(seated_stats)
        # Updated column names
        seated_stats = seated_stats.rename(columns={
            "Group": "Group",
            "SeatedCount": "Seated Count",
            "AverageSeatedDuration": "Average Seated\nDuration",
            "TotalSeatedDuration": "Total Seated\nDuration",
            "TotalSeatedDurationDuringEarthquake": "Total Seated\nDuration During\nEarthquake",
            "AverageSeatedDurationDuringEarthquake": "Average Seated\nDuration During\nEarthquake"
        })
        return seated_stats.drop(columns='ID').groupby('Group').mean().reset_index()
    else:
        return pd.DataFrame()

def generate_report(output_filename="analysis_report.pdf"):
    """Generates a PDF report of the analysis results."""
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Analysis Report of Earthquake Simulation Data", styles['h1']))
    story.append(Spacer(1, 12))

    # Definitions
    story.append(Paragraph("<b>Definitions:</b>", styles['h2']))
    story.append(Paragraph("- <b>Before Earthquake:</b> Time period before the 'EarthquakeStart' event.", styles['Normal']))
    story.append(Paragraph("- <b>During Earthquake:</b> Time period between the 'EarthquakeStart' and the last 'EarthquakeEnd' event.", styles['Normal']))
    story.append(Paragraph("- <b>After Earthquake:</b> Time period after the last 'EarthquakeEnd' event.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Load data file paths
    groups = os.listdir(data_base_dir)
    data_file_paths = {group: glob.glob(f"{data_base_dir}/{group}/*") for group in groups}

    # Analysis and Report Generation

    # Book Placement Analysis
    story.append(Paragraph("<b>Book Placement Analysis (Groups 1 & 2)</b>", styles['h2']))
    book_placement_results = get_average_book_placed_stats_for_all_groups(data_file_paths, groups)
    if not book_placement_results.empty:
        book_placement_results = book_placement_results.round(2)
        data = [book_placement_results.columns.tolist()] + book_placement_results.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 1),
            ('SPLITBYROW', (0, 0), (-1, -1), 1)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Basic Plot for Book Placement
        fig, ax = plt.subplots()
        book_placement_results.plot(x='Group', kind='bar', ax=ax)
        ax.set_ylabel("Average Number of Books Placed")
        ax.set_title("Average Books Placed Before, During, and After Earthquake")
        plt.xticks(rotation=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=400, height=300)
        story.append(img)
        plt.close(fig)
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No data found for Book Placement Analysis.", styles['Normal']))
        story.append(Spacer(1, 12))

    # Item Observation Analysis
    story.append(Paragraph("<b>Item Observation Analysis (Groups 3 & 4)</b>", styles['h2']))
    item_observation_results = get_average_items_observed_stats_for_all_groups(data_file_paths, groups)
    if not item_observation_results.empty:
        item_observation_results = item_observation_results.round(2)
        data = [item_observation_results.columns.tolist()] + item_observation_results.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 1),
            ('SPLITBYROW', (0, 0), (-1, -1), 1)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Basic Plot for Item Observation
        fig, ax = plt.subplots()
        item_observation_results.plot(x='Group', kind='bar', ax=ax)
        ax.set_ylabel("Average Number of Items Observed")
        ax.set_title("Average Items Observed Before, During, and After Earthquake")
        plt.xticks(rotation=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=400, height=300)
        story.append(img)
        plt.close(fig)
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No data found for Item Observation Analysis.", styles['Normal']))
        story.append(Spacer(1, 12))

    # Item Picking Analysis
    story.append(Paragraph("<b>Item Picking Analysis (All Groups)</b>", styles['h2']))
    item_picking_results = get_average_items_picked_stats_for_all_groups(data_file_paths, get_cleaned_data, groups)
    if not item_picking_results.empty:
        item_picking_results = item_picking_results.round(2)
        data = [item_picking_results.columns.tolist()] + item_picking_results.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 1),
            ('SPLITBYROW', (0, 0), (-1, -1), 1)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Basic Plot for Item Picking
        fig, ax = plt.subplots()
        item_picking_results.plot(x='Group', kind='bar', ax=ax)
        ax.set_ylabel("Average Number of Items Picked")
        ax.set_title("Average Items Picked Before, During, and After Earthquake")
        plt.xticks(rotation=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=400, height=300)
        story.append(img)
        plt.close(fig)
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No data found for Item Picking Analysis.", styles['Normal']))
        story.append(Spacer(1, 12))

    # Table Cover Analysis
    story.append(Paragraph("<b>Table Cover Usage Analysis (All Groups)</b>", styles['h2']))
    table_cover_average_results, table_cover_attempt_summary = get_table_cover_stats_for_all_groups(data_file_paths, groups)
    if not table_cover_average_results.empty:
        story.append(Paragraph("<b>Average Table Cover Statistics:</b>", styles['h3']))
        table_cover_average_results = table_cover_average_results.round(2)
        data_average = [table_cover_average_results.columns.tolist()] + table_cover_average_results.values.tolist()
        table_average = Table(data_average, colWidths=[70, 80, 80, 80, 80, 80, 80])
        table_average.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 1),
            ('SPLITBYROW', (0, 0), (-1, -1), 1)
        ]))
        story.append(table_average)
        story.append(Spacer(1, 12))

        if not table_cover_attempt_summary.empty:
            story.append(Paragraph("<b>Table Cover Attempt Summary:</b>", styles['h3']))
            data_attempt = [["Group"] + table_cover_attempt_summary.columns.tolist()] + [[index] + row.tolist() for index, row in table_cover_attempt_summary.iterrows()]
            table_attempt = Table(data_attempt)
            table_attempt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('WORDWRAP', (0, 0), (-1, -1), 1),
                ('SPLITBYROW', (0, 0), (-1, -1), 1)
            ]))
            story.append(table_attempt)
            story.append(Spacer(1, 12))

        # Basic Plot for Table Cover Attempts
        fig, ax = plt.subplots()
        table_cover_attempt_summary.plot(kind='bar', ax=ax)
        ax.set_ylabel("Number of Participants")
        ax.set_title("Table Cover Attempts vs. No Attempts by Group")
        plt.xticks(rotation=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=400, height=300)
        story.append(img)
        plt.close(fig)
        story.append(Spacer(1, 12))

        # Basic Plot for Average Table Cover Duration
        if 'AverageDurationInTableCover' in table_cover_average_results.columns:
            fig, ax = plt.subplots()
            table_cover_average_results.plot(x='Group', y='AverageDurationInTableCover', kind='bar', ax=ax)
            ax.set_ylabel("Average Duration (seconds)")
            ax.set_title("Average Duration in Table Cover by Group")
            plt.xticks(rotation=0)
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image(buf, width=400, height=300)
            story.append(img)
            plt.close(fig)
            story.append(Spacer(1, 12))

    else:
        story.append(Paragraph("No data found for Table Cover Usage Analysis.", styles['Normal']))
        story.append(Spacer(1, 12))

    # Player Sitting Behaviour Analysis
    story.append(Paragraph("<b>Player Sitting Behaviour Analysis (All Groups)</b>", styles['h2']))
    seated_behaviour_results = get_seated_stats_for_all_groups(data_file_paths, groups)
    if not seated_behaviour_results.empty:
        seated_behaviour_results = seated_behaviour_results.round(2)
        data = [seated_behaviour_results.columns.tolist()] + seated_behaviour_results.values.tolist()
        table = Table(data, colWidths=[70, 80, 80, 80, 80, 80, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1), 1),
            ('SPLITBYROW', (0, 0), (-1, -1), 1)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Basic Plot for Average Seated Duration
        fig, ax = plt.subplots()
        seated_behaviour_results.plot(x='Group', y='Average Seated\nDuration', kind='bar', ax=ax)
        ax.set_ylabel("Average Seated Duration (seconds)")
        ax.set_title("Average Seated Duration by Group")
        plt.xticks(rotation=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=400, height=300)
        story.append(img)
        plt.close(fig)
        story.append(Spacer(1, 12))

    else:
        story.append(Paragraph("No data found for Player Sitting Behaviour Analysis.", styles['Normal']))
        story.append(Spacer(1, 12))

    # Build the PDF
    doc.build(story)

if __name__ == "__main__":
    # Ensure the "Unity Data" directory exists
    if not os.path.exists(data_base_dir):
        print(f"Error: The directory '{data_base_dir}' does not exist in the current path.")
        print("Please make sure your data is organized as 'Unity Data/Group X/participant_data.csv'.")
    else:
        generate_report()
        print("PDF report 'analysis_report.pdf' generated successfully.")