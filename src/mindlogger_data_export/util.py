"""Utilities carried over from original scripts."""

import re

import numpy as np
import pandas as pd


def subscale_transform_long_format(data):
    """Transforms subscale columns into rows."""
    # Remove 'legacy_user_id' if it exists
    if "legacy_user_id" in data.columns:
        data = data.drop(columns=["legacy_user_id"])

    id_vars = data[
        [
            "activity_submission_id",
            "activity_flow_submission_id",
            "activity_scheduled_time_utc",
            "activity_start_time_utc",
            "activity_end_time_utc",
            "flag",
            "secret_user_id",
            "userId",
            "source_user_subject_id",
            "source_user_secret_id",
            "source_user_nickname",
            "source_user_relation",
            "source_user_tag",
            "target_user_subject_id",
            "target_user_secret_id",
            "target_user_nickname",
            "target_user_tag",
            "input_user_subject_id",
            "input_user_secret_id",
            "input_user_nickname",
            "activity_id",
            "activity_name",
            "activity_flow_id",
            "activity_flow_name",
            "version",
            "reviewing_id",
            "event_id",
            "timezone_offset",
        ]
    ].columns.to_list()

    value_vars = data.columns[data.columns.get_loc("timezone_offset") + 1 :].tolist()

    if not value_vars:  # Check if the list is empty
        print("No Subscale Scores Present")  # noqa: T201
        return None
    # Reshape the DataFrame using melt for columns after 'timezone_offset'
    reshaped_data = data.melt(
        id_vars=id_vars,  # Columns to keep as identifiers
        value_vars=value_vars,  # Columns to reshape
        var_name="item",  # New column to hold column names
        value_name="response",  # New column to hold corresponding values
    ).dropna(subset=["response"])

    # Classify score types
    reshaped_data["score_type"] = reshaped_data["item"].apply(
        lambda x: "finalscore"
        if x == "Final SubScale Score"
        else "finalscore_text"
        if x == "Optional text for Final SubScale Score"
        #'lookup' if x in subscale_names else
        else "lookup_text"
        if re.match(r"^Optional text for ", x)
        else "subscale"
    )

    # Transform item names based on score types
    def transform_item(row):
        if row["score_type"] == "finalscore":
            return "activity_score"
        if row["score_type"] == "finalscore_text":
            return "activity_score_lookup_text"
        if row["score_type"] == "lookup_text":
            return "subscale_lookup_text_" + row["item"].replace(
                "Optional text for ", ""
            )
        return "subscale_name_" + row["item"]

    reshaped_data["item"] = reshaped_data.apply(transform_item, axis=1)

    # Add additional computed columns
    reshaped_data = reshaped_data.drop(columns=["score_type"]).assign(
        item_id="", prompt="", options="", rawScore=""
    )

    # Prepare a subset of the original DataFrame for alignment
    subset_data = data[reshaped_data.columns.tolist()]

    # Combine the subset and reshaped DataFrame
    return pd.concat([subset_data, reshaped_data], axis=0, ignore_index=True)


# Define column list and response column name
mycolumn_list = [
    "userId",
    "secret_user_id",
    "source_user_secret_id",
    "target_user_secret_id",
    "input_user_secret_id",
    "activity_start_time_utc",
    "activity_end_time_utc",
    "activity_scheduled_time_utc",
    "activity_flow_id",
    "activity_flow_name",
    "activity_id",
    "activity_name",
    "event_id",
    "version",
]


def widen_data(data, column_list):
    """Transforms data into a wide format based on the specified column list."""
    # merge formatted response, values and scores created a single response field
    data = data.copy()
    data["merged_responses"] = (
        data["response_scores"]
        .combine_first(data["response_values"])
        .combine_first(data["formatted_response"])
    )

    # Convert datetime columns to string and handle NaT
    datetime_cols = data.select_dtypes(include=["datetime"]).columns
    data[datetime_cols] = data[datetime_cols].astype(str).replace("NaT", "")

    # Fill missing values in specified columns
    data[column_list] = data[column_list].fillna("")

    # Group by the column list and combine IDs
    answers = (
        data.groupby(column_list)["activity_submission_id"]
        .apply(lambda x: "|".join(x.astype(str)))
        .reset_index()
    )

    # Create combined column names
    data["combined_cols"] = (
        "activityName["
        + data["activity_name"]
        + "]_itemName["
        + data["item"].astype(str)
        + "]_itemId["
        + data["item_id"].astype(str)
        + "]"
    )
    data["combined_cols"] = np.where(
        data["combined_cols"].str.contains("_itemId[]", regex=False),
        data["combined_cols"].str.replace("_itemId[]", "", regex=False),
        data["combined_cols"],
    )

    # Select relevant columns for pivoting
    subset_columns = column_list + ["combined_cols", "merged_responses"]
    dat_subset = data[subset_columns]

    # Pivot the data into wide format
    dat_wide = pd.pivot_table(
        dat_subset,
        index=column_list,
        columns="combined_cols",
        values="merged_responses",
        aggfunc="last",
    ).reset_index()

    # Merge with the combined IDs
    return pd.merge(dat_wide, answers, on=column_list, how="outer")
