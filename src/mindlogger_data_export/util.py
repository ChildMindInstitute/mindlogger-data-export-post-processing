"""Utilities carried over from original scripts."""

import re
from typing import Any

import numpy as np
import pandas as pd


def val_score_mapping(data: pd.DataFrame) -> pd.Series:
    """Parse scoring options and map response values to scores."""
    response_scores = []  # List to store results

    for i in range(len(data["response"])):
        options = data["options"][i]
        response = data["response"][i]
        clean_response = (
            re.sub(r"value: |geo: ", "", response)
            if isinstance(response, str)
            else np.nan
        )

        # Ensure 'options' and 'response' are valid strings
        if not isinstance(options, str) or not isinstance(response, str):
            response_scores.append(clean_response)  # Append NaN for invalid rows
            continue

        if re.search(r"score: ", options):
            split_options = options.strip().split("),")
            split_response = response.strip().split(": ")[1].split(",")
            scores = {}

            for j in split_options:
                if "(score" in j:  # Ensure the string contains the expected structure
                    val_parts = j.split("(score")
                    if len(val_parts) == 2 and ": " in val_parts[0]:  # noqa: PLR2004
                        val_num = val_parts[0].split(": ")[1].strip()
                        score_num = val_parts[1].split(": ")[1].strip(" )")
                        scores[val_num] = score_num

            response_score_mapping = {
                resp.strip(): scores.get(resp.strip(), "N/A")  # Handle missing mappings
                for resp in split_response
            }
            response_scores.append(", ".join(response_score_mapping.values()))
        else:
            response_scores.append(
                clean_response
            )  # Append NaN if no valid scores are found

    return pd.Series(response_scores)


def clean_time_range(df: pd.DataFrame, column_name: str) -> list[str | Any]:
    """Cleanup and split time range in the response."""
    cleaned = []
    for i in range(len(df[column_name])):
        if pd.notna(df[column_name][i]) and str(df[column_name][i]).startswith(
            "time_range"
        ):
            t = re.sub(r"[a-zA-Z\s+(\)_:]", "", df[column_name][i])
            t = t.replace(",", ":")
            if re.search(r"^[0-9]:", t):  # 9,30/12,30
                ttemp = "0" + t
            elif re.search(r":[0-9]$", t):  # 12,5/12,30
                ttemp = t.replace(":", ":0")
            else:
                ttemp = t
            thm = ttemp
        else:
            ttemp = df[column_name][i]
            thm = ttemp
        cleaned.append(thm)
    return cleaned


def add_timezone_offset(mydata: pd.DataFrame, columntoaddto: str) -> pd.Series:
    """Add timezone offset to a column."""
    col_values = pd.to_numeric(mydata[columntoaddto], errors="coerce")
    timezone_offsets = pd.to_numeric(mydata["timezone_offset"], errors="coerce")
    timezone_offsets = timezone_offsets.fillna(0)  # Replace NaN with 0 for offsets
    return col_values + (timezone_offsets * 60 * 1000)


# df['start_Time'] = add_timezone_offset(df, 'activity_start_time')  # noqa: ERA001
# df['end_Time'] = add_timezone_offset(df, 'activity_end_time')  # noqa: ERA001
# df['schedule_Time'] = add_timezone_offset(df, 'activity_scheduled_time')  # noqa: ERA001
