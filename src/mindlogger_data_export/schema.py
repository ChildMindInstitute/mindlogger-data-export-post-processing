"""Schema definitions."""

from typing import Any

import polars as pl

USER_SCHEMA = pl.Schema(
    {
        "id": pl.String,
        "secret_id": pl.String,
        "nickname": pl.String,
        "relation": pl.String,
        "tag": pl.String,
    }
)


##### Full Set of Response Types / Labels #####
# From: https://github.com/ChildMindInstitute/mindlogger-admin/blob/develop/src/shared/consts.tsx#L107
#   ABTrails = 'ABTrails',
#   Audio = 'audio',
#   AudioPlayer = 'audioPlayer',
#   Date = 'date',
#   Drawing = 'drawing',
#   Flanker = 'flanker',
#   Geolocation = 'geolocation',
#   Message = 'message',
#   MultipleSelection = 'multiSelect',
#   MultipleSelectionPerRow = 'multiSelectRows',
#   NumberSelection = 'numberSelect',
#   ParagraphText = 'paragraphText',
#   Photo = 'photo',
#   PhrasalTemplate = 'phrasalTemplate',
#   RequestHealthRecordData = 'requestHealthRecordData',
#   SingleSelection = 'singleSelect',
#   SingleSelectionPerRow = 'singleSelectRows',
#   Slider = 'slider',
#   SliderRows = 'sliderRows',
#   StabilityTracker = 'stabilityTracker',
#   Text = 'text',
#   Time = 'time',
#   TimeRange = 'timeRange',
#   TouchPractice = 'touchPractice',
#   TouchTest = 'touchTest',
#   Unity = 'unity',
#   Video = 'video',
##############################################


RESPONSE_VALUE_SCHEMA = pl.Struct(
    {
        "type": pl.String,
        "raw_value": pl.String,
        "null_value": pl.Boolean,
        "single_value": pl.String,
        "value": pl.List(pl.String),
        "text": pl.String,
        "file": pl.String,
        "date": pl.Date,
        "time": pl.Time,
        "time_range": pl.Duration,
        "geo": pl.Struct({"latitude": pl.Float64, "longitude": pl.Float64}),
        "matrix": pl.List(pl.Struct({"row": pl.String, "value": pl.List(pl.String)})),
        "optional_text": pl.String,
    }
)

RESPONSE_VALUE_DICT_SCHEMA: dict[str, Any] = {
    f.name: None for f in RESPONSE_VALUE_SCHEMA.fields
}

RESPONSE_SCHEMA = pl.Schema(
    {
        "status": pl.String,
        "value": RESPONSE_VALUE_SCHEMA,
        "raw_response": pl.String,
        "raw_score": pl.String,
    }
)

RESPONSE_OPTIONS_SCHEMA = pl.List(
    pl.Struct({"name": pl.String, "value": pl.Int64, "score": pl.Int64})
)

ITEM_SCHEMA = pl.Schema(
    {
        "id": pl.String,
        "name": pl.String,
        "prompt": pl.String,
        "type": pl.String,
        "raw_options": pl.String,
        "response_options": RESPONSE_OPTIONS_SCHEMA,
    }
)
ACTIVITY_FLOW_SCHEMA = pl.Schema(
    {"id": pl.String, "name": pl.String, "submission_id": pl.String}
)
ACTIVITY_SCHEMA = pl.Schema({"id": pl.String, "name": pl.String})
ACTIVITY_SUBMISSION_SCHEMA = pl.Schema({"id": pl.String, "review_id": pl.String})
ACTIVITY_TIME_SCHEMA = pl.Schema(
    {"start_time": pl.Datetime("ms", "UTC"), "end_time": pl.Datetime("ms", "UTC")}
)
ACTIVITY_SCHEDULE_SCHEMA = pl.Schema(
    {"id": pl.String, "history_id": pl.String, "start_time": pl.Datetime("ms", "UTC")}
)

INTERNAL_SCHEMA = pl.Schema(
    {
        "applet_version": pl.String,
        "utc_timezone_offset": pl.Duration("ms"),
        "target_user": pl.Struct(USER_SCHEMA),
        "source_user": pl.Struct(USER_SCHEMA),
        "input_user": pl.Struct(USER_SCHEMA),
        "account_user": pl.Struct(USER_SCHEMA),
        "activity_flow": pl.Struct(ACTIVITY_FLOW_SCHEMA),
        "activity": pl.Struct(ACTIVITY_SCHEMA),
        "activity_time": pl.Struct(ACTIVITY_TIME_SCHEMA),
        "activity_schedule": pl.Struct(ACTIVITY_SCHEDULE_SCHEMA),
        "item": pl.Struct(ITEM_SCHEMA),
        "response": pl.Struct(RESPONSE_SCHEMA),
    }
)
