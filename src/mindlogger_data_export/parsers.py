"""Parse response string to dict."""

from datetime import date, time, timedelta
from typing import Any

import polars as pl
from lark import Lark, Transformer, v_args


class ResponseTransformer(Transformer):
    """Transform Lark parse tree to dict."""

    PIVOT_YEAR = 69
    MAX_TWO_DIGIT = 99

    DEFAULT_SCHEMA = {
        "type": None,
        "raw_value": None,
        "null_value": None,
        "value": None,
        "text": None,
        "file": None,
        "date": None,
        "time": None,
        "time_range": None,
        "geo": None,
        "matrix": None,
        "optional_text": None,
    }

    @v_args(inline=True)
    def start(self, value, optional_text=None):  # noqa: D102
        if optional_text:
            value = value | optional_text
        return self.DEFAULT_SCHEMA | value

    def padded_two_digit(self, items):  # noqa: D102
        return int("".join(items))

    def year(self, items):  # noqa: D102
        iyr = int("".join(items))
        conversion = (
            2000 if iyr < self.PIVOT_YEAR else 1900 if iyr <= self.MAX_TWO_DIGIT else 0
        )
        return iyr + conversion

    @v_args(inline=True)
    def date_resp(self, day, month, year):  # noqa: D102
        return {"type": "date", "date": date(year, month, day)}

    @v_args(inline=True)
    def time(self, hour, minute):  # noqa: D102
        return time(hour, minute)

    @v_args(inline=True)
    def time_resp(self, time):  # noqa: D102
        return {"type": "time", "time": time}

    @v_args(inline=True)
    def time_range_resp(self, from_time, to_time):  # noqa: D102
        return {
            "type": "time_range",
            "time_range": timedelta(
                hours=to_time.hour - from_time.hour,
                minutes=to_time.minute - from_time.minute,
            ),
        }

    @v_args(inline=True)
    def geo_resp(self, latitude, longitude):  # noqa: D102
        return {"type": "geo", "geo": {"latitude": latitude, "longitude": longitude}}

    @v_args(inline=True)
    def text_resp(self, text):  # noqa: D102
        return {"type": "text", "text": text.value}

    @v_args(inline=True)
    def optional_text(self, text):  # noqa: D102
        return {"optional_text": text.value}

    @v_args(inline=True)
    def null_resp(self):  # noqa: D102
        return {"type": "null_value", "null_value": True}

    @v_args(inline=True)
    def multivalue_resp(self, vlist):  # noqa: D102
        return {"type": "value", "value": vlist}

    @v_args(inline=True)
    def row_kv(self, key, row_value):  # noqa: D102
        return {"row": key.value, "value": row_value.value}

    def matrix_resp(self, items):  # noqa: D102
        return {"type": "matrix", "matrix": items}

    @v_args(inline=True)
    def row_kvv(self, key, ilist):  # noqa: D102
        return {"row": key.value, "value": ilist}

    def vlist(self, items):  # noqa: D102
        return [i.value for i in items]

    def ilist(self, items):  # noqa: D102
        return items

    @v_args(inline=True)
    def file_resp(self, path):  # noqa: D102
        return {"type": "file", "file": path.value}

    @v_args(inline=True)
    def raw_value_resp(self, value):  # noqa: D102
        return {"type": "raw_value", "raw_value": value.value}

    SIGNED_FLOAT = float
    INT = int


class ResponseParser:
    """Parse response string to dict using Lark grammar."""

    RESPONSE_GRAMMAR = r"""
    start: text_resp | null_resp | file_resp | raw_value_resp | matrix_resp
        | ( ( multivalue_resp | date_resp | time_resp | time_range_resp | geo_resp ) [ optional_text ] )

    // Matches text response in "text: <text>" format, including multi-line text.
    text_resp.10: "text:" _WSI _text
    _text: /.+/s

    optional_text: _WSI "| text:" _WSI _text

    // Matches null response in "value: null" format.
    null_resp.20: "value: null"

    // Matches single or multiple values in "value: <value>, <value>, ..." format.
    multivalue_resp.10: "value:" _WSI vlist

    // Matches date response in "date: <month>/<day>/<year>" format.
    date_resp.10: "date:" _WSI padded_two_digit "/" padded_two_digit "/" year
    padded_two_digit: DIGIT ~ 1..2
    year: (DIGIT ~ 2) | (DIGIT ~ 4)

    // Matches time response.
    time_resp.10: "time:" _WSI time

    // Matches time range with from and to times.
    time_range_resp.10: "time_range:" _WSI "from" _WSI time _WSI "/" _WSI "to" _WSI time

    // Matches time in "hr <hour> min <minute>" format.
    time: "("? _hour ","? _WSI _minute ")"?
    _hour: "hr" _WSI padded_two_digit
    _minute: "min" _WSI padded_two_digit

    // Matches geo coordinates with latitude and longitude.
    geo_resp.10: "geo:" _WSI _latitude _WSI "/"? _WSI? _longitude
    _latitude: "lat" _WSI "("? SIGNED_FLOAT ")"?
    _longitude: "long" _WSI "("? SIGNED_FLOAT ")"?

    // Matches single multiple rows with single key-value pair per row.
    // row_resp.5: _sep{row_kv, _NL}
    // row_kv: _value ":" _WSI? _row_value
    // _row_value: /[^,\n]+/

    // Matches multiple rows with key and list of values per row.
    matrix_resp.5: (row_kvv _NL?)+
    row_kvv: _value ":" _WSI? vlist

    // Matches file path by ensuring string contains at least one slash and a 2-4 character extension.
    file_resp.5: /.+[-\/].+\.\w{3,4}/

    // Lowest priority catch-all rule.
    raw_value_resp.0: /.+/s

    // list of comma-separated values
    ilist: _sep{INT, _CSV}
    vlist: _sep{_value, _CSV}

    // Value is any non-empty alphanumeric string
    _value: /[.\w]+/
    _CSV: "," _WSI
    _sep{x, sep}: x (sep x)*

    %import common.SIGNED_FLOAT
    %import common.WS_INLINE -> _WSI
    %import common.DIGIT
    %import common.INT
    %import common.NEWLINE -> _NL

    // Disregard commas in parse tree
    %ignore ","
    """

    def __init__(self):
        """Initialize ResponseParser."""
        self._transformer = ResponseTransformer()
        self._parser = Lark(self.RESPONSE_GRAMMAR)
        self._types = self._transformer.DEFAULT_SCHEMA.keys()

    @property
    def types(self):
        """Return response types."""
        return self._types

    @property
    def datatype(self):
        """Return Polars schema for response types."""
        return pl.Struct(
            {
                "type": pl.String,
                "raw_value": pl.String,
                "null_value": pl.Boolean,
                "value": pl.List(pl.String),
                "text": pl.String,
                "file": pl.String,
                "date": date,
                "time": time,
                "time_range": timedelta,
                "geo": pl.Struct({"latitude": pl.Float64, "longitude": pl.Float64}),
                "matrix": pl.List(
                    pl.Struct({"row": pl.String, "value": pl.List(pl.String)})
                ),
                "optional_text": pl.String,
            }
        )

    def parse(self, response: str) -> dict[str, Any]:
        """Parse response string to dict."""
        return self._transformer.transform(self._parser.parse(response))

    def selection_parser(self, field_name: str):
        """Return parser for selection fields."""

        def _parser(response: str) -> dict[str, Any]:
            return self._transformer.transform(self._parser.parse(response)).get(
                field_name
            )

        return _parser


class OptionsTransformer(Transformer):
    """Transform Lark parse tree to dict."""

    def scored_options(self, options):  # noqa: D102
        return options

    @v_args(inline=True)
    def scored_option(self, option, score):  # noqa: D102
        return {"name": option["name"], "value": option["value"], "score": score}

    @v_args(inline=True)
    def score(self, score):  # noqa: D102
        return int(score)

    def value_options(self, options):  # noqa: D102
        return options

    @v_args(inline=True)
    def option(self, name, value):  # noqa: D102
        return {"name": name, "value": value, "score": None}

    @v_args(inline=True)
    def min_max_range(self, minimum, maximum):  # noqa: D102
        return [
            {"name": n, "value": n, "score": n} for n in range(minimum, maximum + 1)
        ]

    @v_args(inline=True)
    def max_min_range(self, maximum, minimum):  # noqa: D102
        return self.min_max_range(minimum, maximum)

    @v_args(inline=True)
    def name(self, name):  # noqa: D102
        return name.value

    INT = int
    SIGNED_INT = int


class OptionsParser:
    """Parse options string to dict using Lark grammar."""

    OPTIONS_GRAMMAR = r"""
    ?start: scored_options | _range | value_options

    scored_options.10: _sep{scored_option, _CSV}
    scored_option: option _WSI score
    score: "(" "score:" _WSI? SIGNED_INT _WSI? ")"

    _range.20: max_min_range | min_max_range
    max_min_range.20: "Max:" _WSI INT "," _WSI "Min:" _WSI INT
    min_max_range.20: "Min:" _WSI INT "," _WSI "Max:" _WSI INT

    // Matches option in "<name>: <value>" format.
    value_options.10: _sep{option, _CSV}
    option: name ":" _WSI? INT

    // Value is any non-empty alphanumeric string
    name: /\w+/
    _CSV: "," _WSI
    _sep{x, sep}: x (sep x)*

    %import common.INT
    %import common.SIGNED_INT
    %import common.WS_INLINE -> _WSI

    // Disregard commas in parse tree
    %ignore ","
    """

    def __init__(self):
        """Initialize ResponseParser."""
        self._transformer = OptionsTransformer()
        self._parser = Lark(self.OPTIONS_GRAMMAR)

    def parse(self, response: str) -> dict[str, Any]:
        """Parse response string to dict."""
        return self._transformer.transform(self._parser.parse(response))
