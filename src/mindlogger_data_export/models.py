"""Data models for Mindlogger data export."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

LOG = logging.getLogger(__name__)


class UserType(StrEnum):
    """Enumeration of Mindlogger user types."""

    SOURCE = "source_user"
    TARGET = "target_user"
    INPUT = "input_user"
    ACCOUNT = "account_user"

    @classmethod
    def columns(cls, user_type: UserType) -> list[str]:
        """Get list of user type columns."""
        match user_type:
            case cls.SOURCE:
                return [
                    "source_id",
                    "source_secret_id",
                    "source_nickname",
                    "source_relation",
                    "source_tag",
                ]
            case cls.TARGET:
                return [
                    "target_id",
                    "target_secret_id",
                    "target_nickname",
                    "target_tag",
                ]
            case cls.INPUT:
                return [
                    "input_id",
                    "input_secret_id",
                    "input_nickname",
                ]
            case cls.ACCOUNT:
                return ["userId", "secret_user_id"]
        return []


@dataclass
class MindloggerResponseOption:
    """Data model of a Mindlogger response option."""

    name: str
    value: int
    score: int | None


@dataclass
class MindloggerUser:
    """Data model of a Mindlogger user."""

    user_type: UserType
    id: str
    secret_id: str
    nickname: str | None = None
    tag: str | None = None
    relation: str | None = None

    @classmethod
    def from_struct(cls, user_type: UserType, struct: dict[str, str]) -> MindloggerUser:
        """Create MindloggerUser object from source struct."""
        return cls(user_type, **struct)

    @classmethod
    def from_struct_factory(
        cls, user_type: UserType
    ) -> Callable[[dict[str, str]], MindloggerUser]:
        """Create MindloggerUser object from struct."""

        def _from_struct_partial(struct: dict[str, str]) -> MindloggerUser:
            """Partial function to create MindloggerUser from struct."""
            return cls.from_struct(user_type, struct)

        return _from_struct_partial
