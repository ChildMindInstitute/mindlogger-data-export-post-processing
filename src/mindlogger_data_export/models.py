"""Data models for Mindlogger data export."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

LOG = logging.getLogger(__name__)


class UserType(StrEnum):
    """Enumeration of Mindlogger user types."""

    SOURCE = "source"
    TARGET = "target"
    INPUT = "input"
    ACCOUNT = "account"

    @classmethod
    def columns(cls, user_type: UserType) -> list[str]:
        """Get list of user type columns."""
        match user_type:
            case cls.SOURCE:
                return [
                    "source_user_subject_id",
                    "source_user_secret_id",
                    "source_user_nickname",
                    "source_user_relation",
                    "source_user_tag",
                ]
            case cls.TARGET:
                return [
                    "target_user_subject_id",
                    "target_user_secret_id",
                    "target_user_nickname",
                    "target_user_tag",
                ]
            case cls.INPUT:
                return [
                    "input_user_subject_id",
                    "input_user_secret_id",
                    "input_user_nickname",
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
    subject_id: str
    secret_id: str
    nickname: str | None = None
    tag: str | None = None
    relation: str | None = None

    @classmethod
    def from_source_struct(cls, struct: dict[str, str]) -> MindloggerUser:
        """Create MindloggerUser object from source struct."""
        return cls(
            UserType.SOURCE,
            struct["source_user_subject_id"],
            struct["source_user_secret_id"],
            struct["source_user_nickname"],
            struct["source_user_relation"],
            struct["source_user_tag"],
        )

    @classmethod
    def from_target_struct(cls, struct: dict[str, str]) -> MindloggerUser:
        """Create MindloggerUser object from target struct."""
        return cls(
            UserType.TARGET,
            struct["target_user_subject_id"],
            struct["target_user_secret_id"],
            struct["target_user_nickname"],
            struct["target_user_tag"],
        )

    @classmethod
    def from_input_struct(cls, struct: dict[str, str]) -> MindloggerUser:
        """Create MindloggerUser object from input struct."""
        return cls(
            UserType.INPUT,
            struct["input_user_subject_id"],
            struct["input_user_secret_id"],
            struct["input_user_nickname"],
        )

    @classmethod
    def from_account_struct(cls, struct: dict[str, str]) -> MindloggerUser:
        """Create MindloggerUser object from account struct."""
        return cls(
            UserType.ACCOUNT,
            struct["userId"],
            struct["secret_user_id"],
        )

    @classmethod
    def from_struct_factory(
        cls, user_type: UserType
    ) -> Callable[[dict[str, str]], MindloggerUser]:
        """Create MindloggerUser object from struct."""
        match user_type:
            case UserType.SOURCE:
                return cls.from_source_struct
            case UserType.TARGET:
                return cls.from_target_struct
            case UserType.INPUT:
                return cls.from_input_struct
            case UserType.ACCOUNT:
                return cls.from_account_struct
