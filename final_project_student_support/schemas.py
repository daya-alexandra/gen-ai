"""Строгие схемы всех сообщений между этапами проекта."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Category = Literal["schedule", "documents", "tuition", "academic_leave", "technical"]
ToolName = Literal["search_rules", "get_contact", "calculate_days"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceQuote(StrictModel):
    quote: str = Field(min_length=2, max_length=400)
    meaning: str = Field(min_length=2, max_length=300)

    @field_validator("quote")
    @classmethod
    def one_line_quote(cls, value: str) -> str:
        value = value.strip()
        if "\n" in value or "\r" in value:
            raise ValueError("Цитата должна быть одной точной строкой из обращения")
        return value


class RequestExtraction(StrictModel):
    request_id: str = Field(pattern=r"^R\d{2}$")
    message_date: date
    category: Category
    intent: str = Field(min_length=5, max_length=350)
    urgency: int = Field(ge=1, le=5)
    contains_sensitive_data: bool
    requested_deadline: date | None = None
    evidence: list[EvidenceQuote] = Field(min_length=1, max_length=4)

    @field_validator("requested_deadline")
    @classmethod
    def deadline_not_before_message(cls, value: date | None, info):
        message_date = info.data.get("message_date")
        if value is not None and message_date is not None and value < message_date:
            raise ValueError("Запрошенный будущий срок не может быть раньше даты сообщения")
        return value


class ToolCall(StrictModel):
    name: ToolName
    arguments: dict[str, str]


class Plan(StrictModel):
    route: Category
    short_rationale: str = Field(min_length=5, max_length=350)
    tools: list[ToolCall] = Field(min_length=2, max_length=4)

    @field_validator("tools")
    @classmethod
    def mandatory_tools(cls, value: list[ToolCall]):
        names = [item.name for item in value]
        if names[0] != "search_rules":
            raise ValueError("Первым должен выполняться поиск по утверждённой базе")
        if "get_contact" not in names:
            raise ValueError("Маршрут должен включать проверяемый контакт")
        return value

    @model_validator(mode="after")
    def no_duplicate_tools(self):
        names = [item.name for item in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Один инструмент нельзя вызывать дважды в одном плане")
        return self


class Citation(StrictModel):
    source_id: str = Field(pattern=r"^[a-z_]+$")
    quote: str = Field(min_length=5, max_length=700)


class Resolution(StrictModel):
    answer: str = Field(min_length=20, max_length=2400)
    message_quotes: list[str] = Field(min_length=1, max_length=3)
    citations: list[Citation] = Field(min_length=1, max_length=5)
    abstained: bool
    escalation_needed: bool

    @field_validator("message_quotes")
    @classmethod
    def clean_message_quotes(cls, value: list[str]):
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("Пустые цитаты запрещены")
        return cleaned


class JudgeVerdict(StrictModel):
    correctness: int = Field(ge=0, le=4)
    groundedness: int = Field(ge=0, le=4)
    path_quality: int = Field(ge=0, le=2)
    approved: bool
    problems: list[str] = Field(max_length=5)
    short_reason: str = Field(min_length=5, max_length=500)

    @model_validator(mode="after")
    def approval_matches_scores(self):
        should_approve = self.correctness >= 3 and self.groundedness >= 3 and self.path_quality >= 1
        if self.approved != should_approve:
            raise ValueError("approved должен следовать из выставленных баллов")
        return self
