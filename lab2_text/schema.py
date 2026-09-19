"""Модели для анализа синтетических отзывов о планировщике задач."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Aspect = Literal['производительность', 'дизайн', 'поддержка', 'цена', 'реклама', 'надёжность']
ASPECTS = list(Aspect.__args__)


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SourceReview(Model):
    review_id: str = Field(pattern=r'^R\d{2}$')
    text: str = Field(min_length=40)


class Evidence(Model):
    review_id: str = Field(pattern=r'^R\d{2}$')
    quote: str = Field(min_length=8)


class Claim(Model):
    text: str = Field(min_length=8)
    evidence: list[Evidence] = Field(min_length=1, max_length=10)


class Review(Model):
    review_id: str = Field(pattern=r'^R\d{2}$')
    author: Optional[str] = None
    published_at: Optional[date] = None
    rating: Optional[Literal[1, 2, 3, 4, 5]] = None
    platform: Literal['Android', 'iOS', 'не указана']
    app_version: Optional[str] = None
    issues: list[Claim]
    praise: list[Claim]

    @field_validator('published_at')
    @classmethod
    def no_future_review(cls, value):
        if value is not None and value > date.today():
            raise ValueError('Дата отзыва не может быть позже сегодняшней')
        return value


class Reviews(Model):
    reviews: list[Review] = Field(min_length=1)


class AspectScore(Model):
    aspect: Aspect
    score: Optional[int] = Field(default=None, ge=-2, le=2, strict=True)
    explanation: str = Field(min_length=8)
    evidence: list[Evidence]

    @model_validator(mode='after')
    def unmentioned_is_not_neutral(self):
        if (self.score is None) != (len(self.evidence) == 0):
            raise ValueError('Неупомянутый аспект: score=null и evidence=[]; оценка требует цитаты')
        return self


class ReviewAspects(Model):
    review_id: str = Field(pattern=r'^R\d{2}$')
    aspects: list[AspectScore] = Field(min_length=6, max_length=6)

    @model_validator(mode='after')
    def six_different_aspects(self):
        if sorted(x.aspect for x in self.aspects) != sorted(ASPECTS):
            raise ValueError('Нужны все шесть аспектов ровно по одному разу')
        return self


class Aspects(Model):
    reviews: list[ReviewAspects] = Field(min_length=1)


class MapSummary(Model):
    source_ids: list[str] = Field(min_length=1)
    facts: list[Claim] = Field(min_length=1, max_length=8)


class Action(Claim):
    action_id: str = Field(pattern=r'^A\d+$')
    priority: Literal['высокий', 'средний', 'низкий']


class Summary(Model):
    title: str = Field(min_length=8)
    findings: list[Claim] = Field(min_length=3, max_length=7)
    action_items: list[Action] = Field(min_length=2, max_length=5)
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_actions(self):
        ids = [x.action_id for x in self.action_items]
        if len(ids) != len(set(ids)):
            raise ValueError('action_id должны быть уникальны')
        return self


class ActionVerdict(Model):
    action_id: str
    support: Literal['supported', 'weakly_supported', 'not_supported']
    explanation: str = Field(min_length=10)
    evidence: list[Evidence]


class Distortion(Model):
    original: Evidence
    mistaken_text: str = Field(min_length=8)
    correction: str = Field(min_length=8)


class JudgeReport(Model):
    faithfulness: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    actionability: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    verdicts: list[ActionVerdict]
    distortions: list[Distortion]
    feedback: list[str]

    @model_validator(mode='after')
    def consistent_score(self):
        expected = .4 * self.faithfulness + .35 * self.coverage + .25 * self.actionability
        if abs(self.overall_score - expected) > .015:
            raise ValueError('overall_score = 0.4*faithfulness + 0.35*coverage + 0.25*actionability')
        if any(v.support == 'supported' and not v.evidence for v in self.verdicts):
            raise ValueError('supported требует хотя бы одну цитату')
        return self


class DiscoveredAspect(Model):
    name: str = Field(pattern=r'^[a-z][a-z0-9_]+$')
    description: str = Field(min_length=10)
    related_fixed_aspect: Optional[Aspect] = None
    evidence: list[Evidence] = Field(min_length=1)


class Discovery(Model):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=10)

    @model_validator(mode='after')
    def unique_names(self):
        names = [x.name for x in self.aspects]
        if len(names) != len(set(names)):
            raise ValueError('Имена обнаруженных аспектов не должны повторяться')
        return self
