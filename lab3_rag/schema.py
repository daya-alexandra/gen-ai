from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunk_id: str
    quote: str = Field(min_length=8)


class RAGAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['answered','insufficient_context']
    answer: str = Field(min_length=12)
    confidence: float = Field(ge=0,le=1)
    citations: list[Citation] = Field(max_length=5)

    @model_validator(mode='after')
    def supported_answer(self):
        if self.status == 'answered' and not self.citations:
            raise ValueError('Ответ требует хотя бы одну цитату из контекста')
        if self.status == 'insufficient_context' and self.confidence > .5:
            raise ValueError('При недостаточном контексте confidence не выше 0.5')
        if len({(c.chunk_id,c.quote) for c in self.citations}) != len(self.citations):
            raise ValueError('Не повторяй одинаковые цитаты')
        return self
