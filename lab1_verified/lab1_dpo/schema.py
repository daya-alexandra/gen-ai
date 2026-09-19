"""Схемы ДЗ № 1. Все заявки — синтетические, возраст фиксируется на 2026 год."""

from typing import Literal, get_args
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

REFERENCE_YEAR = 2026
CITIES = (
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Самара", "Краснодар", "Красноярск", "Воронеж",
)
Speciality = Literal[
    "Инженер", "Программист", "Учитель", "Экономист", "Бухгалтер",
    "Менеджер по персоналу", "Маркетолог", "Специалист по логистике",
]
Course = Literal[
    "Анализ данных", "Python для автоматизации", "Управление проектами",
    "Цифровой маркетинг", "Финансовый анализ", "Управление персоналом",
]
SPECIALITIES = get_args(Speciality)
COURSES = get_args(Course)
CUSTOM_ERROR_TYPES = {
    "city_not_allowed", "invalid_full_name", "experience_exceeds_age",
    "graduation_too_early",
}


class Address(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    city: str = Field(description="Один из десяти разрешённых городов")
    district: str = Field(min_length=2, max_length=80)

    @field_validator("city")
    @classmethod
    def check_city(cls, value: str) -> str:
        if value not in CITIES:
            raise PydanticCustomError(
                "city_not_allowed", "Город должен входить в список: {cities}",
                {"cities": ", ".join(CITIES)},
            )
        return value


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    full_name: str = Field(min_length=5, max_length=120)
    age: int = Field(ge=22, le=65, strict=True)
    address: Address
    speciality: Speciality
    desired_course: Course
    years_of_experience: int = Field(ge=0, le=40, strict=True)
    graduation_year: int = Field(ge=1980, le=2024, strict=True)

    @field_validator("full_name")
    @classmethod
    def check_full_name(cls, value: str) -> str:
        words = value.split()
        if len(words) != 3 or not all(
            re.fullmatch(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", word) for word in words
        ):
            raise PydanticCustomError(
                "invalid_full_name", "Нужны фамилия, имя и отчество кириллицей"
            )
        return " ".join(words)

    @field_validator("years_of_experience")
    @classmethod
    def check_experience(cls, value: int, info: ValidationInfo) -> int:
        age = info.data.get("age")
        # Учебное допущение: учитывается работа с 18 лет, включая работу до выпуска.
        if age is not None and value > age - 18:
            raise PydanticCustomError(
                "experience_exceeds_age", "Стаж не должен превышать возраст минус 18"
            )
        return value

    @field_validator("graduation_year")
    @classmethod
    def check_graduation(cls, value: int, info: ValidationInfo) -> int:
        age = info.data.get("age")
        # Нет дат рождения: возраст рассматривается как возраст на конец 2026 года.
        # Для этого учебного датасета принимаем возраст выпуска не менее 20 лет.
        if age is not None and value < REFERENCE_YEAR - age + 20:
            raise PydanticCustomError(
                "graduation_too_early", "По возрасту заявителя выпуск получился раньше 20 лет"
            )
        return value


def flatten(app: Application) -> dict:
    """Адрес в CSV представляется двумя обычными столбцами."""
    data = app.model_dump()
    address = data.pop("address")
    return {"full_name": data.pop("full_name"), "age": data.pop("age"), **address, **data}


def from_flat(row: dict) -> Application:
    """Повторная проверка CSV; преобразование целых чисел выполняется явно."""
    data = dict(row)
    data["address"] = {"city": data.pop("city"), "district": data.pop("district")}
    for name in ("age", "years_of_experience", "graduation_year"):
        data[name] = int(data[name])
    return Application.model_validate(data)
