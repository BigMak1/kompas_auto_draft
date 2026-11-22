"""Структурированный паспорт: наблюдения чертежа, а не инструкции построения CAD."""

from typing import Annotated, Literal, Self

from pydantic import Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from src.agent.schemas.base import StrictModel

Identifier = Annotated[str, Field(min_length=1, max_length=64)]
Text = Annotated[str, Field(min_length=1, max_length=2000)]
Status = Literal["observed", "hypothesis", "unresolved"]


class View(StrictModel):
    id: Identifier
    label: Text
    kind: Literal["view", "section", "detail", "table", "other"]
    description: Text


class Coordinates(StrictModel):
    origin: Text
    axis_z: Text
    front_view_id: Identifier | None
    status: Status


class Feature(StrictModel):
    id: Identifier
    kind: Text
    description: Text
    view_ids: Annotated[list[Identifier], Field(max_length=32)]
    status: Status


class Dimension(StrictModel):
    id: Identifier
    name: Text
    raw: Text | None
    value: Annotated[float, Field(allow_inf_nan=False)] | None
    unit: Literal["mm", "degree", "count", "dimensionless"] | None = Field(
        description="Для числа обязательна единица: mm, degree, count или dimensionless для коэффициента."
    )
    view_id: Identifier | None
    feature_id: Identifier | None = Field(
        description="ID элемента; null при неизвестной привязке. Общие примечания идут в general_notes."
    )
    reading_status: Status
    binding_status: Status

    @model_validator(mode="after")
    def evidence(self) -> Self:
        errors = []

        def issue(field: str, message: str) -> None:
            errors.append({"type": PydanticCustomError("passport_evidence", message), "loc": (field,)})

        if self.reading_status == "unresolved" and self.value is not None:
            issue("value", "Непрочитанный размер: value должен быть null")
        if self.reading_status == "observed" and (not self.raw or not self.view_id):
            issue("reading_status", "Наблюдаемый размер требует исходной надписи raw и view_id")
        if self.value is not None and self.unit is None:
            issue("unit", "Число требует единицы; для безразмерного коэффициента — dimensionless")
        if self.binding_status == "unresolved" and self.feature_id is not None:
            issue("feature_id", "Неизвестный владелец размера: feature_id должен быть null")
        if self.binding_status == "observed" and self.feature_id is None:
            issue("feature_id", "Наблюдаемая привязка требует feature_id; общее примечание — general_notes")
        if errors:
            raise ValidationError.from_exception_data(type(self).__name__, errors)
        return self


class Relation(StrictModel):
    feature_ids: Annotated[list[Identifier], Field(min_length=2, max_length=16)]
    description: Text
    status: Status


class GeneralNote(StrictModel):
    raw: Text = Field(description="Общее примечание как на листе: например, неуказанные допуски.")
    view_id: Identifier | None
    status: Status


class Passport(StrictModel):
    schema_version: Literal["drawing-passport-2"]
    views: Annotated[list[View], Field(min_length=1, max_length=32)]
    coordinates: Coordinates
    features: Annotated[list[Feature], Field(max_length=80)]
    dimensions: Annotated[list[Dimension], Field(max_length=160)]
    relations: Annotated[list[Relation], Field(max_length=80)]
    general_notes: Annotated[list[GeneralNote], Field(max_length=80)]
    unresolved: Annotated[list[Text], Field(max_length=80)]

    @model_validator(mode="after")
    def references(self) -> Self:
        groups = (self.views, self.features, self.dimensions)
        ids = [item.id for group in groups for item in group]
        if len(ids) != len(set(ids)):
            raise PydanticCustomError("passport_reference", "ID записей паспорта должны быть уникальны")
        views = {item.id for item in self.views}
        features = {item.id for item in self.features}
        errors = []

        def check(ref: str | None, known: set[str], location: tuple) -> None:
            if ref is not None and ref not in known:
                errors.append(
                    {
                        "type": PydanticCustomError(
                            "passport_reference", "Отсутствующий вид или элемент: {ref}", {"ref": ref}
                        ),
                        "loc": location,
                    }
                )

        check(self.coordinates.front_view_id, views, ("coordinates", "front_view_id"))
        for i, feature in enumerate(self.features):
            for j, ref in enumerate(feature.view_ids):
                check(ref, views, ("features", i, "view_ids", j))
        for i, dimension in enumerate(self.dimensions):
            check(dimension.view_id, views, ("dimensions", i, "view_id"))
            check(dimension.feature_id, features, ("dimensions", i, "feature_id"))
        for i, relation in enumerate(self.relations):
            for j, ref in enumerate(relation.feature_ids):
                check(ref, features, ("relations", i, "feature_ids", j))
        for i, note in enumerate(self.general_notes):
            check(note.view_id, views, ("general_notes", i, "view_id"))
        if errors:
            raise ValidationError.from_exception_data(type(self).__name__, errors)
        return self
