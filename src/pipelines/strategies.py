"""Реестр конкретных стратегий пайплайнов (direct, passport_png, passport_only, notes_png)."""

from src.pipelines.base import Pipeline, PipelineContext, execute_two_or_one_stage
from src.schemas.passport import Passport


class DirectPipeline:
    name = "direct"
    required_prompts = ("builder",)
    reading_stage = None

    async def execute(self, ctx: PipelineContext) -> dict:
        return await execute_two_or_one_stage(ctx, builder_sees_image=True)


class PassportPngPipeline:
    name = "passport_png"
    required_prompts = ("passport", "builder")
    reading_stage = "passport"

    async def execute(self, ctx: PipelineContext) -> dict:
        return await execute_two_or_one_stage(
            ctx,
            reading_stage="passport",
            reading_prompt_role="passport",
            reading_output_type=Passport,
            reading_filename="passport.json",
            builder_sees_image=True,
        )


class PassportOnlyPipeline:
    name = "passport_only"
    required_prompts = ("passport", "builder")
    reading_stage = "passport"

    async def execute(self, ctx: PipelineContext) -> dict:
        return await execute_two_or_one_stage(
            ctx,
            reading_stage="passport",
            reading_prompt_role="passport",
            reading_output_type=Passport,
            reading_filename="passport.json",
            builder_sees_image=False,
        )


class NotesPngPipeline:
    name = "notes_png"
    required_prompts = ("notes", "builder")
    reading_stage = "notes"

    async def execute(self, ctx: PipelineContext) -> dict:
        return await execute_two_or_one_stage(
            ctx,
            reading_stage="notes",
            reading_prompt_role="notes",
            reading_output_type=None,
            reading_filename="notes.md",
            builder_sees_image=True,
        )


PIPELINES: dict[str, Pipeline] = {
    pipeline.name: pipeline
    for pipeline in (
        DirectPipeline(),
        PassportPngPipeline(),
        PassportOnlyPipeline(),
        NotesPngPipeline(),
    )
}


def get_pipeline(name: str) -> Pipeline:
    if name not in PIPELINES:
        known = ", ".join(sorted(PIPELINES))
        raise ValueError(f"Неизвестный pipeline {name!r}. Доступны: {known}")
    return PIPELINES[name]
