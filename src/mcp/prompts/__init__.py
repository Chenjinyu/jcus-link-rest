"""Prompt templates and MCP prompt registrations."""

from .resume_prompts import (
    build_analysis_prompt,
    build_job_requirements_prompt,
    build_resume_from_source_prompt,
    build_resume_prompt,
    register_prompts,
)

__all__ = [
    "build_analysis_prompt",
    "build_job_requirements_prompt",
    "build_resume_from_source_prompt",
    "build_resume_prompt",
    "register_prompts",
]
