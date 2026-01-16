from __future__ import annotations

import json
from typing import Any, Iterable

from fastmcp import FastMCP
from fastmcp.server import Context

from src.schemas import ResumeMatch


def _get_resume_value(resume: ResumeMatch | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(resume, dict):
        return resume.get(key, default)
    return getattr(resume, key, default)


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def build_resume_prompt(
    job_description: str,
    matched_resumes: Iterable[ResumeMatch | dict[str, Any]],
) -> str:
    context = "Matched candidate profiles:\n\n"
    for index, resume in enumerate(matched_resumes, 1):
        skills = _ensure_list(_get_resume_value(resume, "skills", []))
        experience_years = _get_resume_value(resume, "experience_years", "Unknown")
        similarity = _get_resume_value(resume, "similarity_score", 0.0)
        context += f"{index}. Skills: {', '.join(skills)}\n"
        context += f"   Experience: {experience_years} years\n"
        context += f"   Match Score: {float(similarity):.2f}\n\n"

    return f"""Based on this job description and matched candidate profiles, generate an optimized resume.

Job Description:
{job_description}

{context}

Generate a professional resume that highlights relevant skills and experience for this role.
Include:
- Professional Summary
- Key Skills
- Work Experience (tailored to job requirements)
- Education
- Notable Achievements

Format in clean, professional markdown.
"""


def build_analysis_prompt(text: str) -> str:
    return f"""Analyze this job description and extract key information:

{text}

Provide:
1. Required skills (list)
2. Experience level (Entry/Mid/Senior/Lead)
3. Key responsibilities (list)
4. Estimated match threshold (0.0-1.0)

Format as JSON.
"""


def build_job_requirements_prompt(text: str) -> str:
    return f"""You are categorizing a job description into structured buckets.

Job Description:
{text}

Return JSON with the following keys, each containing a list of strings:
- work_experience: roles, responsibilities, projects, impact
- skills: explicit skills, tools, tech stacks
- education: degrees, schools, majors
- certification: certifications, badges, licenses
- project: standalone projects (if separate from roles)
- requirements: must-haves, minimum qualifications
- nice_to_have: preferred qualifications
- responsibilities: core duties (if not already in work_experience)

Rules:
- Use concise bullet-style strings.
- Do not invent details.
- Keep each entry self-contained for similarity search.
"""


def build_resume_from_source_prompt(
    job_description: str,
    resume_source: dict[str, Any],
    match_summary: dict[str, Any],
) -> str:
    resume_source_json = json.dumps(resume_source, ensure_ascii=True)
    match_summary_json = json.dumps(match_summary, ensure_ascii=True)
    return f"""You are updating the author's resume for a specific job description.

Rules:
- Use ONLY the facts present in Resume Source.
- Do not invent dates, roles, companies, or skills not listed.
- Prefer items most relevant to the job description and match summary.

Job Description:
{job_description}

Match Summary:
{match_summary_json}

Resume Source (JSON):
{resume_source_json}

Return a professional resume in markdown with:
- Professional Summary
- Key Skills
- Work Experience
- Education
- Certifications (if present)
- Projects (if present)
"""


def register_prompts(mcp: FastMCP) -> None:
    @mcp.prompt()
    async def resume_generation_prompt( 
        job_description: str,
        matched_resumes: list[dict[str, Any]] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, str]]:
        """Prompt to generate a tailored resume from a job description."""
        if ctx:
            await ctx.info("Building resume generation prompt")
        prompt = build_resume_prompt(job_description, matched_resumes or [])
        return [{"role": "user", "content": prompt}]

    @mcp.prompt()
    async def job_analysis_prompt(
        job_description: str,
        ctx: Context | None = None,
    ) -> list[dict[str, str]]:
        """Prompt to analyze a job description into structured fields."""
        if ctx:
            await ctx.info("Building job analysis prompt")
        prompt = build_analysis_prompt(job_description)
        return [{"role": "user", "content": prompt}]

    @mcp.prompt()
    async def job_requirements_prompt(
        job_description: str,
        ctx: Context | None = None,
    ) -> list[dict[str, str]]:
        """Prompt to categorize job requirements into buckets."""
        if ctx:
            await ctx.info("Building job requirements prompt")
        prompt = build_job_requirements_prompt(job_description)
        return [{"role": "user", "content": prompt}]

    @mcp.prompt()
    async def resume_from_source_prompt(
        job_description: str,
        resume_source: dict[str, Any],
        match_summary: dict[str, Any],
        ctx: Context | None = None,
    ) -> list[dict[str, str]]:
        """Prompt to generate a resume using structured source data."""
        if ctx:
            await ctx.info("Building resume from source prompt")
        prompt = build_resume_from_source_prompt(
            job_description,
            resume_source,
            match_summary,
        )
        return [{"role": "user", "content": prompt}]
