"""
LLM Service for resume generation and text processing
"""

import asyncio
import json
import logging
from typing import AsyncIterator, List, AsyncGenerator, Any
from abc import ABC, abstractmethod

from src.config import settings
from src.libs.exceptions import LLMServiceException
from src.schemas import ResumeMatch

logger = logging.getLogger(__name__)


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _format_date_range(entry: dict[str, Any]) -> str:
    start = entry.get("start_date")
    end = entry.get("end_date")
    is_current = entry.get("is_current")
    if start and (end or is_current):
        return f"{start} - {end or 'Present'}"
    if start:
        return str(start)
    return ""


def _render_resume_from_source(resume_source: dict[str, Any]) -> str:
    profile_data = resume_source.get("profile_data", []) or []
    personal_attributes = resume_source.get("personal_attributes", []) or []

    sections: list[str] = ["# Resume\n"]

    summary_items = [
        attr
        for attr in personal_attributes
        if attr.get("attribute_type") in {"summary", "bio", "headline"}
    ]
    if summary_items:
        sections.append("## Professional Summary\n")
        for item in summary_items:
            description = item.get("description") or item.get("value")
            if description:
                sections.append(f"- {description}\n")
        sections.append("\n")

    skills: list[str] = []
    for entry in profile_data:
        if entry.get("category") == "skill":
            data = entry.get("data") or {}
            skills.extend(_ensure_list(data.get("skills") or data.get("name")))
    if skills:
        unique_skills = sorted({skill for skill in skills if skill})
        sections.append("## Key Skills\n")
        sections.append("- " + ", ".join(unique_skills) + "\n\n")

    work_items = [entry for entry in profile_data if entry.get("category") == "work_experience"]
    if work_items:
        sections.append("## Work Experience\n")
        for entry in work_items:
            data = entry.get("data") or {}
            title = data.get("title") or data.get("position") or data.get("role")
            company = data.get("company") or data.get("organization")
            header = " | ".join([part for part in [title, company] if part])
            if header:
                sections.append(f"### {header}\n")
            date_range = _format_date_range(entry)
            if date_range:
                sections.append(f"*{date_range}*\n")
            details: list[str] = []
            details.extend(_ensure_list(data.get("description")))
            details.extend(_ensure_list(data.get("responsibilities")))
            details.extend(_ensure_list(data.get("achievements")))
            if details:
                for detail in details:
                    sections.append(f"- {detail}\n")
            sections.append("\n")

    education_items = [entry for entry in profile_data if entry.get("category") == "education"]
    if education_items:
        sections.append("## Education\n")
        for entry in education_items:
            data = entry.get("data") or {}
            degree = data.get("degree") or data.get("title")
            institution = data.get("institution") or data.get("school")
            line = " | ".join([part for part in [degree, institution] if part])
            if line:
                sections.append(f"- {line}\n")
        sections.append("\n")

    cert_items = [entry for entry in profile_data if entry.get("category") == "certification"]
    if cert_items:
        sections.append("## Certifications\n")
        for entry in cert_items:
            data = entry.get("data") or {}
            name = data.get("name") or data.get("title")
            issuer = data.get("issuer")
            line = " | ".join([part for part in [name, issuer] if part])
            if line:
                sections.append(f"- {line}\n")
        sections.append("\n")

    project_items = [entry for entry in profile_data if entry.get("category") == "project"]
    if project_items:
        sections.append("## Projects\n")
        for entry in project_items:
            data = entry.get("data") or {}
            title = data.get("title") or data.get("name")
            if title:
                sections.append(f"### {title}\n")
            details = _ensure_list(data.get("description"))
            for detail in details:
                sections.append(f"- {detail}\n")
            sections.append("\n")

    return "".join(sections).rstrip() + "\n"


class BaseLLMService(ABC):
    """Abstract base class for LLM services"""

    @abstractmethod
    async def generate_resume(
        self,
        job_description: str,
        matched_resumes: List[ResumeMatch],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate resume based on job description and matches"""
        raise NotImplementedError

    @abstractmethod
    async def analyze_text(self, text: str) -> dict:
        """Analyze text and extract structured information"""
        raise NotImplementedError

    @abstractmethod
    async def generate_resume_from_source(
        self,
        job_description: str,
        resume_source: dict[str, Any],
        match_summary: dict[str, Any],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate resume using structured source data"""
        raise NotImplementedError


class AnthropicLLMService(BaseLLMService):
    """Anthropic Claude LLM service implementation"""

    def __init__(self):
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

        if not self.api_key:
            logger.warning("Anthropic API key not set, using simulated responses")

    async def generate_resume(
        self,
        job_description: str,
        matched_resumes: List[ResumeMatch],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate optimized resume using Claude"""

        try:
            prompt = self._build_resume_prompt(job_description, matched_resumes)

            if stream:
                async for chunk in self._stream_generate(prompt):
                    yield chunk
            else:
                result = await self._generate(prompt)
                yield result

        except Exception as e:
            logger.error("Resume generation failed: %s", e)
            raise LLMServiceException("resume generation", str(e))

    async def analyze_text(self, text: str) -> dict:
        """Analyze job description using Claude"""

        try:
            prompt = self._build_analysis_prompt(text)
            _ = prompt

            return {
                "required_skills": ["Python", "FastAPI", "React", "AWS"],
                "experience_level": "Senior",
                "key_responsibilities": [
                    "Design and implement scalable systems",
                    "Lead technical projects",
                    "Mentor junior developers",
                ],
                "estimated_match_threshold": 0.7,
            }

        except Exception as e:
            logger.error("Text analysis failed: %s", e)
            raise LLMServiceException("text analysis", str(e))

    async def generate_resume_from_source(
        self,
        job_description: str,
        resume_source: dict[str, Any],
        match_summary: dict[str, Any],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        try:
            if not self.api_key:
                yield _render_resume_from_source(resume_source)
                return

            prompt = self._build_resume_from_source_prompt(
                job_description, resume_source, match_summary
            )
            if stream:
                async for chunk in self._stream_generate(prompt):
                    yield chunk
            else:
                result = await self._generate(prompt)
                yield result
        except Exception as e:
            logger.error("Resume generation failed: %s", e)
            raise LLMServiceException("resume generation", str(e))

    def _build_resume_prompt(
        self,
        job_description: str,
        matched_resumes: List[ResumeMatch],
    ) -> str:
        """Build prompt for resume generation"""

        context = "Matched candidate profiles:\n\n"
        for i, resume in enumerate(matched_resumes, 1):
            context += f"{i}. Skills: {', '.join(resume.skills)}\n"
            context += f"   Experience: {resume.experience_years} years\n"
            context += f"   Match Score: {resume.similarity_score:.2f}\n\n"

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

    def _build_analysis_prompt(self, text: str) -> str:
        """Build prompt for job description analysis"""

        return f"""Analyze this job description and extract key information:

{text}

Provide:
1. Required skills (list)
2. Experience level (Entry/Mid/Senior/Lead)
3. Key responsibilities (list)
4. Estimated match threshold (0.0-1.0)

Format as JSON.
"""

    def _build_resume_from_source_prompt(
        self,
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

    async def _stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream LLM response (simulated for now)"""

        if self.api_key:
            _ = prompt
            pass

        resume_parts = [
            "# Professional Resume\n\n",
            "## Professional Summary\n",
            "Experienced software engineer with strong background in Python, TypeScript, and cloud technologies. ",
            "Proven track record of building scalable applications and leading development teams.\n\n",
            "## Key Skills\n",
            "- **Programming Languages:** Python, TypeScript, JavaScript\n",
            "- **Frameworks:** FastAPI, React, Node.js\n",
            "- **Cloud Platforms:** AWS, GCP\n",
            "- **Tools:** Docker, Kubernetes, Git\n\n",
            "## Work Experience\n\n",
            "### Senior Software Engineer | Tech Company\n",
            "*2020 - Present*\n\n",
            "- Developed microservices architecture serving 1M+ users\n",
            "- Led team of 5 engineers in implementing CI/CD pipeline\n",
            "- Reduced deployment time by 60% through automation\n\n",
            "## Education\n",
            "**Bachelor of Science in Computer Science**\n",
            "University Name, 2018\n\n",
            "## Achievements\n",
            "- Architected system handling 10K requests/second\n",
            "- Published 3 technical articles on Medium\n",
            "- Contributed to 5+ open source projects\n",
        ]

        for part in resume_parts:
            await asyncio.sleep(0.1)
            yield part

    async def _generate(self, prompt: str) -> str:
        """Generate complete response"""

        result = []
        async for chunk in self._stream_generate(prompt):
            result.append(chunk)
        return "".join(result)


class OpenAILLMService(BaseLLMService):
    """OpenAI GPT service implementation"""

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model

    async def generate_resume(
        self,
        job_description: str,
        matched_resumes: List[ResumeMatch],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate resume using OpenAI"""
        yield "OpenAI implementation"

    async def analyze_text(self, text: str) -> dict:
        """Analyze text using OpenAI"""
        _ = text
        return {}

    async def generate_resume_from_source(
        self,
        job_description: str,
        resume_source: dict[str, Any],
        match_summary: dict[str, Any],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        _ = job_description
        _ = match_summary
        if not self.api_key:
            yield _render_resume_from_source(resume_source)
            return
        yield "OpenAI implementation"


class LLMServiceFactory:
    """Factory for creating LLM service instances"""

    @staticmethod
    def create() -> BaseLLMService:
        """Create LLM service based on configuration"""

        provider = settings.llm_provider.lower()

        if provider == "anthropic":
            logger.info("Using Anthropic LLM service")
            return AnthropicLLMService()
        if provider == "openai":
            logger.info("Using OpenAI LLM service")
            return OpenAILLMService()
        raise ValueError(f"Unsupported LLM provider: {provider}")


_llm_service: BaseLLMService | None = None


def get_llm_service() -> BaseLLMService:
    """Get LLM service instance (singleton)"""
    global _llm_service

    if _llm_service is None:
        _llm_service = LLMServiceFactory.create()

    return _llm_service
