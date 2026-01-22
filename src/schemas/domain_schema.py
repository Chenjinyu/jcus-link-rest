# src/models/domain_schema.py
"""
Domain-specific data schema with Pydantic models
"""
from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import List, Optional

_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "libs" / "resume_template.json"

class BasicInfo(BaseModel):
    """Basic contact information for the resume."""

    name: str
    address: str
    mobile: str
    email: str
    github: str | None = None
    linkedin: str | None = None

def _default_basic_info() -> BasicInfo:
    data = _load_resume_template().get("basic_info", {})
    return BasicInfo(**data)

class ResumeMatch(BaseModel):
    """Resume match result from vector search"""
    resume_id: str
    content: str
    skills: List[str]
    experience_years: int
    similarity_score: float = Field(ge=0.0, le=1.0)


class JobAnalysis(BaseModel):
    """Job description analysis result"""
    required_skills: List[str]
    experience_level: str
    key_responsibilities: List[str]
    estimated_match_threshold: float = Field(ge=0.0, le=1.0)


class ProjectEntry(BaseModel):
    """Project details for a role."""

    project_summary: str | None = None
    project_details: List[str]


class PositionEntry(BaseModel):
    """Position entry within a professional experience item."""

    role: str | None = None
    projects: List[ProjectEntry]


class ProfessionalExperienceEntry(BaseModel):
    """Professional experience item."""

    company: str
    city: str
    state_or_country: str
    start_date: str
    end_date: str
    job_title: str
    skill_keywords: str | None = None
    positions: List[PositionEntry]


class EducationEntry(BaseModel):
    """Education entry."""

    institution: str
    degree: str
    start_date: str
    end_date: str


class CertificationEntry(BaseModel):
    """License or certification entry."""

    type: str
    name: str
    issuer: str
    validation_number: str | None = None

class SerializedJobReqCategory(BaseModel):
    """Structured job requirements grouped by category."""

    work_experience: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    certification: List[str] = Field(default_factory=list)
    project: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)


def _load_resume_template() -> dict:
    try:
        with _TEMPLATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _default_education() -> list[EducationEntry]:
    items = _load_resume_template().get("education", []) or []
    return [EducationEntry(**it) for it in items]


def _default_licenses() -> list[CertificationEntry]:
    items = _load_resume_template().get("licenses_and_certifications", []) or []
    return [CertificationEntry(**it) for it in items]

class ResumeSchema(BaseModel):
    """Full resume payload schema matching resume_data.json."""

    basic_info: BasicInfo = Field(default_factory=_default_basic_info)
    professional_experience: List[ProfessionalExperienceEntry]
    education: List[EducationEntry] = Field(default_factory=_default_education)
    licenses_and_certifications: List[CertificationEntry] = Field(default_factory=_default_licenses)