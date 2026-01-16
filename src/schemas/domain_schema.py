# src/models/domain_schema.py
"""
Domain-specific data schema with Pydantic models
"""

from pydantic import BaseModel, Field
from typing import List, Optional


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


class BasicInfo(BaseModel):
    """Basic contact information for the resume."""

    name: str
    address: str
    mobile: str
    email: str
    github: str | None = None
    linkedin: str | None = None


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


class ResumeSchema(BaseModel):
    """Full resume payload schema matching resume_data.json."""

    basic_info: BasicInfo
    professional_experience: List[ProfessionalExperienceEntry]
    education: List[EducationEntry]
    licenses_and_certifications: List[CertificationEntry]

