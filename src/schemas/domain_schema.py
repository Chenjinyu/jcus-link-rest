# src/models/domain_schema.py
"""
Domain-specific data schema with Pydantic models
"""

from pydantic import BaseModel, Field
from typing import List


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


