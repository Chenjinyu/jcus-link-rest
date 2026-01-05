# src/models/domain_schema.py
"""
Domain-specific data schema with Pydantic models
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


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


class SearchMatchesRequest(BaseModel):
    """Request for searching matching resumes"""
    job_description: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchMatchesResponse(BaseModel):
    """Response for searching matching resumes"""
    matches: List[ResumeMatch]
    total_found: int


class AnalyzeJobRequest(BaseModel):
    """Request for analyzing job description"""
    job_description: str


class GenerateResumeRequest(BaseModel):
    """Request for generating resume"""
    job_description: str
    matched_resumes: List[ResumeMatch]
    stream: bool = True


class UploadJobResponse(BaseModel):
    """Response for job description upload"""
    status: str
    job_description: str
    matches: List[ResumeMatch]
    upload_time: datetime


class ResumeData(BaseModel):
    """Resume data stored in vector database"""
    id: str
    content: str
    skills: List[str]
    experience_years: int
    education: Optional[str] = None
    certifications: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: dict = Field(default_factory=dict)