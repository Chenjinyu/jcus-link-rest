"""
Resume Service - High-level business logic for resume operations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, AsyncIterator, cast
from collections.abc import AsyncGenerator

from src.config import settings
from src.libs.resume_cache import ResumeCacheEntry, get_resume_cache
from src.libs.vector_database import EmbeddingModel
from src.services.llm_service import get_llm_service
from src.services.profile_service import get_profile_service, ProfileService
from src.mcp.prompts import (
    build_analysis_prompt,
    build_resume_from_source_prompt,
    build_resume_prompt,
)
from src.schemas import (
    ResumeMatch,
    JobAnalysis,
    SearchMatchesRequest,
    SearchMatchesResponse,
)

logger = logging.getLogger(__name__)


def _default_analysis_result() -> dict:
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


def _parse_analysis_response(response: str | None) -> dict:
    if not response:
        return _default_analysis_result()
    raw = response.strip()
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _default_analysis_result()
    if not isinstance(data, dict):
        return _default_analysis_result()
    return data


@dataclass
class MatchSummary:
    summary: str
    match_rate: float
    match_rate_percent: int
    matched_skills: list[str]
    missing_skills: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "match_rate": self.match_rate,
            "match_rate_percent": self.match_rate_percent,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
        }


class ResumeService:
    """Service for resume-related operations"""

    def __init__(self) -> None:
        self.profile_service: ProfileService | None
        try:
            self.profile_service = get_profile_service()
        except Exception as exc:
            logger.warning("Profile service unavailable: %s", exc)
            self.profile_service = None

        self.resume_cache = get_resume_cache(
            max_entries=settings.resume_cache_max_entries,
            ttl_seconds=settings.resume_cache_ttl_seconds,
            cache_path=settings.resume_cache_path,
        )

    def _resolve_user_id(self, user_id: str | None) -> str:
        resolved = user_id or settings.author_user_id
        if not resolved:
            raise ValueError("user_id is required (set author_user_id or pass user_id)")
        return resolved

    def _map_search_results_to_matches(
        self,
        results: list[dict[str, Any]],
    ) -> list[ResumeMatch]:
        matches: list[ResumeMatch] = []
        for result in results:
            metadata = result.get("metadata") or {}
            profile_data = metadata.get("profile_data") or {}
            skills = profile_data.get("skills") or []
            if isinstance(skills, str):
                skills = [skills]
            experience_years = profile_data.get("experience_years") or 0
            try:
                experience_years = int(experience_years)
            except (TypeError, ValueError):
                experience_years = 0
            resume_id = (
                result.get("profile_data_id")
                or result.get("document_id")
                or result.get("article_id")
                or "unknown"
            )
            matches.append(
                ResumeMatch(
                    resume_id=str(resume_id),
                    content=result.get("chunk_text") or result.get("content") or "",
                    skills=skills if isinstance(skills, list) else [],
                    experience_years=experience_years,
                    similarity_score=float(result.get("similarity") or 0.0),
                )
            )
        return matches

    async def _search_profile_matches(
        self,
        job_description: str,
        user_id: str,
        top_k: int,
        embedding_model: EmbeddingModel,
    ) -> tuple[list[dict[str, Any]], list[ResumeMatch]]:
        if self.profile_service is None:
            return [], []
        raw_results = await self.profile_service.search_job_matches(
            job_description=job_description,
            user_id=user_id,
            top_k=top_k,
            threshold=settings.min_similarity_threshold,
            embedding_model=embedding_model,
        )
        matches = self._map_search_results_to_matches(raw_results)
        return raw_results, matches

    async def search_matching_resumes(
        self,
        request: SearchMatchesRequest,
        provider: str,
        embedding_model: EmbeddingModel,
        user_id: str | None = None,
    ) -> SearchMatchesResponse:
        """Search for resumes matching the job description"""

        logger.info("Searching for top %s matching resumes", request.top_k)

        resolved_user_id = self._resolve_user_id(user_id)
        _, matches = await self._search_profile_matches(
            request.job_description,
            resolved_user_id,
            request.top_k,
            embedding_model,
        )

        logger.info("Found %s matching resumes", len(matches))

        return SearchMatchesResponse(matches=matches, total_found=len(matches))

    async def analyze_job_description(
        self,
        job_description: str,
        provider: str,
    ) -> JobAnalysis:
        """Analyze job description to extract key information"""

        logger.info("Analyzing job description")

        llm_service = get_llm_service(provider)
        prompt = build_analysis_prompt(job_description)
        response = await llm_service.generate_text_response(prompt)
        analysis_data = _parse_analysis_response(response)

        return JobAnalysis(
            required_skills=analysis_data.get("required_skills", []),
            experience_level=analysis_data.get("experience_level", "Mid"),
            key_responsibilities=analysis_data.get("key_responsibilities", []),
            estimated_match_threshold=analysis_data.get(
                "estimated_match_threshold",
                0.7,
            ),
        )

    async def summarize_matches(
        self,
        job_description: str,
        matches: list[ResumeMatch],
        provider: str,
    ) -> MatchSummary:
        analysis = await self.analyze_job_description(job_description, provider)
        required_skills = {skill.lower() for skill in analysis.required_skills}
        matched_skill_pool = {skill.lower() for m in matches for skill in m.skills}
        matched_skills = sorted({skill for skill in required_skills & matched_skill_pool})
        missing_skills = sorted({skill for skill in required_skills - matched_skill_pool})

        if matches:
            similarity_avg = sum(m.similarity_score for m in matches) / len(matches)
        else:
            similarity_avg = 0.0

        if required_skills:
            skill_coverage = len(matched_skills) / len(required_skills)
            match_rate = min(1.0, 0.7 * similarity_avg + 0.3 * skill_coverage)
        else:
            match_rate = similarity_avg

        match_rate_percent = int(round(match_rate * 100))

        if not matches:
            summary = "No strong matches found for this job description."
        elif required_skills:
            summary = (
                f"Match rate {match_rate_percent}% with {len(matched_skills)} of "
                f"{len(required_skills)} required skills aligned."
            )
        else:
            summary = f"Match rate {match_rate_percent}% based on semantic similarity."

        return MatchSummary(
            summary=summary,
            match_rate=match_rate,
            match_rate_percent=match_rate_percent,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        )

    async def generate_optimized_resume(
        self,
        job_description: str,
        matched_resumes: List[ResumeMatch],
        provider: str,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate an optimized resume based on job description and matches"""

        logger.info("Generating resume for %s matched profiles", len(matched_resumes))

        llm_service = get_llm_service(provider)
        prompt = build_resume_prompt(job_description, matched_resumes)
        resume_generator = cast(
            AsyncGenerator[str, None],
            llm_service.generate_stream_text(prompt, stream=stream),
        )
        async for chunk in resume_generator:
            yield chunk

        logger.info("Resume generation complete")

    async def generate_updated_resume(
        self,
        job_description: str,
        provider: str,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        user_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        resolved_user_id = self._resolve_user_id(user_id)

        if self.profile_service is None:
            raise ValueError("Profile service unavailable")

        raw_matches, matches = await self._search_profile_matches(
            job_description,
            resolved_user_id,
            top_k,
            embedding_model,
        )

        match_summary = await self.summarize_matches(job_description, matches, provider)

        profile_ids = [
            str(match.get("profile_data_id"))
            for match in raw_matches
            if match.get("profile_data_id")
        ]
        resume_source = self.profile_service.get_resume_source(
            user_id=resolved_user_id,
            profile_ids=profile_ids or None,
        )
        profile_fingerprint = self.profile_service.fingerprint_resume_source(resume_source)
        cache_key = self.resume_cache.build_key(job_description, profile_fingerprint)

        if use_cache:
            cached_entry = await self.resume_cache.get(cache_key)
            if cached_entry:
                cached_summary = cached_entry.metadata.get("match_summary")
                return {
                    "resume": cached_entry.resume_text,
                    "match_summary": cached_summary
                    or {
                        "summary": cached_entry.summary,
                        "match_rate": cached_entry.match_rate,
                        "match_rate_percent": int(round(cached_entry.match_rate * 100)),
                    },
                    "matches": [m.model_dump() for m in matches],
                    "cache_hit": True,
                }

        resume_chunks: list[str] = []
        prompt = build_resume_from_source_prompt(
            job_description,
            resume_source,
            match_summary.to_dict(),
        )
        resume_text = await get_llm_service(provider).generate_text_response(prompt)

        await self.resume_cache.set(
            ResumeCacheEntry(
                key=cache_key,
                resume_text=resume_text,
                summary=match_summary.summary,
                match_rate=match_summary.match_rate,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=settings.resume_cache_ttl_seconds),
                metadata={
                    "profile_fingerprint": profile_fingerprint,
                    "user_id": resolved_user_id,
                    "match_summary": match_summary.to_dict(),
                },
            )
        )

        return {
            "resume": resume_text,
            "match_summary": match_summary.to_dict(),
            "matches": [m.model_dump() for m in matches],
            "cache_hit": False,
        }

    async def generate_latest_resume(
        self,
        provider: str,
        user_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        resolved_user_id = self._resolve_user_id(user_id)

        if self.profile_service is None:
            raise ValueError("Profile service unavailable")

        resume_source = self.profile_service.get_resume_source(user_id=resolved_user_id)
        profile_fingerprint = self.profile_service.fingerprint_resume_source(resume_source)
        cache_key = self.resume_cache.build_key("latest_resume", profile_fingerprint)

        if use_cache:
            cached_entry = await self.resume_cache.get(cache_key)
            if cached_entry:
                return {
                    "resume": cached_entry.resume_text,
                    "match_summary": {
                        "summary": cached_entry.summary,
                        "match_rate": cached_entry.match_rate,
                        "match_rate_percent": int(round(cached_entry.match_rate * 100)),
                        "matched_skills": [],
                        "missing_skills": [],
                    },
                    "cache_hit": True,
                }

        match_summary = MatchSummary(
            summary="Generated latest resume from current profile data.",
            match_rate=1.0,
            match_rate_percent=100,
            matched_skills=[],
            missing_skills=[],
        )

        resume_chunks: list[str] = []
        prompt = build_resume_from_source_prompt(
            "",
            resume_source,
            match_summary.to_dict(),
        )
        resume_text = await get_llm_service(provider).generate_text_response(prompt)

        await self.resume_cache.set(
            ResumeCacheEntry(
                key=cache_key,
                resume_text=resume_text,
                summary=match_summary.summary,
                match_rate=match_summary.match_rate,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=settings.resume_cache_ttl_seconds),
                metadata={
                    "profile_fingerprint": profile_fingerprint,
                    "user_id": resolved_user_id,
                },
            )
        )

        return {
            "resume": resume_text,
            "match_summary": match_summary.to_dict(),
            "cache_hit": False,
        }


_resume_service: ResumeService | None = None


def get_resume_service() -> ResumeService:
    """Get resume service instance (singleton)"""
    global _resume_service

    if _resume_service is None:
        _resume_service = ResumeService()

    return _resume_service
