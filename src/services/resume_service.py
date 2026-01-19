# """
# Resume Service - High-level business logic for resume operations
# """

# from __future__ import annotations

# import json
# import logging
# from dataclasses import dataclass
# from datetime import datetime, timedelta, timezone
# from typing import Any, List, AsyncIterator, cast
# from collections.abc import AsyncGenerator

# from src.config import settings
# from src.libs.resume_cache import ResumeCacheEntry, get_resume_cache
# from src.libs.vector_database import EmbeddingModel
# from libs.llm import get_llm
# from src.mcp.prompts import (
#     build_analysis_prompt,
#     build_resume_from_source_prompt,
#     build_job_requirements_prompt,
# )
# from src.schemas import (
#     ResumeMatch,
#     ResumeSchema,
#     SerializedJobReqCategory,
# )

# logger = logging.getLogger(__name__)

# # TODO(jc): why needs this function?
# def _default_analysis_result() -> dict:
#     return {}

# # TODO(jc): why needs this function?
# def _parse_analysis_response(response: str | None) -> dict:
#     if not response:
#         return _default_analysis_result()
#     raw = response.strip()
#     if "{" in raw and "}" in raw:
#         raw = raw[raw.find("{") : raw.rfind("}") + 1]
#     try:
#         data = json.loads(raw)
#     except json.JSONDecodeError:
#         return _default_analysis_result()
#     if not isinstance(data, dict):
#         return _default_analysis_result()
#     return data

# # TODO(jc): why needs this function?
# def _parse_job_requirements_response(response: str | None) -> dict:
#     if not response:
#         return {}
#     raw = response.strip()
#     if "{" in raw and "}" in raw:
#         raw = raw[raw.find("{") : raw.rfind("}") + 1]
#     try:
#         data = json.loads(raw)
#     except json.JSONDecodeError:
#         return {}
#     if not isinstance(data, dict):
#         return {}
#     return data

# @dataclass
# class MatchSummary:
#     summary: str
#     match_rate: float
#     match_rate_percent: int
#     matched_skills: list[str]
#     missing_skills: list[str]

#     def to_dict(self) -> dict[str, Any]:
#         return {
#             "summary": self.summary,
#             "match_rate": self.match_rate,
#             "match_rate_percent": self.match_rate_percent,
#             "matched_skills": self.matched_skills,
#             "missing_skills": self.missing_skills,
#         }


# class ResumeService:
#     """Service for resume-related operations"""

#     def __init__(self, user_id:str) -> None:
#         self.user_id = user_id
#         self.profile_service: ProfileService | None
#         try:
#             self.profile_service = get_profile_service()
#         except Exception as exc:
#             logger.warning("Profile service unavailable: %s", exc)
#             self.profile_service = None

#         self.resume_cache = get_resume_cache(
#             max_entries=settings.resume_cache_max_entries,
#             ttl_seconds=settings.resume_cache_ttl_seconds,
#             cache_path=settings.resume_cache_path,
#         )


#     async def _search_profile_matches(
#         self,
#         job_description: str,
#         top_k: int,
#         embedding_model: EmbeddingModel,
#     ) -> tuple[list[dict[str, Any]], list[ResumeMatch]]:
#         if self.profile_service is None:
#             return [], []
#         raw_results = await self.profile_service.search_job_matches(
#             job_description=job_description,
#             user_id=self.user_id,
#             top_k=top_k,
#             threshold=settings.min_similarity_threshold,
#             embedding_model=embedding_model,
#         )
#         matches = self._map_search_results_to_matches(raw_results)
#         return raw_results, matches

#     async def categorize_job_requirements(
#         self,
#         job_description: str,
#         provider: str,
#     ) -> SerializedJobReqCategory:
#         """Categorize job requirements into structured buckets."""
#         llm_service = get_llm_service(provider)
#         prompt = build_job_requirements_prompt(job_description)
#         llm_response = await llm_service.extract_jobs_insights(prompt)
#         data = _parse_job_requirements_response(llm_response)
#         try:
#             return SerializedJobReqCategory.model_validate(data)
#         except Exception:
#             return SerializedJobReqCategory()

#     # NOTE(jc): new function.
#     async def get_matched_profiles_for_resume_generation(
#         self,
#         top_k: int,
#         embedding_model: EmbeddingModel,
#         job_description_category: SerializedJobReqCategory,
#     ) -> Any:
#         """Generate matched profiles for resume
#         1. for loop each category in job_description_category as the search query to get profile matches by using ProfileService function.
#         2. find out the top_k matches for each category and aggregate the results.
#         3. call llm with prompt to aggreate the matched profiles into work experience entries.
#         4. update the result to ResumeSchema.professional_experiences.
#         5. call pdf_geneartor, to generate the resume pdf.
#         """
#         similarity_matches = []
#         # Iterate through each category in the job description to get matches profile
#         for category in job_description_category.model_dump().values():
#             if isinstance(category, str):
#                 category = [category]
#             if isinstance(category, list):
#                 for item in category:
#                     if isinstance(item, str) and item.strip():
#                         _, matches = await self._search_profile_matches(
#                             job_description=item,
#                             top_k=top_k,
#                             embedding_model=embedding_model,
#                         )
#                         similarity_matches.extend(matches)
            
#         pass
    

# _resume_service: ResumeService | None = None


# def get_resume_service(user_id: str | None = None) -> ResumeService:
#     """Get resume service instance (singleton)"""
#     global _resume_service
#     user_id = user_id or settings.user_id or "default_user"
#     if _resume_service is None:
#         _resume_service = ResumeService(user_id=user_id)

#     return _resume_service
