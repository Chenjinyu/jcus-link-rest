"""
MCP tool registrations for resume matching and generation.
For questions in the chat window, the Next.js will connect to supabase vector DB
and complete the conversation.

This module defines tools for:
- Listing matched job skills from a job description.
- Generating an updated resume based on a job description.
- Downloading the latest resume without a job description.
- Checking resume cache status.
"""

from __future__ import annotations
import json
import logging
from fastmcp import FastMCP
from fastmcp.server import Context

from supabase import Client

from src.config import settings
from src.services import (
    get_model_id, 
    profile_similarity_search_rpc
)
from src.libs import (
    VectorDB, 
    get_llm, 
    FileUploadException, 
    parser_job_description
)
from src.utils import util

logger = logging.getLogger(__name__)
_vector_db_client: Client | None = None

def _get_user_id() -> str:
    """Resolve user ID, defaulting to system user if not provided."""
    return settings.user_id or "system"


def _get_provider():
    """Get the default LLM provider."""
    return settings.default_llm_provider or "ollama"


def _get_embeding_model_and_dimissions():
    """Get the default embedding model dimensions."""
    model_name = settings.default_embedding_model_name or settings.ollama_embedding_model_name
    
    if model_name.startswith("text-embedding-3"):
        return model_name, 1536
    else:
        return model_name, 768

def _get_supabase_db_client():
    global _vector_db_client
    if _vector_db_client is None:
          _vector_db_client = VectorDB().get_supabase_db_client()
    return _vector_db_client

def register_tools(mcp: FastMCP) -> None:
    
    _vector_db_client = _get_supabase_db_client()
    _user_id = _get_user_id()
    _provider = _get_provider()
    _embedding_model_name, _dimissions = _get_embeding_model_and_dimissions()
    _llm = get_llm()    
    
    @mcp.tool()
    async def generate_matched_resume(
        input_data: str,
        input_type: str = "file",
        filename: str | None = None,
        user_id: str | None = None,
        top_k: int = 10,
        threshold: float | None = None,
        provider: str | None = None,
        ctx: Context | None = None,
    ):
        """
        Generate an updated resume based on matching a job description.
        1. get a uploaded job description, it could be a pdf, docx, txt file or url, 
        parse the job description to text
            1.1. call document parser to parse the file/url to text.
            1.2. the uploaded content could be:
                - base64 encoded file content
                - a url link
                - raw text
            1.3. DocumentParser.parse(
                    content: Union[bytes, str],
                    file_type: str | None = None,
                    is_url: bool = False
                ) 
        2. chunk the job description text and generate embedding
        3. call vector DB to search similar profile content by embedding
        4. aggregate the matched profile content and call LLM to generate updated resume
        5. return the updated resume content as base64 encoded pdf file
        """
        try:
            user_id = user_id or _user_id
            provider = provider or _provider
            top_k = top_k or settings.default_top_k
            threshold = threshold or settings.min_similarity_threshold
            if ctx:
                # sends info message tied to the current MCP request and send to MCP client.
                await ctx.info("Parsing job description for skill matching")
            
            # 1. parse the job description from input data with type.
            job_description = await parser_job_description(
                input_data=input_data,
                input_type=input_type,
                filename=filename,
            )

            if not job_description.strip():
                raise ValueError("Job description cannot be empty")

            if ctx:
                await ctx.info("Running similarity search against profile data")
            # 2. chunk the job description text and generate embedding
            chunk_job_descs = util.chunk_text(job_description)
            
            # get model id for searching corresponding embeded text
            model_id = await get_model_id(
                vector_db_client=_vector_db_client,
                provider=provider,
                model_identifier=_embedding_model_name
            )
            
            match_items = []
            for chunk in chunk_job_descs:
                query_embedding = await _llm.generate_embeddings(
                    text=chunk
                )
                
                rpc_response = await profile_similarity_search_rpc(
                    vector_db_client=_vector_db_client,
                    query_embedding=query_embedding,
                    model_id=model_id,
                    embedding_dimensions=_dimissions,
                    user_id=user_id,
                    top_k=top_k,
                    threshold=threshold
                )
                match_items.extend(rpc_response.data) # type: ignore
            print(json.dumps(match_items, indent=2))
            return match_items
            #     similarity_total = 0.0
            #     for match in matches:
            #         similarity = float(match.get("similarity") or 0.0)
            #         similarity_total += similarity
            #         match_items.append(
            #             {
            #                 "chunk_text": match.get("chunk_text") or "",
            #                 "similarity": similarity,
            #                 "match_rate_percent": int(round(similarity * 100)),
            #                 "title": match.get("title"),
            #                 "content_type": match.get("content_type"),
            #                 "metadata": match.get("metadata"),
            #             }
            #         )

            # match_rate = similarity_total / len(match_items) if match_items else 0.0

            # return json.dumps(
            #     {
            #         "status": "success",
            #         "job_description_preview": job_description,
            #         "match_rate": match_rate,
            #         "match_rate_percent": int(round(match_rate * 100)),
            #         "matches": match_items,
            #         "total_matches": len(match_items),
            #     },
            #     indent=2,
            #     ensure_ascii=True,
            # )
        except Exception as exc:
            if ctx:
                await ctx.error(f"Skill match failed: {util.format_exception_message(exc)}")
            logger.exception("Skill match failed")
            raise FileUploadException(util.format_exception_message(exc)) from exc


