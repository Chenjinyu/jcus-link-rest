#!/usr/bin/env python3
"""Example client for calling MCP tools, prompts, and resources.

Prompt construction and LLM calls happen server-side; this script only invokes MCP APIs.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_tool(url: str, tool_name: str, arguments: dict[str, Any], request_id: int) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    return _post_json(url, payload)


def list_prompts(url: str, request_id: int) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "prompts/list",
        "params": {},
    }
    return _post_json(url, payload)


def get_prompt(url: str, prompt_name: str, arguments: dict[str, Any], request_id: int) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "prompts/get",
        "params": {
            "name": prompt_name,
            "arguments": arguments,
        },
    }
    return _post_json(url, payload)


def list_resources(url: str, request_id: int) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "resources/list",
        "params": {},
    }
    return _post_json(url, payload)


def read_resource(url: str, uri: str, request_id: int) -> dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "resources/read",
        "params": {"uri": uri},
    }
    return _post_json(url, payload)


def _encode_file(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def run_examples(base_url: str) -> int:
    request_id = 1

    def _print(label: str, payload: dict[str, Any]) -> None:
        print(f"\n=== {label} ===")
        print(json.dumps(payload, indent=2, ensure_ascii=True))

    try:
        result = call_tool(
            base_url,
            "list_matched_job_skills",
            {
                "input_type": "text",
                "input_data": "We need a backend engineer with FastAPI and AWS.",
                "top_k": 5,
                "threshold": 0.5,
            },
            request_id,
        )
        _print("tool:list_matched_job_skills", result)
        request_id += 1

        result = call_tool(
            base_url,
            "extract_jobs_insights",
            {
                "job_description": "We need a backend engineer with FastAPI and AWS.",
                "top_k": 5,
                "use_cache": True,
            },
            request_id,
        )
        _print("tool:extract_jobs_insights", result)
        request_id += 1

        result = call_tool(
            base_url,
            "download_latest_resume",
            {"use_cache": True},
            request_id,
        )
        _print("tool:download_latest_resume", result)
        request_id += 1

        result = call_tool(base_url, "resume_cache_status", {}, request_id)
        _print("tool:resume_cache_status", result)
        request_id += 1

        result = list_prompts(base_url, request_id)
        _print("prompts:list", result)
        request_id += 1

        result = get_prompt(
            base_url,
            "resume_generation_prompt",
            {
                "job_description": "We need a backend engineer with FastAPI and AWS.",
                "matched_resumes": [],
            },
            request_id,
        )
        _print("prompts:get resume_generation_prompt", result)
        request_id += 1

        result = get_prompt(
            base_url,
            "job_analysis_prompt",
            {"job_description": "We need a backend engineer with FastAPI and AWS."},
            request_id,
        )
        _print("prompts:get job_analysis_prompt", result)
        request_id += 1

        result = get_prompt(
            base_url,
            "resume_from_source_prompt",
            {
                "job_description": "We need a backend engineer with FastAPI and AWS.",
                "resume_source": {"skills": ["FastAPI", "AWS"]},
                "match_summary": {"top_matches": ["FastAPI"]},
            },
            request_id,
        )
        _print("prompts:get resume_from_source_prompt", result)
        request_id += 1

        result = list_resources(base_url, request_id)
        _print("resources:list", result)
        request_id += 1

        result = read_resource(base_url, "resource://mcp/server-info", request_id)
        _print("resources:read server-info", result)
        request_id += 1

        result = read_resource(base_url, "resource://mcp/prompts", request_id)
        _print("resources:read prompts", result)
        request_id += 1

        result = read_resource(
            base_url,
            "resource://mcp/prompts/resume_generation_prompt",
            request_id,
        )
        _print("resources:read prompts/resume_generation_prompt", result)

    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}")
        print(exc.read().decode("utf-8"))
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Call MCP tools/prompts/resources.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/mcp",
        help="MCP endpoint URL (default: http://localhost:8000/mcp)",
    )
    parser.add_argument(
        "--file",
        help="Optional file path for list_matched_job_skills input_type=file",
    )
    args = parser.parse_args()

    if args.file:
        encoded = _encode_file(args.file)
        payload = call_tool(
            args.url,
            "list_matched_job_skills",
            {
                "input_type": "file",
                "filename": args.file.split("/")[-1],
                "input_data": encoded,
                "top_k": 5,
            },
            request_id=1,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    return run_examples(args.url)


if __name__ == "__main__":
    sys.exit(main())
