# jcus-link-rest
REST API with MCP


## How to Run
**Option 1**: Using the run script
```sh
uv run python run_server.py
```
**Option 2**: Using uvicorn directly
```sh
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
**Option 3**: Using Python module syntax
```sh
uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```


## MCP tool examples

Below are example JSON-RPC payloads you can send to `/mcp` for the MCP tools.

### list_matched_job_skills
Parse a job description (file/text/url) and return matched chunks with similarity rates.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "list_matched_job_skills",
    "arguments": {
      "input_type": "file",
      "filename": "job-description.pdf",
      "input_data": "<base64-encoded-file>",
      "top_k": 10,
      "threshold": 0.5
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "list_matched_job_skills",
    "arguments": {
      "input_type": "text",
      "input_data": "We need a backend engineer with FastAPI, Postgres, and AWS experience.",
      "top_k": 5
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "list_matched_job_skills",
    "arguments": {
      "input_type": "url",
      "input_data": "https://example.com/job-posting"
    }
  }
}
```

### generate_updated_resume
Generate a resume tailored to a job description (uses cache when enabled).

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "generate_updated_resume",
    "arguments": {
      "job_description": "We need a backend engineer with FastAPI, Postgres, and AWS experience.",
      "top_k": 5,
      "use_cache": true
    }
  }
}
```

### download_latest_resume
Return the latest resume without a job description (uses cache when enabled).

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "download_latest_resume",
    "arguments": {
      "use_cache": true
    }
  }
}
```

### resume_cache_status
Return resume cache statistics for health checks.

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "resume_cache_status",
    "arguments": {}
  }
}
```
