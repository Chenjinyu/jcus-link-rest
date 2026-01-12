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

FastMCP’s streamable HTTP transport can reply in two modes on the same endpoint:

- application/json for normal JSON‑RPC responses (single response body).
- text/event-stream for streamed responses and session events.

The server enforces that the client explicitly declares it can handle both, because it may choose either based on the request or server state. That’s why it rejects requests that only accept one of them.
So: FastMCP streamable HTTP requires clients to accept both
`application/json` and `text/event-stream`, so include the Accept header in
requests.

Below are example JSON-RPC payloads you can send to `/mcp` for the MCP tools.

### Tools
1. List matched job skills
```sh
curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/call",
      "params": {
        "name": "list_matched_job_skills",
        "arguments": {
          "input_type": "text",
          "input_data": "We need a backend engineer with FastAPI and AWS.",
          "top_k": 5,
          "threshold": 0.5
        }
      }
    }'
```

2. generate a latest/updated resume
```sh
curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 2,
      "method": "tools/call",
      "params": {
        "name": "generate_updated_resume",
        "arguments": {
          "job_description": "We need a backend engineer with FastAPI and AWS.",
          "top_k": 5,
          "use_cache": true
        }
      }
    }'
```

3. download the latest resume
```sh
curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 3,
      "method": "tools/call",
      "params": {
        "name": "download_latest_resume",
        "arguments": {
          "use_cache": true
        }
      }
    }'
```

4. Check the cache status for storing resume with LRU
```sh
curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 4,
      "method": "tools/call",
      "params": {
        "name": "resume_cache_status",
        "arguments": {}
      }
    }'
```

### Resources
1. get all resources
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 9,
      "method": "resources/list",
      "params": {}
    }'
```

2. get server info
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 10,
      "method": "resources/read",
      "params": {
        "uri": "resource://mcp/server-info"
      }
    }'
```

3. get prompts info
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 11,
      "method": "resources/read",
      "params": {
        "uri": "resource://mcp/prompts"
      }
    }'
```

4. get resume generation prompt from resources
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 12,
      "method": "resources/read",
      "params": {
        "uri": "resource://mcp/prompts/resume_generation_prompt"
      }
    }'
```
### Prompts
1. List all prompts
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 5,
      "method": "prompts/list",
      "params": {}
    }'
```

3. get resume gerneation prompt
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 6,
      "method": "prompts/get",
      "params": {
        "name": "resume_generation_prompt",
        "arguments": {
          "job_description": "We need a backend engineer with FastAPI and AWS.",
          "matched_resumes": []
        }
      }
    }'
```

4. get job analysis prompt
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 7,
      "method": "prompts/get",
      "params": {
        "name": "job_analysis_prompt",
        "arguments": {
          "job_description": "We need a backend engineer with FastAPI and AWS."
        }
      }
    }'
```

5. get resume from source prompt
```sh
  curl -s http://localhost:8000/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{
      "jsonrpc": "2.0",
      "id": 8,
      "method": "prompts/get",
      "params": {
        "name": "resume_from_source_prompt",
        "arguments": {
          "job_description": "We need a backend engineer with FastAPI and AWS.",
          "resume_source": {"skills": ["FastAPI", "AWS"]},
          "match_summary": {"top_matches": ["FastAPI"]}
        }
      }
    }'
```
