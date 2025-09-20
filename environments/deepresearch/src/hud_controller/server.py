"""Simple DeepResearch MCP server for HUD."""

import httpx
import os
from urllib.parse import urlparse
from hud.server import MCPServer
from hud.server.context import attach_context
from typing import List, Dict

mcp = MCPServer(name="deepresearch")
ctx = None


@mcp.initialize
async def init(init_ctx):
    global ctx
    ctx = attach_context("/tmp/hud_ctx.sock")


@mcp.shutdown
async def cleanup():
    global ctx
    ctx = None


@mcp.tool()
async def search(query: str) -> List[Dict[str, str]]:
    """
    Search the web for information and return titles and URLs using Exa API.

    Args:
        query: The search query string

    Returns:
        List of dictionaries containing 'title' and 'url' for each result
    """
    results = []
    max_results = 1  # Hardcoded to 1 result

    # Get Exa API key from environment
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        return [
            {
                "error": "Exa API key not found",
                "message": "Please set EXA_API_KEY environment variable",
                "instructions": "Get your API key from https://dashboard.exa.ai/home",
            }
        ]

    try:
        # Use Exa search API
        search_url = "https://api.exa.ai/search"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                search_url,
                headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
                json={
                    "query": query,
                    "numResults": max_results,
                    "type": "keyword",
                    "userLocation": "us",  # Bias results for US region
                    "contents": {
                        "text": {"maxCharacters": 1000}  # Get text snippets
                    },
                },
            )

            response.raise_for_status()
            data = response.json()

            # Extract results
            for result in data.get("results", []):
                title = result.get("title", "")
                url = result.get("url", "")

                if title and url:
                    results.append({"title": title, "url": url})

            # If no results, provide helpful feedback
            if not results:
                return [
                    {
                        "message": "No results found",
                        "query": query,
                        "autopromptString": data.get("autopromptString", query),
                    }
                ]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return [
                {
                    "error": "Invalid Exa API key",
                    "message": "Please check your EXA_API_KEY environment variable",
                    "status_code": str(e.response.status_code),
                }
            ]
        elif e.response.status_code == 429:
            return [
                {
                    "error": "Exa API rate limit exceeded",
                    "message": "Please wait before making more requests",
                    "status_code": str(e.response.status_code),
                }
            ]
        else:
            return [
                {
                    "error": f"Exa API error: {e.response.status_code}",
                    "message": str(e),
                    "response": e.response.text[:500],
                }
            ]
    except Exception as e:
        return [{"error": f"Search failed: {type(e).__name__}", "message": str(e)}]

    # Store search history in context
    ctx.add_search(query, results)

    return results


@mcp.tool()
async def fetch(url: str) -> str:
    """
    Fetch and extract content from a URL using Exa API, including summary, highlights, and full text.

    Args:
        url: The URL to fetch content from

    Returns:
        Formatted content including:
        - Summary with main takeaways
        - 3 key highlights (5 sentences each)
        - Full text content (truncated to 2500 characters)
    """
    max_length = 2500  # Hardcoded to 2500 characters
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return f"Invalid URL: {url}"

    # Get Exa API key
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        # Return failure message if no API key
        return "Fetch failed: No Exa API key available"

    try:
        # Use Exa contents API for reliable fetching
        contents_url = "https://api.exa.ai/contents"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                contents_url,
                headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
                json={
                    "urls": [url],
                    "text": {"maxCharacters": max_length, "includeHtmlTags": False},
                    "highlights": {"numSentences": 5, "highlightsPerUrl": 3},
                    "summary": {"query": "main takeaways"},
                    "livecrawl": "fallback",
                },
            )

            response.raise_for_status()
            data = response.json()

            # Extract text content, summary, and highlights
            results = data.get("results", [])
            if results and len(results) > 0:
                result = results[0]
                text = result.get("text", "")
                summary = result.get("summary", "")
                highlights = result.get("highlights", [])

                # Build formatted response
                formatted_content = []

                # Add summary
                if summary:
                    formatted_content.append("=== SUMMARY (Main Takeaways) ===")
                    formatted_content.append(summary)
                    formatted_content.append("")

                # Add highlights
                if highlights:
                    formatted_content.append("=== KEY HIGHLIGHTS ===")
                    for i, highlight in enumerate(highlights[:3], 1):
                        formatted_content.append(f"\nHighlight {i}:")
                        formatted_content.append(highlight)
                    formatted_content.append("")

                # Add main text (truncated if needed)
                if text:
                    formatted_content.append("=== FULL CONTENT ===")
                    if len(text) > max_length:
                        text = text[:max_length] + "...[truncated]"
                    formatted_content.append(text)

                final_content = "\n".join(formatted_content)

                # Store fetch history in context
                ctx.add_fetch(url, len(final_content))

                return final_content if final_content else "No content available"
            else:
                return "No content available for this URL"

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            # Invalid API key, return failure message
            return "Fetch failed: Invalid Exa API key"
        elif e.response.status_code == 429:
            return "Exa API rate limit exceeded. Please wait before fetching more content."
        else:
            return f"Exa API error: {e.response.status_code} - {e.response.text[:200]}"
    except Exception as e:
        # Return failure message on any error
        return f"Fetch failed: {type(e).__name__} - {str(e)}"


@mcp.tool()
async def answer(final_answer: str) -> str:
    """
    Submit the final answer to the research question.

    Args:
        final_answer: The agent's final answer to the task

    Returns:
        Confirmation message
    """
    ctx.submit_answer(final_answer)
    return f"Answer submitted: {final_answer}"


@mcp.tool()
async def setup() -> str:
    """Required for HUD environments. Initialize for a new task."""
    # Reset for a fresh task
    ctx.reset_stats()
    return ""


@mcp.tool()
async def evaluate(expected_answer: str) -> dict:
    """
    Required for HUD environments. Evaluates if the submitted answer contains the expected answer.

    Args:
        expected_answer: The correct answer to check against

    Returns:
        Evaluation result with reward and content string
    """
    submitted = ctx.get_submitted_answer()

    # Check if an answer was submitted
    if submitted is None:
        return {
            "reward": 0.0,
            "content": f"No answer submitted. Searches: {ctx.get_search_count()}, Fetches: {ctx.get_fetch_count()}",
        }

    # Check if expected answer is contained in submitted answer (case-insensitive)
    submitted_clean = submitted.strip().lower()
    expected_clean = expected_answer.strip().lower()

    # Accept if expected answer is contained within submitted answer
    is_correct = expected_clean in submitted_clean

    # Build result message
    result_msg = f"{'✅ Correct!' if is_correct else '❌ Incorrect.'} "
    result_msg += f"Submitted: '{submitted}', Expected: '{expected_answer}'. "
    result_msg += f"Stats: {ctx.get_search_count()} searches, {ctx.get_fetch_count()} fetches, {ctx.get_total_operations()} total operations."

    return {"reward": 1.0 if is_correct else 0.0, "content": result_msg}


if __name__ == "__main__":
    mcp.run()
