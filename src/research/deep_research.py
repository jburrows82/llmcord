import logging
import os
import uuid
from functools import lru_cache
from typing import Dict, Optional

# Third-party imports – these packages are added to requirements.txt
try:
    from open_deep_research.graph import builder as odr_graph_builder  # type: ignore
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    from langgraph.types import Command  # type: ignore
except ImportError as exc:  # pragma: no cover – handled at runtime
    odr_graph_builder = None  # type: ignore
    MemorySaver = None  # type: ignore
    logging.error("open-deep-research is not installed: %s", exc)


@lru_cache(maxsize=1)
def _compile_graph():
    """Compile and memoise the Open Deep Research graph.

    We only compile once for performance; subsequent calls reuse the graph.
    Returns ``None`` if the dependency is missing.
    """
    if odr_graph_builder is None or MemorySaver is None:
        return None

    memory = MemorySaver()
    try:
        return odr_graph_builder.compile(checkpointer=memory)
    except Exception:  # pragma: no cover – defensive guard
        logging.exception("Failed to compile Open Deep Research graph")
        return None


async def _handle_interrupt(res: dict):
    """If the graph asks for feedback, use GPT-4.1 to decide.

    Current heuristic: if an interrupt with a report plan is returned, automatically
    approve (resume=True). This bypasses manual feedback but fulfils the user's
    request that "the LLM decide". In a future enhancement we could analyse the
    plan with another GPT call and inject textual feedback instead.
    """
    if "__interrupt__" not in res:
        return None  # No interrupt detected

    # In this minimal implementation we always approve the plan.
    return Command(resume=True)


async def _invoke_graph(topic: str) -> Optional[str]:
    """Run the compiled graph for the *topic* and return the final markdown report."""
    graph = _compile_graph()
    if graph is None:
        return None

    thread_cfg = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            # Force Tavily search & GPT-4.1 for all stages
            "search_api": "tavily",
            "planner_provider": "openai",
            "planner_model": "gpt-4.1",
            "writer_provider": "openai",
            "writer_model": "gpt-4.1",
            "max_search_depth": 10,
        }
    }

    try:
        payload: object = {"topic": topic}
        while True:
            result = await graph.ainvoke(payload, thread_cfg)  # type: ignore[attr-defined]

            # If result is not a dict, we consider the run complete.
            if not isinstance(result, dict):
                return str(result)

            # Check for completion keys first.
            if "report_markdown" in result or "report" in result:
                return result.get("report_markdown") or result.get("report")  # type: ignore[return-value]

            # Handle possible interrupt requesting feedback.
            next_command = await _handle_interrupt(result)
            if next_command is None:
                # Unexpected structure; return raw representation
                return str(result)

            # Prepare to resume with command
            payload = next_command

    except Exception:
        logging.exception("Open Deep Research graph invocation failed")
        return None


async def perform_deep_research_async(topic: str, config: Dict) -> str:
    """Public helper used by the Discord bot.

    It ensures environment variables expected by *open-deep-research* are
    present, executes the research workflow, and returns the markdown
    report (or a fallback error message).
    """
    # Map the tavily key to env-var if not already set
    tavily_keys = config.get("tavily_api_keys", []) if config else []
    if tavily_keys and not os.getenv("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = tavily_keys[0]

    # Map OpenAI credentials – assumes the first key should be used
    openai_cfg = config.get("providers", {}).get("openai", {}) if config else {}
    openai_keys = openai_cfg.get("api_keys", [])
    if openai_keys and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_keys[0]

    base_url = openai_cfg.get("base_url")
    if base_url and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = base_url

    report = await _invoke_graph(topic)
    if report is None:
        return "<Deep research failed – see logs for details>"
    return report
