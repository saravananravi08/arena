"""
arena_agent.py — Agent Arena Multi-Turn Google ADK Agent
=========================================================

A fully autonomous multi-turn agent that navigates the Agent Arena:
  - Registers once, then loops indefinitely
  - Fetches the assigned task for the current level
  - Solves it and submits
  - On LEVEL_UP → fetches the next level's task and continues
  - On NO_TASKS or repeated failure → reports and exits cleanly
  - Prints a running scoreboard after each task attempt

The ADK agent drives all tool calls autonomously — the LLM decides
when to register, fetch, solve, submit, and when to stop.

Each turn in the multi-turn loop corresponds to one task attempt.
The session is preserved across turns so the LLM has full context
of everything that happened in previous turns.

Dependencies
------------
    pip install google-adk fastmcp traceloop-sdk google-genai

Environment variables
---------------------
    GEMINI_API_KEY       — required
    TRACELOOP_API_KEY    — optional

Usage
-----
    export GEMINI_API_KEY="your-key"
    python arena_agent.py
"""

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Optional

# ── Google ADK ────────────────────────────────────────────────────────────────
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# ── FastMCP ───────────────────────────────────────────────────────────────────
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport

# ── Traceloop ─────────────────────────────────────────────────────────────────
from traceloop.sdk import Traceloop, set_association_properties
from traceloop.sdk.decorators import workflow
from traceloop.sdk.tracing import set_conversation_id

# ── OTel logging ──────────────────────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.resources import Resource


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MCP_ENDPOINT = "https://agent-arena.dev/mcp"


ID_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjNiMDk1NzQ3YmY4MzMxZWE0YWQ1M2YzNzBjNjMyNjAxNzliMGQyM2EiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoic2FyYXZhbmFuIHJhdmkiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jTHQ2di16N05STjZPOFNrMEgxYkRpYlZka013aUVFaGRSQUlXY0gxcWlmTkJxWDJVQy09czk2LWMiLCJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vZ2RlLWFnZW50LWV2YWwiLCJhdWQiOiJnZGUtYWdlbnQtZXZhbCIsImF1dGhfdGltZSI6MTc3NzEwMTA3OCwidXNlcl9pZCI6Im9OdjN4RVVvbTFZUWtDd25vcFJWSDMydkhNczEiLCJzdWIiOiJvTnYzeEVVb20xWVFrQ3dub3BSVkgzMnZITXMxIiwiaWF0IjoxNzc3MTAxMDc4LCJleHAiOjE3NzcxMDQ2NzgsImVtYWlsIjoic2FyYXZhbmFucmF2aTExQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7Imdvb2dsZS5jb20iOlsiMTA4NTYwNDMyNDI5MDMzNjI3ODUyIl0sImVtYWlsIjpbInNhcmF2YW5hbnJhdmkxMUBnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJnb29nbGUuY29tIn19.ELTP7HudpnJlLSrYSCTS3drJdIwKleftOwT2j4QQ-zUnJ42s9iY2zZXv_-6CnRi18pD5TKDmFbJgPsebNFAaxrOMVY0YSqatdqVNmYYUrXrbeOz885IZsJSr5j7oVGQv_yTTHUKCibptXzSLpBL7agktISOSeo6XrH85Vni7YlVbXVRe1UimM4tfBHMA5RPnZkwvoW90BK9hfE6d19hSF3zpkE8_Jx0C-5t2QK3Ekh9ni2NkWClJ20F1-e2V5padAxu9dQK8QUhucRbf4ozE5AjCBZ_jJV3kV53gUNTA7mpLJ818IoGTLijqtHII7UewvVURe62NK5Su2qfPAcUIww"

AGENT_NAME = "Agent-ssp"
AGENT_STACK = "Python / Google ADK / Gemini 3.1 Flash Lite / Traceloop"
LINKEDIN_URL = "https://www.linkedin.com/in/saravananravi08/"  # ← update if needed
GITHUB_URL = "https://github.com/saravananravi08/arena"  # ← update if needed
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TRACELOOP_API_KEY = os.environ.get("TRACELOOP_API_KEY", "")

MAX_TURNS = 20  # safety cap — stops after this many task attempts


# ─────────────────────────────────────────────────────────────────────────────
# Run-scoped state  (shared across all tool calls in one run)
# ─────────────────────────────────────────────────────────────────────────────


class RunState:
    """Mutable state shared across all tool calls and turns in one run."""

    def __init__(self) -> None:
        self.run_id = str(uuid.uuid4())
        self.execution_id = str(uuid.uuid4())
        self.agent_id = ""
        self.task_id = ""
        self.conversation_id = ""

        # Scoreboard
        self.current_level = 1
        self.total_score = 0
        self.tasks_attempted = 0
        self.tasks_passed = 0
        self.level_history: list[dict] = []  # [{level, task_title, score, levelled_up}]

    def record(
        self, level: int, task_title: str, score: int, levelled_up: bool
    ) -> None:
        self.tasks_attempted += 1
        self.total_score += score
        if levelled_up or score >= 70:
            self.tasks_passed += 1
        if levelled_up:
            self.current_level = level + 1
        self.level_history.append(
            {
                "level": level,
                "task": task_title,
                "score": score,
                "levelled_up": levelled_up,
            }
        )

    def scoreboard(self) -> str:
        lines = [
            f"\n{'─'*55}",
            f"  SCOREBOARD  (run {self.run_id[:8]})",
            f"{'─'*55}",
            f"  Current Level : {self.current_level}",
            f"  Total Score   : {self.total_score}",
            f"  Tasks Done    : {self.tasks_attempted}  "
            f"(passed: {self.tasks_passed})",
            f"{'─'*55}",
        ]
        for entry in self.level_history:
            icon = (
                "✓" if entry["levelled_up"] else ("~" if entry["score"] >= 70 else "✗")
            )
            lines.append(
                f"  {icon} L{entry['level']}  {entry['task'][:35]:<35}  {entry['score']:>3}/100"
            )
        lines.append(f"{'─'*55}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


class _OtelOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        tid = getattr(record, "otelTraceID", "0")
        return tid not in ("0", "00000000000000000000000000000000", None, "")


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(h)
    return logger


agent_logger = _make_logger("arena.agent")
task_logger = _make_logger("arena.task")


# ─────────────────────────────────────────────────────────────────────────────
# Traceloop
# ─────────────────────────────────────────────────────────────────────────────


def init_tracing() -> None:
    Traceloop.init(
        app_name="arena-adk-agent",
        api_key=TRACELOOP_API_KEY or None,
        disable_batch=True,
        telemetry_enabled=False,
    )
    log_provider = LoggerProvider(
        resource=Resource.create({"service.name": "arena-adk-agent"})
    )
    exporter = ConsoleLogExporter()
    if TRACELOOP_API_KEY:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        exporter = OTLPLogExporter(
            endpoint="https://api.traceloop.com/v1/logs",
            headers={
                "Authorization": f"Bearer {TRACELOOP_API_KEY}",
                "x-traceloop-sdk-version": "traceloop-sdk",
            },
        )
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    for logger in (agent_logger, task_logger):
        h = LoggingHandler(logger_provider=log_provider)
        h.setLevel(logging.INFO)
        h.addFilter(_OtelOnlyFilter())
        logger.addHandler(h)
    print("[TRACELOOP] Initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# MCP helper
# ─────────────────────────────────────────────────────────────────────────────


async def _mcp_call(tool_name: str, arguments: dict, state: RunState) -> str:
    """Fresh MCP session per call — avoids timeout on long Gemini generation."""
    from fastmcp.exceptions import ToolError

    transport = StreamableHttpTransport(url=MCP_ENDPOINT)
    try:
        async with Client(transport=transport, name="arena-adk-agent") as client:
            set_association_properties(
                {
                    "execution.id": state.execution_id,
                    "run.id": state.run_id,
                    "agent.id": state.agent_id,
                    "task.id": state.task_id,
                    "agent.name": AGENT_NAME,
                    "agent.stack": AGENT_STACK,
                }
            )
            if state.conversation_id:
                set_conversation_id(state.conversation_id)

            result = await client.call_tool(tool_name, arguments)
            if result is None:
                return f"ERROR: {tool_name} returned no response"
            return "\n".join(
                getattr(b, "text", "")
                for b in result.content
                if getattr(b, "text", None)
            )
    except ToolError as e:
        # Return the error message as a string so the LLM can read it and
        # decide what to do next (e.g. skip_task on ALREADY_SUBMITTED).
        print(f"  [TOOL ERROR] {tool_name}: {e}")
        return f"ERROR: {e}"
    except Exception as e:
        print(f"  [MCP ERROR] {tool_name}: {e}")
        return f"ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool factory  — binds RunState into each tool closure
# ─────────────────────────────────────────────────────────────────────────────


def make_tools(state: RunState) -> list:
    """
    Return the four arena tool functions with RunState bound via closure.

    ADK registers plain async functions as tools — the LLM sees their
    docstrings and type hints. RunState is captured per-run so tools
    can update agent_id, task_id, and the scoreboard without globals.
    """

    async def register_agent(name: str, stack: str) -> str:
        """
        Register this agent in the Agent Arena.

        Call this exactly once at the start. Returns AGENT_ID and current level.
        If already registered the existing agent is returned — no duplicate created.

        Args:
            name:  Agent display name (use the configured AGENT_NAME).
            stack: Tech stack description.

        Returns:
            Server response containing AGENT_ID and current level.
        """
        result = await _mcp_call(
            "register_agent",
            {
                "idToken": ID_TOKEN,
                "name": name,
                "stack": stack,
                "linkedinUrl": LINKEDIN_URL,
                "githubUrl": GITHUB_URL,
            },
            state,
        )

        match = re.search(r"AGENT_ID:\s*(\S+?)\.?(\s|$)", result)
        if match:
            state.agent_id = match.group(1)
            state.conversation_id = state.agent_id
            set_association_properties(
                {"agent.id": state.agent_id, "run.id": state.run_id}
            )
            set_conversation_id(state.agent_id)

        level_match = re.search(r"Level[:\s]+(\d+)", result)
        if level_match:
            state.current_level = int(level_match.group(1))

        agent_logger.info(
            "Registered", extra={"agent_id": state.agent_id, "run_id": state.run_id}
        )
        print(
            f"  [register_agent] agent_id={state.agent_id}  level={state.current_level}"
        )
        return result

    async def get_tasks(agent_id: str) -> str:
        """
        Fetch the currently assigned task for this agent's level.

        Tasks are sticky — same task is returned until skip_task is called.
        Returns a JSON object: {id, title, description, level, points, difficulty}.

        Args:
            agent_id: The AGENT_ID returned by register_agent.

        Returns:
            JSON task object, or NO_TASKS if nothing available at this level.
        """
        result = await _mcp_call(
            "get_tasks",
            {
                "idToken": ID_TOKEN,
                "agentId": agent_id,
            },
            state,
        )

        try:
            data = json.loads(result)
            if isinstance(data, dict) and "id" in data:
                state.task_id = data["id"]
                state.conversation_id = f"{state.agent_id}-{state.task_id}"
                set_association_properties(
                    {"task.id": state.task_id, "execution.id": state.execution_id}
                )
                set_conversation_id(state.conversation_id)
                print(
                    f"  [get_tasks] task={state.task_id}  '{data.get('title')}'  L{data.get('level')}"
                )
        except json.JSONDecodeError:
            pass

        return result

    async def skip_task(agent_id: str, task_id: str, reason: str = "") -> str:
        """
        Abandon the current task and allow get_tasks to return a new one.

        Call this if the task is impossible or already submitted.
        After skipping, call get_tasks again to receive a fresh challenge.

        Args:
            agent_id: The AGENT_ID returned by register_agent.
            task_id:  The task ID to abandon.
            reason:   Optional reason (logged in audit trail).

        Returns:
            Confirmation from the server.
        """
        print(f"  [skip_task] skipping {task_id[:8]}  reason={reason[:50]}")
        return await _mcp_call(
            "skip_task",
            {
                "idToken": ID_TOKEN,
                "agentId": agent_id,
                "taskId": task_id,
                "reason": reason,
            },
            state,
        )

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """
        Submit the complete answer for the current task for AI evaluation.

        A score ≥ 70/100 promotes the agent to the next level (LEVEL_UP).
        Each task can only be submitted once. Make the answer thorough and complete.

        After submitting, always call get_tasks again to check if a new task
        is available at the next level, then continue until NO_TASKS is returned.

        Args:
            agent_id: The AGENT_ID from register_agent.
            task_id:  The task ID from get_tasks.
            content:  Full solution — detailed, correct, well-structured.

        Returns:
            Evaluation result: score, feedback, and LEVEL_UP status.
        """
        # Stamp execution on submission span
        new_exec = str(uuid.uuid4())
        state.execution_id = new_exec
        set_association_properties(
            {
                "execution.id": new_exec,
                "task.id": task_id,
                "agent.id": agent_id,
            }
        )

        task_logger.info(
            "Submitting",
            extra={
                "agent_id": agent_id,
                "task_id": task_id,
                "execution_id": new_exec,
            },
        )

        result = await _mcp_call(
            "submit_task",
            {
                "idToken": ID_TOKEN,
                "agentId": agent_id,
                "taskId": task_id,
                "executionId": new_exec,
                "content": content,
                "metadata": {
                    "agent_name": AGENT_NAME,
                    "agent_stack": AGENT_STACK,
                    "run_id": state.run_id,
                    "execution_id": new_exec,
                    "model": GEMINI_MODEL,
                },
            },
            state,
        )

        # Parse and record result
        score_match = re.search(r"Score:\s*(\d+)/100", result)
        score = int(score_match.group(1)) if score_match else -1
        levelled_up = "LEVEL_UP" in result

        # Fetch current task title from state
        task_title = state.task_id  # fallback to ID

        state.record(state.current_level, task_title, score, levelled_up)
        print(state.scoreboard())

        task_logger.info(
            "Submitted",
            extra={
                "agent_id": agent_id,
                "task_id": task_id,
                "score": score,
                "levelled_up": levelled_up,
            },
        )
        return result

    async def report_status() -> str:
        """
        Report the current agent status: level, score, and task history.

        Call this at any time to get a summary of progress so far.
        Always call this before stopping.

        Returns:
            Formatted status report.
        """
        return (
            f"Agent: {AGENT_NAME}  ID: {state.agent_id}\n"
            f"Level: {state.current_level}  Total Score: {state.total_score}\n"
            f"Tasks attempted: {state.tasks_attempted}  Passed: {state.tasks_passed}\n"
            f"History: {json.dumps(state.level_history, indent=2)}"
        )

    return [register_agent, get_tasks, skip_task, submit_task, report_status]


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an autonomous agent competing in the Agent Arena evaluation system.
Your goal is to navigate as many levels as possible by solving tasks.

LIFECYCLE (follow this exactly):

1. REGISTER: Call register_agent(name="{AGENT_NAME}", stack="{AGENT_STACK}")
   - Do this once only. Note the AGENT_ID and current level.
   - linkedinUrl and githubUrl are handled automatically — do not pass them.

2. LOOP — repeat until NO_TASKS or told to stop:
   a. Call get_tasks(agent_id) to fetch the current challenge.
   b. If NO_TASKS: call report_status() and stop.
   c. Read the task description carefully.
   d. Craft a high-quality, complete, technically correct answer IN YOUR RESPONSE.
      The answer must be thorough — the evaluator scores on correctness,
      efficiency, and robustness. Aim for 90+/100.
   e. Call submit_task(agent_id, task_id, content=<your full answer>).
   f. Read the result:
      - LEVEL_UP → call get_tasks again for the next level challenge.
      - Score < 70 → the task is NOT re-submittable. Call get_tasks to check
        if a new task appears (the server may assign a different one), then continue.
      - ALREADY_SUBMITTED or ERROR: ALREADY_SUBMITTED → call skip_task then get_tasks for a fresh challenge.
      - NO_TASKS → call report_status() and stop.

3. After stopping, always call report_status() to summarise progress.

RULES:
- Never submit the same task_id twice.
- Always use the task_id from the most recent get_tasks call.
- Make your answers as complete and detailed as possible — quality matters.
- Do not ask for confirmation — act autonomously.
""".strip()


def build_agent(state: RunState) -> LlmAgent:
    return LlmAgent(
        name="arena_agent",
        model=GEMINI_MODEL,
        instruction=SYSTEM_PROMPT,
        tools=make_tools(state),
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn runner
# ─────────────────────────────────────────────────────────────────────────────


async def run_turn(
    runner: Runner,
    session_service: InMemorySessionService,
    session_id: str,
    message: str,
) -> str:
    """
    Send one message to the agent and collect the final text response.

    The ADK runner streams events — tool calls, tool responses, and the
    final text reply. This function drives one complete turn and returns
    the agent's final text output for that turn.

    Args:
        runner:          The ADK Runner with the agent attached.
        session_service: The session service keeping conversation history.
        session_id:      Session ID — persists full conversation across turns.
        message:         User message to send this turn.

    Returns:
        The agent's final text response for this turn.
    """
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id="arena-user",
        session_id=session_id,
        new_message=content,
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args_str = str(dict(fc.args))
                preview = args_str[:80]
                print(f"  → [{fc.name}] {preview}{'...' if len(args_str) > 80 else ''}")

            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                resp_str = str(fr.response)[:100].replace("\n", " ")
                print(
                    f"  ← [{fr.name}] {resp_str}{'...' if len(str(fr.response)) > 100 else ''}"
                )

            elif hasattr(part, "text") and part.text and event.turn_complete:
                final_text = part.text

    return final_text


@workflow(name="arena_adk_run")
async def run() -> None:
    """
    Multi-turn workflow: autonomous arena navigation.

    The agent runs multiple turns, each building on the previous:
    Turn 1: Register + get first task + solve + submit
    Turn N: After level-up, get next task + solve + submit
    Final:  Report full scoreboard

    The ADK session preserves the full conversation history across turns
    so the LLM always has context of what it did before.
    """
    state = RunState()

    print(f"\n{'═' * 60}")
    print(f"  AGENT ARENA  —  Google ADK Multi-Turn Agent")
    print(f"{'═' * 60}")
    print(f"  Agent        : {AGENT_NAME}")
    print(f"  Model        : {GEMINI_MODEL}")
    print(f"  Run ID       : {state.run_id}")
    print(f"  Execution ID : {state.execution_id}")
    print(f"  Max turns    : {MAX_TURNS}")
    print(f"{'═' * 60}\n")

    set_association_properties(
        {
            "run.id": state.run_id,
            "execution.id": state.execution_id,
            "agent.name": AGENT_NAME,
            "agent.stack": AGENT_STACK,
        }
    )

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="arena-adk-agent",
        user_id="arena-user",
        session_id=state.run_id,
    )

    agent = build_agent(state)
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name="arena-adk-agent",
    )

    # ── Turn 1: kick off the full lifecycle ───────────────────────────────────
    print(f"[TURN 1] Starting autonomous lifecycle...\n")
    response = await run_turn(
        runner,
        session_service,
        state.run_id,
        "Start now. Register, fetch your task, solve it completely, and submit. "
        "After submitting, immediately fetch the next task and continue until "
        "NO_TASKS is returned.",
    )
    if response:
        print(f"\n[AGENT] {response[:500]}{'...' if len(response) > 500 else ''}")

    # ── Turns 2-N: continue after each level-up ───────────────────────────────
    # The agent may have stopped early (e.g. task still pending evaluation).
    # We nudge it to continue fetching and submitting.
    for turn in range(2, MAX_TURNS + 1):
        # Check if there's anything left to do
        if state.tasks_attempted == 0:
            break  # agent hasn't started — something went wrong

        print(f"\n[TURN {turn}] Nudging agent to continue...\n")
        response = await run_turn(
            runner,
            session_service,
            state.run_id,
            "Continue. Call get_tasks to check for the next challenge. "
            "If tasks are available, solve and submit. "
            "If NO_TASKS, call report_status and stop.",
        )
        if response:
            print(f"\n[AGENT] {response[:500]}{'...' if len(response) > 500 else ''}")

        # Stop if the agent reported completion
        if any(
            kw in response.lower()
            for kw in ("no_tasks", "no tasks", "stopped", "complete", "finished")
        ):
            print("\n[LOOP] Agent signalled completion.")
            break

        # Stop if no new tasks were attempted this turn
        # (avoid infinite loop if agent keeps failing to make progress)
        prev_attempted = state.tasks_attempted
        if state.tasks_attempted == prev_attempted and turn > 3:
            print("\n[LOOP] No new tasks attempted — stopping.")
            break

    # ── Final scoreboard ──────────────────────────────────────────────────────
    print(state.scoreboard())
    agent_logger.info(
        "Run complete",
        extra={
            "run_id": state.run_id,
            "total_score": state.total_score,
            "tasks_attempted": state.tasks_attempted,
            "final_level": state.current_level,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_tracing()
    asyncio.run(run())
