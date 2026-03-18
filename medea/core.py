"""
Core Medea agent system implementations.

This module provides the main entry points for running Medea:
- medea(): Full multi-agent system with parallel or sequential execution
  Takes user_instruction and experiment_instruction as separate parameters
- experiment_analysis(): Research planning + in-silico experiment
- literature_reasoning(): Literature search and reasoning
"""

import os
import multiprocessing as mp

# Use 'spawn' to avoid fork() deadlock warnings in multi-threaded environments
_mp_ctx = mp.get_context("spawn")
from typing import Dict, Optional, Any
from medea.modules.langchain_agents import TaskPackage

# Optional: psutil for better process management
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def experiment_analysis(query: str, research_planning_module, analysis_module) -> tuple:
    """
    Execute research planning and code analysis using the agent system.

    Args:
        query: User's research question
        research_planning_module: Agent for generating research plans
        analysis_module: Agent for generating and executing in-silico experiments

    Returns:
        Tuple of (research_plan_text, analysis_response)

    Example:
        >>> from medea import experiment_analysis, ResearchPlanning, Analysis
        >>> research_planning_module = ResearchPlanning(...)
        >>> analysis_module = Analysis(...)
        >>> plan, result = experiment_analysis(
        ...     "Which gene is the best therapeutic target for RA?",
        ...     research_planning_module,
        ...     analysis_module
        ... )
    """
    from .modules.utils import Proposal

    # Generate research plan
    research_plan_task_dict = {"user_query": query}
    research_plan_taskpack = TaskPackage(instruction=str(research_plan_task_dict))

    max_agent_retries = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
    research_plan_response = None
    for attempt in range(max_agent_retries):
        try:
            research_plan_response = research_planning_module(research_plan_taskpack)
            if research_plan_response is not None:
                break
        except Exception as e:
            print(
                f"Research plan agent call failed (attempt {attempt + 1}/{max_agent_retries}): {e}",
                flush=True,
            )
            if attempt == max_agent_retries - 1:
                return "None", "None"

    # Execute experiment analysis if research plan is valid
    analysis_response, research_plan_text = "None", "None"
    if isinstance(research_plan_response, dict) and isinstance(
        research_plan_response.get("proposal_draft"), Proposal
    ):
        research_plan_text = research_plan_response["proposal_draft"].proposal

        analysis_task_dict = {"task": query, "instruction": research_plan_text}
        analysis_taskpack = TaskPackage(instruction=str(analysis_task_dict))

        for attempt in range(max_agent_retries):
            try:
                analysis_response = analysis_module(analysis_taskpack)
                if analysis_response is not None and analysis_response != "None":
                    break
            except Exception as e:
                print(
                    f"Analysis agent call failed (attempt {attempt + 1}/{max_agent_retries}): {e}",
                    flush=True,
                )
                if attempt == max_agent_retries - 1:
                    return research_plan_text, "None"

    return research_plan_text, analysis_response


def literature_reasoning(query: str, literature_module) -> Any:
    """
    Execute literature-based reasoning using the agent system.

    Args:
        query: User's research question
        literature_module: Agent for literature search and reasoning

    Returns:
        Reasoning response from the agent

    Example:
        >>> from medea import literature_reasoning, LiteratureReasoning
        >>> agent = LiteratureReasoning(...)
        >>> result = literature_reasoning(
        ...     "What are the therapeutic targets for RA?",
        ...     agent
        ... )
    """
    task_dict = {"user_query": query, "hypothesis": None}
    reason_taskpack = TaskPackage(instruction=str(task_dict))

    max_agent_retries = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
    reasoning_response = "None"
    for attempt in range(max_agent_retries):
        try:
            reasoning_response = literature_module(reason_taskpack)
            if reasoning_response is not None and reasoning_response != "None":
                break
        except Exception as e:
            import traceback
            print(
                f"Reasoning agent call failed (attempt {attempt + 1}/{max_agent_retries}): {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()

    return reasoning_response


# ============================================================================
# MULTIPROCESSING WRAPPERS
# ============================================================================


def _experiment_wrapper(inputs_for_coding, coding_result):
    """Wrapper for experiment analysis module in multiprocessing context.
    Runs research planning and analysis in two phases so the research plan
    is saved even if analysis times out."""
    import traceback
    from .modules.utils import Proposal

    query, research_planning_module, analysis_module = inputs_for_coding

    try:
        # Phase 1: Research planning (save immediately)
        research_plan_task_dict = {"user_query": query}
        research_plan_taskpack = TaskPackage(instruction=str(research_plan_task_dict))

        max_retries = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
        research_plan_response = None
        for attempt in range(max_retries):
            try:
                research_plan_response = research_planning_module(
                    research_plan_taskpack
                )
                if research_plan_response is not None:
                    break
            except Exception as e:
                print(
                    f"Research plan agent call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}",
                    flush=True,
                )
                traceback.print_exc()

        research_plan_text = "None"
        if isinstance(research_plan_response, dict) and isinstance(
            research_plan_response.get("proposal_draft"), Proposal
        ):
            research_plan_text = research_plan_response["proposal_draft"].proposal

        # Save research plan immediately — survives if analysis times out
        coding_result["research_plan"] = research_plan_text

        # Phase 2: Analysis (may be killed by timeout)
        analysis_response = "None"
        if research_plan_text != "None":
            analysis_task_dict = {"task": query, "instruction": research_plan_text}
            analysis_taskpack = TaskPackage(instruction=str(analysis_task_dict))

            for attempt in range(max_retries):
                try:
                    analysis_response = analysis_module(analysis_taskpack)
                    if analysis_response is not None and analysis_response != "None":
                        break
                except Exception as e:
                    print(
                        f"Analysis agent call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc()

        coding_result["data"] = (research_plan_text, analysis_response)
        coding_result["success"] = True
    except Exception as e:
        print(f"[CODING_PROCESS] Error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        coding_result["error"] = str(e)
        coding_result["success"] = False


def _reasoning_wrapper(inputs_for_reasoning, reasoning_result):
    """Wrapper for literature reasoning module in multiprocessing context."""
    try:
        result = literature_reasoning(*inputs_for_reasoning)
        reasoning_result["data"] = result
        reasoning_result["success"] = True
    except Exception as e:
        import traceback
        print(f"[REASONING_PROCESS] Error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        reasoning_result["error"] = str(e)
        reasoning_result["success"] = False


def medea(
    user_instruction: str,
    experiment_instruction: Optional[str] = None,
    research_planning_module=None,
    analysis_module=None,
    literature_module=None,
    debate_rounds: int = 2,
    panelist_llms: list = None,
    include_backbone_llm: bool = True,
    vote_merge: bool = True,
    full_instruction: bool = False,
    timeout: int = 2400,
    sequential: bool = False,
) -> Dict[str, Any]:
    """
    Execute full Medea multi-agent system with parallel or sequential execution.

    Runs research planning, in-silico experiment, and literature reasoning,
    then synthesizes results through multi-round panel discussion.

    Args:
        user_instruction: User's research question/instruction
        experiment_instruction: Optional additional experiment context and instructions (default: None)
        research_planning_module: Agent for generating research plans (default: None)
        analysis_module: Agent for in-silico experiment analysis (default: None)
        literature_module: Agent for literature-based reasoning (default: None)
        debate_rounds: Number of panel discussion rounds (default: 2)
        panelist_llms: List of LLM models for panel discussion (default: None)
        include_backbone_llm: Include backbone LLM in panel (default: True)
        vote_merge: Merge similar votes from different panelists (default: True)
        full_instruction: Use full query in panel or user instruction only (default: False)
        timeout: Timeout in seconds for each parallel process (default: 2400)
        sequential: Run agents sequentially instead of in parallel (default: False).
            Recommended when using a local LLM server with limited concurrency
            (e.g., Ollama with OLLAMA_NUM_PARALLEL=1).

    Returns:
        Dictionary containing:
            - 'P': Research plan text
            - 'PA': Analysis response
            - 'R': Literature reasoning response
            - 'final': Panel discussion hypothesis
            - 'llm': Panel LLM responses

    Example:
        >>> from medea import medea, AgentLLM, LLMConfig
        >>> from medea import ResearchPlanning, Analysis, LiteratureReasoning
        >>>
        >>> # Initialize agents
        >>> llm_config = LLMConfig({"temperature": 0.4})
        >>> llm = AgentLLM(llm_config)
        >>>
        >>> research_planning_module = ResearchPlanning(llm, actions=[...])
        >>> analysis_module = Analysis(llm, actions=[...])
        >>> literature_module = LiteratureReasoning(llm, actions=[...])
        >>>
        >>> # Run Medea (parallel, default)
        >>> result = medea(
        ...     user_instruction="Which gene is the best therapeutic target for RA in CD4+ T cells?",
        ...     research_planning_module=research_planning_module,
        ...     analysis_module=analysis_module,
        ...     literature_module=literature_module,
        ... )
        >>>
        >>> # Run Medea (sequential, for local LLMs)
        >>> result = medea(
        ...     user_instruction="Which gene is the best therapeutic target for RA?",
        ...     research_planning_module=research_planning_module,
        ...     analysis_module=analysis_module,
        ...     literature_module=literature_module,
        ...     sequential=True,
        ... )
        >>>
        >>> print(result['final'])  # Final hypothesis from panel discussion
    """
    from .modules.discussion import multi_round_discussion
    from .tool_space.agentic_tool import reset_call_budget

    # Reset per-sample caches and call budgets
    reset_call_budget()

    # Combine user instruction with experiment instruction for full query
    full_query = (
        user_instruction
        if experiment_instruction is None
        else user_instruction + " " + experiment_instruction
    )

    research_plan_text, analysis_response, literature_response = "None", "None", "None"

    if sequential:
        research_plan_text, analysis_response, literature_response = (
            _run_sequential(full_query, user_instruction, research_planning_module,
                            analysis_module, literature_module, timeout)
        )
    else:
        research_plan_text, analysis_response, literature_response = (
            _run_parallel(full_query, user_instruction, research_planning_module,
                          analysis_module, literature_module, timeout)
        )

    # Build output dict
    agent_output_dict = {}

    if research_plan_text != "None":
        agent_output_dict["P"] = research_plan_text

    if analysis_response != "None":
        agent_output_dict["PA"] = analysis_response

    if literature_response != "None":
        agent_output_dict["R"] = literature_response

    # Log the generated research plan if available
    if research_plan_text:
        print(f"[Research Plan]: {research_plan_text}\n", flush=True)

    # LLM-based Panel Discussion
    panel_query = user_instruction if not full_instruction else full_query

    # Default panelist LLMs if not provided
    if panelist_llms is None:
        from .tool_space.env_utils import get_panelist_llms
        panelist_llms = get_panelist_llms()

    # Each agent output is assigned an LLM to join panel discussion
    hypothesis_response, llm_hypothesis_response = multi_round_discussion(
        query=panel_query,
        include_llm=include_backbone_llm,
        mod="diff_context",
        panelist_llms=panelist_llms,
        proposal_response=research_plan_text,
        coding_response=analysis_response,
        reasoning_response=literature_response,
        vote_merge=vote_merge,
        round=debate_rounds,
    )

    agent_output_dict["llm"] = llm_hypothesis_response
    agent_output_dict["final"] = hypothesis_response

    return agent_output_dict


def _run_with_timeout(func, args, timeout, label):
    """Run a function in a thread with timeout. Returns the result or None.

    Uses threading instead of multiprocessing to avoid pickle issues with
    LangChain/Pydantic objects under the 'spawn' multiprocessing context.
    """
    import threading

    result_container = {"data": None, "success": False, "error": None}

    def _wrapper():
        try:
            result_container["data"] = func(*args)
            result_container["success"] = True
        except Exception as e:
            import traceback
            print(f"[MEDEA] {label} error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            result_container["success"] = False

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(
            f"Error: {label} exceeded {timeout}s timeout.",
            flush=True,
        )
        # Daemon thread will be cleaned up on process exit
        return None

    if result_container["success"]:
        return result_container["data"]
    return None


def _run_sequential(full_query, user_instruction, research_planning_module,
                    analysis_module, literature_module, timeout):
    """Run experiment analysis and literature reasoning sequentially with per-phase timeout."""
    research_plan_text, analysis_response, literature_response = "None", "None", "None"

    print(
        f"\n[MEDEA] Starting sequential execution (timeout={timeout}s per phase): "
        f"Research Planning + In-silico Experiment → Literature Reasoning",
        flush=True,
    )

    # Phase 1: Experiment analysis (research planning + code analysis)
    print(f"[MEDEA] Phase 1: Running experiment analysis...", flush=True)
    result = _run_with_timeout(
        experiment_analysis,
        (full_query, research_planning_module, analysis_module),
        timeout,
        "Experiment analysis",
    )
    if result is not None:
        research_plan_text, analysis_response = result
        if research_plan_text != "None":
            print(f"[MEDEA] ✓ Research plan completed", flush=True)
        if analysis_response != "None":
            print(f"[MEDEA] ✓ In-silico experiment completed", flush=True)
        else:
            print(f"[MEDEA] ⚠ In-silico experiment: no result", flush=True)
    else:
        print(f"[MEDEA] ⚠ Experiment analysis: no result", flush=True)

    # Phase 2: Literature reasoning
    print(f"[MEDEA] Phase 2: Running literature reasoning...", flush=True)
    result = _run_with_timeout(
        literature_reasoning,
        (user_instruction, literature_module),
        timeout,
        "Literature reasoning",
    )
    if result is not None:
        literature_response = result
        if literature_response != "None":
            print(f"[MEDEA] ✓ Literature reasoning completed", flush=True)
        else:
            print(f"[MEDEA] ⚠ Literature reasoning: no result", flush=True)
    else:
        print(f"[MEDEA] ⚠ Literature reasoning: no result", flush=True)

    return research_plan_text, analysis_response, literature_response


def _run_parallel(full_query, user_instruction, research_planning_module,
                  analysis_module, literature_module, timeout):
    """Run experiment analysis and literature reasoning in parallel with timeout."""
    research_plan_text, analysis_response, literature_response = "None", "None", "None"

    print(
        f"\n[MEDEA] Starting parallel execution: "
        f"Research Planning + In-silico Experiment + Literature Reasoning",
        flush=True,
    )

    inputs_for_coding = (full_query, research_planning_module, analysis_module)
    inputs_for_reasoning = (user_instruction, literature_module)

    # Single Manager for both result dicts (avoids spawning two server processes)
    manager = _mp_ctx.Manager()
    analysis_result = manager.dict()
    literature_result = manager.dict()

    # Start both processes with module-level wrapper functions
    print(f"[MEDEA] Launching in-silico experiment process...", flush=True)
    analysis_process = _mp_ctx.Process(
        target=_experiment_wrapper, args=(inputs_for_coding, analysis_result)
    )

    print(f"[MEDEA] Launching literature reasoning process...", flush=True)
    literature_process = _mp_ctx.Process(
        target=_reasoning_wrapper, args=(inputs_for_reasoning, literature_result)
    )

    analysis_process.start()
    print(
        f"[MEDEA] Experiment analysis process started (PID: {analysis_process.pid})",
        flush=True,
    )

    literature_process.start()
    print(
        f"[MEDEA] Literature reasoning process started (PID: {literature_process.pid})",
        flush=True,
    )
    print(f"[MEDEA] Both processes running in parallel...", flush=True)

    # Wait for analysis process with timeout
    analysis_process.join(timeout=timeout)
    if analysis_process.is_alive():
        print(
            f"Error: Analysis task exceeded {timeout}s timeout. Forcefully killing process...",
            flush=True,
        )
        _kill_process(analysis_process)

    # Wait for literature reasoning process with timeout
    literature_process.join(timeout=timeout)
    if literature_process.is_alive():
        print(
            f"Error: Literature reasoning task exceeded {timeout}s timeout. Forcefully killing process...",
            flush=True,
        )
        _kill_process(literature_process)

    # Extract results
    print(f"\n[MEDEA] Collecting results from parallel processes...", flush=True)

    if analysis_result.get("success", False):
        research_plan_text, analysis_response = analysis_result["data"]
        print(f"[MEDEA] ✓ Analysis process completed successfully", flush=True)
    else:
        # Even if analysis timed out or failed, try to recover the research plan
        if (
            "research_plan" in analysis_result
            and analysis_result["research_plan"] != "None"
        ):
            research_plan_text = analysis_result["research_plan"]
            print(
                f"[MEDEA] ⚠ Analysis timed out, but research plan was recovered",
                flush=True,
            )
        elif "error" in analysis_result:
            print(
                f"[MEDEA] ✗ Analysis process failed: {analysis_result['error']}",
                flush=True,
            )
        else:
            print(f"[MEDEA] ⚠ Analysis process: no result", flush=True)

    if literature_result.get("success", False):
        literature_response = literature_result["data"]
        print(
            f"[MEDEA] ✓ Literature reasoning process completed successfully", flush=True
        )
    elif "error" in literature_result:
        print(
            f"[MEDEA] ✗ Literature reasoning process failed: {literature_result['error']}",
            flush=True,
        )
    else:
        print(f"[MEDEA] ⚠ Literature reasoning process: no result", flush=True)

    manager.shutdown()
    return research_plan_text, analysis_response, literature_response


def _kill_process(process):
    """Forcefully kill a multiprocessing process and its children."""
    try:
        if PSUTIL_AVAILABLE:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            process.join(timeout=5)
        else:
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
    except psutil.NoSuchProcess if PSUTIL_AVAILABLE else Exception:
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()

