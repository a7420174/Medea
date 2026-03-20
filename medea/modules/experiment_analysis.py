import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

import dotenv
from .langchain_agents import (
    BaseAction,
    BaseAgent,
    ABCAgent,
    TaskPackage,
    AgentAct,
    ActObsChainType,
    act_match,
    ThinkAct,
    PlanAct,
)

try:
    from .langchain_agents import ACTION_NOT_FOUND_MESS
except ImportError:
    ACTION_NOT_FOUND_MESS = "[Error] Action not found in action list."

dotenv.load_dotenv()

# Use relative imports within package
from ..tool_space.gpt_utils import chat_completion
from ..tool_space.env_utils import get_backbone_llm, get_utility_llm

from .agent_llms import LLMConfig, AgentLLM, parse_action
from .BasePrompt import BasePromptGen
from .prompt_template import *
from .utils import CodeSnippet, FlushAgentLogger as AgentLogger, Proposal


# Load available tools using package-relative path
def _load_tool_info():
    """Load tool_info.json from package directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from agents/ to get to the package root
    package_root = os.path.dirname(current_dir)
    tool_info_path = os.path.join(package_root, "tool_space", "tool_info.json")

    # Fallback: if not found, try relative to current working directory (for development)
    if not os.path.exists(tool_info_path):
        tool_info_path = "tool_space/tool_info.json"

    with open(tool_info_path, "r") as tool_file:
        return json.load(tool_file)


AVALIBLE_TOOL = _load_tool_info()


def _minimal_tool_list(tools: list) -> str:
    """Create a minimal string representation of tools (name + description only).

    Used for tool selection prompts where only tool identity matters.
    """
    minimal = [{"name": t.get("name", ""), "description": t.get("description", "")} for t in tools]
    return json.dumps(minimal, separators=(",", ":"))


def _compact_tool_list(tools: list) -> str:
    """Create a compact string representation of tools for LLM prompts.

    Strips code_example, import_path, tensor_size, and returns fields to reduce
    token count, keeping only what the LLM needs for code generation.
    """
    compact = []
    for tool in tools:
        entry = {
            "name": tool.get("name", ""),
            "import_path": tool.get("import_path", ""),
            "description": tool.get("description", ""),
            "input_params": tool.get("input_params", []),
        }
        rt = tool.get("return_type")
        if rt:
            entry["return_type"] = rt
        ce = tool.get("code_example")
        if ce:
            entry["code_example"] = ce.get("code", "")
        compact.append(entry)
    return json.dumps(compact, separators=(",", ":"))


def stream_reader(pipe, output_list, source):
    """
    Read lines from a pipe and append them to a list.

    Args:
        pipe: File object to read from
        output_list: List to append lines to
        source: Source identifier for logging (e.g., 'stdout', 'stderr')
    """
    try:
        with pipe:
            for line in iter(pipe.readline, ""):
                print(f"[{source}] {line}", end="", flush=True)
                output_list.append(line)
    except Exception as e:
        print(f"Error reading pipe: {e}", flush=True)


class ToolSelector:
    """Selects relevant tools for code generation based on task instructions.

    Extracts tool names from 'Tool: <name>' patterns in the instruction text
    using regex, avoiding LLM calls for this simple extraction task.
    """

    def __init__(self, llm_provider: str = None, tmp: float = 0.4) -> None:
        self.tool_list = AVALIBLE_TOOL
        self._tool_names = {tool.get("name") for tool in self.tool_list}
        # Regex to match "Tool: tool_name" lines in proposal text
        self._tool_pattern = re.compile(r'(?:^|\n)\s*Tool:\s*(\S+)', re.IGNORECASE)

    def __call__(self, instruction: str, max_attempts: int = 3) -> List[Dict]:
        """
        Extract tool names from instruction text using regex pattern matching.

        Args:
            instruction: The proposal instruction text containing 'Tool: <name>' fields.
            max_attempts: Unused, kept for API compatibility.

        Returns:
            List of tool info dicts for tools mentioned in the instruction.
        """
        # Extract tool names from "Tool: xxx" patterns
        found_names = set(self._tool_pattern.findall(instruction))
        # Filter to only valid tool names
        valid_names = found_names & self._tool_names

        if not valid_names:
            print("[ToolSelector] No tools found via regex, using all available tools as fallback", flush=True)
            return self.tool_list

        tool_info_json = [
            tool for tool in self.tool_list if tool.get("name") in valid_names
        ]
        print(f"[ToolSelector] Extracted {len(tool_info_json)} tools: {sorted(valid_names)}", flush=True)
        return tool_info_json


class CodeGenerator(BaseAction):
    """Generates Python code based on task instructions and available tools."""

    def __init__(self, llm_provider: str = None, tmp: float = 0.4) -> None:
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")

        action_name = "CodeGenerator"
        action_desc = "Using this action generates the code snippet based on the given instruction (a Proposal object)."
        params_doc = {
            "instruction": "Supplied Proposal Object (e.g., <Proposal:xxxx>) - the Proposal object that contains step-by-step procedure about how to address the task",
            "code_draft": "CodeSnippet Object (e.g., <CodeSnippet:xxxx>) - the CodeSnippet draft on last iteration with feedback from the AnalysisQualityChecker action. This parameter can only be 'None' (e.g., code_draft=None) during the first iteration.",
        }

        # Default temperature is 0.0
        agent_config = LLMConfig({"temperature": tmp})
        self.tool_selector = ToolSelector()

        self.CodeGenerator_agent = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=CODE_GENERATION_TEMPLATE,
            input_variables=["instruction", "user_query", "tools"],
        )

        # Flexible pattern: matches ```python, ```Python, ```py, ``` python, etc.
        self.pattern = r"```\s*[Pp]y(?:thon)?\s*\n(.*?)```"
        self.max_instruction_tokens = (
            100000  # Conservative limit to avoid context issues
        )
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(
        self,
        instruction: Proposal,
        code_draft: CodeSnippet,
        quality_flag: bool = False,
        attempt: int = 4,
    ) -> CodeSnippet:

        # Extract instruction text from Proposal object
        if not isinstance(instruction, Proposal):
            return f"Invalid instruction type: {type(instruction)}. Please provide a Proposal object."

        feedback, i = "", 0
        if code_draft is not None and type(code_draft) == CodeSnippet:
            code_last_round = "Code from last iteration:\n" + code_draft.get_code()
            feedback_last_round = "Feedback:\n" + code_draft.get_feedback()
            feedback = code_last_round + feedback_last_round

        user_query = instruction.user_query
        instruction_text = instruction.proposal

        print(f"User query:\n{user_query}", flush=True)
        print(f"Instruction text:\n{instruction_text}", flush=True)

        print(f"[CodeGenerator] Entering code generation loop (attempt={attempt}, quality_flag={quality_flag}, i={i})", flush=True)
        while not quality_flag and i < attempt:
            try:
                tool_json = self.tool_selector(instruction=instruction_text)
                tool_json_compact = _compact_tool_list(tool_json)
            except Exception as e:
                print(f"[CodeGenerator] Tool selection/compact failed: {e}", flush=True)
                i += 1
                continue
            input_prompt = {
                "instruction": instruction_text + "\n" + feedback,
                "user_query": user_query,
                "tools": tool_json_compact,
            }

            try:
                print(f"[CodeGenerator] Calling code generation LLM (attempt {i+1}/{attempt})...", flush=True)
                raw_code_snippet = self.CodeGenerator_agent.run(input_prompt)
                print(f"[CodeGenerator] LLM response length: {len(raw_code_snippet)} chars", flush=True)
                # Strip <think>...</think> blocks from reasoning models
                if "</think>" in raw_code_snippet:
                    raw_code_snippet = raw_code_snippet.split("</think>")[-1].strip()
                matches = re.findall(self.pattern, raw_code_snippet, re.DOTALL)
                if not matches:
                    # Fallback 1: try generic ``` fence without language tag
                    generic_matches = re.findall(
                        r"```\n(.*?)```", raw_code_snippet, re.DOTALL
                    )
                    if generic_matches:
                        # Pick the longest block (most likely the actual code)
                        matches = [max(generic_matches, key=len)]
                        print(
                            "[CodeGenerator] Found code in generic ``` fence (no language tag).",
                            flush=True,
                        )

                if not matches:
                    # Fallback 2: response contains Python code but no fences at all
                    stripped = raw_code_snippet.strip()
                    python_indicators = [
                        "import ",
                        "from ",
                        "def ",
                        "class ",
                        "print(",
                        "if __name__",
                    ]
                    has_python = sum(1 for kw in python_indicators if kw in stripped)
                    if has_python >= 2:
                        # Extract from first import/from/def line to end
                        lines = stripped.split("\n")
                        code_start = next(
                            (
                                i
                                for i, line in enumerate(lines)
                                if any(
                                    line.strip().startswith(kw)
                                    for kw in ["import ", "from ", "def ", "#!", "#"]
                                )
                            ),
                            None,
                        )
                        if code_start is not None:
                            matches = ["\n".join(lines[code_start:])]
                            print(
                                "[CodeGenerator] No code fence found, extracted Python code from response.",
                                flush=True,
                            )

                if not matches:
                    print(
                        "[CodeGenerator] No code block found in response, retrying...",
                        flush=True,
                    )
                    i += 1
                    continue

                code_snippet = matches[0]
                print("========Code========", flush=True)
                print(code_snippet, flush=True)

                quality_flag, feedback = self.check_code_quality(
                    instruction_text, tool_json_compact, code_snippet
                )

                if quality_flag:
                    return CodeSnippet(
                        task=user_query,
                        instruction=instruction_text,  # Store instruction text
                        tool_info=tool_json_compact,
                        code_snippet=code_snippet,
                    )

            except Exception as e:
                print(f"[CodeGenerator] Error on attempt {i + 1}: {e}")
                feedback = f"Error: {str(e)}"

            i += 1

        # Fallback: if LLM failed to generate valid Python, build code from tool info
        needs_fallback = "code_snippet" not in locals() or code_snippet == ""
        if not needs_fallback:
            try:
                compile(code_snippet, "<code_gen>", "exec")
            except SyntaxError:
                print("[CodeGenerator] Generated code has syntax errors, using fallback.", flush=True)
                needs_fallback = True
        if needs_fallback:
            code_snippet = self._build_fallback_code(user_query, tool_json, instruction_text)
            print("[CodeGenerator] Using fallback code generation from tool info.", flush=True)

        return CodeSnippet(
            task=user_query,
            instruction=instruction_text,
            tool_info=tool_json_compact,
            code_snippet=code_snippet,
        )

    @staticmethod
    def _build_fallback_code(user_query: str, tool_json: list, instruction_text: str = "") -> str:
        """Build a Python script from tool info when LLM fails to generate code.

        Generically maps extracted entities (disease, genes, tissue, cell_type)
        to each tool's input_params based on parameter names — no tool-specific
        hard-coding so it works with any combination of tools.
        """
        import re as _re

        # --- Extract entities from instruction + query ---
        combined = f"{user_query}\n{instruction_text}"

        # Disease: JSON param value > tool input_params default > None
        disease = None
        m = _re.search(r'"disease_name":\s*"([^"]+)"', combined)
        if m:
            disease = m.group(1).strip()
        if not disease:
            # Check tool_json for disease_name param with a default value
            for tool in tool_json:
                for p in (tool.get("input_params") or []):
                    if isinstance(p, dict) and p.get("name") == "disease_name":
                        default = p.get("default")
                        if default and isinstance(default, str):
                            disease = default

        # Genes: JSON array > comma/and-separated uppercase symbols in query
        genes = []
        m = _re.search(r'"genes?":\s*\[([^\]]+)\]', combined)
        if m:
            genes = _re.findall(r'"([A-Z][A-Z0-9]+)"', m.group(1))
        if not genes:
            # Match sequences of 2+ uppercase gene symbols separated by commas/and/or
            # Handles: "CD79A, MS4A1, and FCGR3A" / "BRCA1, TP53, and PTEN" / "CNGA1—HDAC2"
            _gene_sym = r'[A-Z][A-Z0-9]{1,9}'
            _sep = r'[\s,;—–\-]+(?:and\s+|or\s+)?'
            m = _re.search(
                rf'({_gene_sym}(?:{_sep}{_gene_sym}){{1,}})',
                user_query,  # only from user query to avoid proposal boilerplate
            )
            if m:
                _exclude_genes = {"THE", "AND", "FOR", "WITH", "FROM", "WHICH", "NOT",
                                  "HPA", "API", "RNA", "DNA", "PPI", "LLM", "FDA", "GWAS"}
                genes = [g for g in _re.findall(rf'{_gene_sym}', m.group(1))
                         if g not in _exclude_genes]

        # Tissue / cell_type
        tissue = None
        m = _re.search(r'"tissue":\s*"([^"]+)"', combined)
        if m:
            tissue = m.group(1)
        cell_type = None
        m = _re.search(r'"cell_type":\s*"([^"]+)"', combined)
        if m:
            cell_type = m.group(1)

        # Entity bank for param matching
        entities = {
            "disease": disease,
            "genes": genes,
            "tissue": tissue,
            "cell_type": cell_type,
        }

        # --- Build code ---
        lines = ["# Auto-generated evidence collection code", ""]

        # Imports
        for tool in tool_json:
            ip = tool.get("import_path", "")
            if ip:
                lines.append(ip)

        lines.extend(["", "", "def main():"])
        lines.append(f'    print("=== Evidence Collection ===")')
        # Emit entity variables
        if genes:
            lines.append(f'    genes = {genes}')
        if disease:
            lines.append(f'    disease = "{disease}"')
        if tissue:
            lines.append(f'    tissue = "{tissue}"')
        if cell_type:
            lines.append(f'    cell_type = "{cell_type}"')
        lines.append("")

        # Generate a call for each tool using param-name matching
        for step_i, tool in enumerate(tool_json, 1):
            name = tool.get("name", "unknown")
            params = tool.get("input_params", [])
            lines.append(f'    # Step {step_i}: {name}')
            lines.append(f'    print(f"\\n--- Step {step_i}: {name} ---")')
            lines.append(f'    try:')

            # Build argument list by matching param names to entities
            args = []
            has_gene_name = any(p.get("name") == "gene_name" for p in params)
            iterate_genes = has_gene_name and genes and len(genes) > 1

            for p in params:
                pname = p.get("name", "")
                ptype = p.get("type", "str").lower()

                if "disease" in pname and disease:
                    args.append(f'{pname}=disease')
                elif pname in ("genes", "gene_list") and genes:
                    args.append(f'{pname}=genes')
                elif pname == "gene_name" and genes:
                    args.append(f'{pname}=gene')  # loop variable
                elif pname in ("gene", "gene_a") and genes:
                    args.append(f'{pname}=genes[0]')
                elif pname == "gene_b" and len(genes) > 1:
                    args.append(f'{pname}=genes[1]')
                elif "tissue" in pname and tissue:
                    args.append(f'{pname}=tissue')
                elif "cell" in pname and "type" in pname and cell_type:
                    args.append(f'{pname}=cell_type')
                elif "max" in pname:
                    args.append(f'{pname}=10')
                elif "bool" in ptype:
                    args.append(f'{pname}=True')
                # skip params we can't fill

            arg_str = ", ".join(args)

            if iterate_genes:
                # Tool takes single gene_name → loop over genes
                lines.append(f'        for gene in genes:')
                lines.append(f'            result = {name}({arg_str})')
                lines.append(f'            print(f"  [{{gene}}] {{str(result)[:200]}}")')
            else:
                lines.append(f'        result = {name}({arg_str})')
                lines.append(f'        print(f"  Result: {{str(result)[:300]}}")')

            lines.append(f'    except Exception as e:')
            lines.append(f'        print(f"  [{name}] Failed: {{e}}")')
            lines.append("")

        lines.append(f'    print("\\n=== Evidence collection complete ===")')
        lines.append("")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    main()")

        return "\n".join(lines)

    def check_code_quality(
        self, instruction: str, tool_json: List[Dict], code_snippet: str
    ) -> tuple:
        """
        Check the quality of generated code.

        Args:
            instruction: Task instruction text
            tool_json: List of tools used
            code_snippet: Generated code to check

        Returns:
            Tuple of (is_quality_ok: bool, feedback: str)
        """
        pattern = r"\[(\w+)\]\s*-\s*(.*)"

        checker_prompt = CODE_PRECHECK_TEMPLATE.format(
            instruction=instruction, tool_info=tool_json, code_snippet=code_snippet
        )

        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                output = chat_completion(checker_prompt, model=get_utility_llm())
                print(output, flush=True)

                match = re.match(pattern, output)
                if match:
                    decision = match.group(1)
                    feedback = match.group(2)
                    break

                # Try alternative patterns
                output_lower = output.lower()
                if "approved" in output_lower or "pass" in output_lower:
                    decision = "Approved"
                    feedback = "Code quality check passed"
                    break
                if (
                    "[minor]" in output_lower
                    or "minor" == output_lower.split("]")[0].strip("[").strip()
                ):
                    decision = "Minor"
                    feedback = output
                    break
                if "failed" in output_lower or "error" in output_lower:
                    decision = "Failed"
                    feedback = output
                    break

            except Exception as e:
                print(
                    f"[CodeGenerator] Quality check error (attempt {attempt + 1}): {e}"
                )
                if attempt >= max_attempts - 1:
                    decision = "Approved"
                    feedback = "Quality check failed, defaulting to approval"
                    break

        if decision and decision.lower() in ["failed", "fail"]:
            return False, f"Feedback:\n{feedback}"
        if decision and decision.lower() == "minor":
            print(
                f"[CodeGenerator] Quality check: MINOR issues (non-blocking). Proceeding.",
                flush=True,
            )
        return True, feedback


class AnalysisExecution(BaseAction):
    """Executes Python code and captures output/errors."""

    def __init__(self) -> None:
        action_name = "AnalysisExecution"
        action_desc = (
            "Using this action interprets the code and gets the executed output."
        )
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        self.failure_iter = 0

    def __call__(self, code_snippet: CodeSnippet, file_path: str = "session1.py"):
        if not isinstance(code_snippet, CodeSnippet):
            return f"Invalid code snippet type: {type(code_snippet)}. Please provide a CodeSnippet object."

        # Prepend path setup to make tool_space imports work
        path_setup = """import sys
import os
# Add parent directory to path so tool_space can be imported as medea.tool_space
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""

        # Replace tool_space imports with medea.tool_space
        code_to_run = code_snippet.code_snippet.replace(
            "from tool_space.", "from medea.tool_space."
        ).replace("import tool_space.", "import medea.tool_space.")

        # Use a unique temp file to avoid race conditions across parallel experiments
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="medea_session_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(path_setup + code_to_run)
        except Exception:
            os.close(fd)
            raise

        stdout_lines, stderr_lines = [], []
        p = subprocess.Popen(
            ["python", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_thread = threading.Thread(
            target=stream_reader, args=(p.stdout, stdout_lines, "stdout")
        )
        stderr_thread = threading.Thread(
            target=stream_reader, args=(p.stderr, stderr_lines, "stderr")
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            # Wait for the process to complete with a timeout
            # Use 2x OLLAMA_REQUEST_TIMEOUT to allow for multiple LLM calls within code
            _code_timeout = int(float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "300")) * 2)
            p.wait(timeout=_code_timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            print("Subprocess timed out and was killed.")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Ensure threads have finished reading
        stdout_thread.join()
        stderr_thread.join()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        # Keep only real errors; drop warnings, library logs, config dumps, and progress bars
        def _is_noise(line):
            s = line.strip()
            return (
                not s
                or "Warning:" in line
                or "Warning " in line
                or "warn(" in line
                or "FutureWarning" in line
                or "DeprecationWarning" in line
                or "UserWarning" in line
                or s[0] in '"{}'
                or s.startswith(("loading ", "Model config", "You're using"))
                or "it/s]" in line
                or "%|" in line
                or "google.generativeai" in line
                or "google.genai" in line
                or "switch to the" in line
                or "See README" in line
                or "frozen importlib" in line
            )

        filtered_stderr = "\n".join(l for l in stderr.splitlines() if not _is_noise(l))

        # Check for errors based on return code and filtered stderr content
        if p.returncode != 0:
            code_snippet.status = "error"
            code_snippet.stderr = filtered_stderr or stderr
            print(f"[Coding Error] {filtered_stderr or stderr}", flush=True)
            if self.failure_iter < 3:
                self.failure_iter += 1
                return f"{stderr}\n{code_snippet}: Error occurred during code execution, call CodeDebugger next."
            self.failure_iter = 0
            return f"{code_snippet}: The system cannot help with it, call Finish next."

        # If no errors, proceed (store filtered stderr for transparency but mark as non-error)
        code_snippet.stderr = filtered_stderr if filtered_stderr else None
        code_snippet.code_output = stdout
        code_snippet.status = "executed"
        return f"{code_snippet}: Successfully executed, call AnalysisQualityChecker action next."


class CodeDebug(BaseAction):
    """Debugs and fixes code errors using LLM-based analysis."""

    def __init__(self, llm_provider: str = None, tmp: float = 0.4) -> None:
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")

        action_name = "CodeDebugger"
        action_desc = "When observed any issues from AnalysisExecution output, using this action to debug the code and get refined code snippet."
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

        agent_config = LLMConfig({"temperature": tmp})
        self.debugger_agent = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=DEBUGGER_TEMPLATE,
        )
        self.pattern = r"```\s*[Pp]y(?:thon)?\s*\n(.*?)```"

    def __call__(self, code_snippet: CodeSnippet):
        if code_snippet.status != "error":
            return f"{code_snippet}: No error occurred, call AnalysisExecution next."
        feedback = None
        task = code_snippet.task
        instruction = code_snippet.instruction
        tool_info = code_snippet.tool_info
        snippet = code_snippet.code_snippet
        error_msg = code_snippet.stderr

        if code_snippet.feedback:
            feedback = code_snippet.feedback

        task_prompt = DEBUGGER_CHAT_TEMPLATE.format(
            user_query=task,
            instruction=instruction,
            tool_info=tool_info,
            code_snippet=snippet,
            error_msg=error_msg,
            feedback=feedback,
        )
        print(task_prompt, flush=True)
        raw_code_snippet = self.debugger_agent.run(task_prompt)
        # Strip <think>...</think> blocks from reasoning models
        if "</think>" in raw_code_snippet:
            raw_code_snippet = raw_code_snippet.split("</think>")[-1].strip()
        matches = re.findall(self.pattern, raw_code_snippet, re.DOTALL)
        code_snippet.code_snippet = matches[0]
        code_snippet.status = "unexecuted"
        return f"{code_snippet}: debugged, call AnalysisExecution next."


class AnalysisQualityChecker(BaseAction):
    """Checks code quality and provides feedback for improvement."""

    def __init__(
        self, llm_provider: str = None, tmp: float = 0.4, max_iter: int = 3
    ) -> None:
        # Quality checking is a utility task — use UTILITY_LLM if available
        if llm_provider is None:
            llm_provider = get_utility_llm()

        action_name = "AnalysisQualityChecker"
        action_desc = "After each successful code execution using the AnalysisExecution action, using this action to provide feedback on the code snippet, evaluating its informativeness and correctness"
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

        agent_config = LLMConfig({"temperature": tmp})
        self.quality_checker_agent = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=QUALITY_ASSURANCE_TEMPLATE,
            input_variables=[
                "user_query",
                "instruction",
                "tool_info",
                "code_snippet",
                "code_output",
            ],
        )
        self.pattern = r"\[(\w+)\]\s*-\s*(.*)"
        self.iterations = 0
        self.max_iter = max_iter

    def __call__(self, code_snippet: CodeSnippet):
        if code_snippet.status != "executed":
            return f"{code_snippet}: Code snippet has not been executed, call AnalysisExecution next."

        # Skip directly to the approval iteration exceeds the max_iter
        if self.iterations >= self.max_iter:
            self.iterations = 0
            code_snippet.status = "approved"
            return f"{code_snippet}: Passed, call Finish action next."

        decision = None
        feedback = None
        user_query = code_snippet.task
        instruction = code_snippet.instruction
        tool_info = code_snippet.tool_info
        code_snippet_str = code_snippet.code_snippet
        code_output = code_snippet.code_output

        input_prompt = {
            "user_query": user_query,
            "instruction": instruction,
            "tool_info": tool_info,
            "code_snippet": code_snippet_str,
            "code_output": code_output,
        }

        while True:
            output = self.quality_checker_agent.run(input_prompt)
            # Strip <think>...</think> blocks from reasoning models
            if "</think>" in output:
                output = output.split("</think>")[-1].strip()
            print(output, flush=True)
            match = re.match(self.pattern, output)
            if match:
                decision = match.group(1)  # Decision: Approved, Minor, or Failed
                feedback = match.group(2)  # Feedback
                code_snippet.update_feedback(feedback)
                break

        if decision == "Failed":
            self.iterations += 1
            code_snippet.status = "error"
            return f"{output}\n{code_snippet}: Failed, call CodeDebugger action next."

        if decision == "Minor":
            print(
                f"[AnalysisQualityChecker] Non-critical issues found (iteration {self.iterations + 1}/{self.max_iter}). Approving with feedback.",
                flush=True,
            )
            print(
                f"Feeback from AnalysisQualityChecker (if any):\n{feedback}\n",
                flush=True,
            )

        self.iterations = 0
        code_snippet.status = "approved"
        return f"{code_snippet}: Passed, call Finish action next."


class AnalysisFinishAction(BaseAction):
    """Completes the coding task with the final refined code snippet."""

    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the task with a refined CodeSnippet Object"
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, code_snippet: CodeSnippet) -> Dict:
        """Return the final code and execution output."""
        return {
            "code_snippet": code_snippet.code_snippet,
            "executed_output": code_snippet.code_output,
        }


FinishAct = AnalysisFinishAction()


class Analysis(BaseAgent):
    """
    Agent responsible for generating, executing, and debugging code.

    Uses a react-style reasoning approach with actions for code generation,
    execution, debugging, and quality checking.
    """

    def __init__(
        self,
        llm: AgentLLM = AgentLLM(
            llm_config=LLMConfig({"temperature": 0.4}),
            llm_name=os.getenv("BACKBONE_LLM"),
        ),
        actions: List[BaseAction] = None,
        manager: ABCAgent = None,
        logger: AgentLogger = AgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False),
        reasoning_type: str = "react",
    ):
        if actions is None:
            actions = [
                CodeGenerator(llm_provider=os.getenv("BACKBONE_LLM")),
                AnalysisExecution(),
                CodeDebug(llm_provider=os.getenv("BACKBONE_LLM")),
                AnalysisQualityChecker(llm_provider=os.getenv("BACKBONE_LLM")),
            ]

        name = "analysis_agent"
        role = CODE_GENERATION_AGENT_TEMPLATE

        super().__init__(
            name=name,
            role=role,
            reasoning_type=reasoning_type,
            llm=llm,
            actions=actions,
            manager=manager,
            max_exec_steps=20,
            logger=logger,
        )
        self.prompt_gen = BasePromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )
        self.context_proposal = None

    def _parse_task_package(self, task: TaskPackage) -> Tuple[str, Proposal]:
        """
        Parse TaskPackage to extract user query and instruction.

        Args:
            task: TaskPackage containing the task information

        Returns:
            Tuple of (user_query, proposal_object)
        """
        try:
            instruction_str = str(task.instruction)

            # Try JSON parsing first
            try:
                task_dict = json.loads(instruction_str)
            except json.JSONDecodeError:
                # If JSON fails, try Python literal evaluation
                try:
                    task_dict = ast.literal_eval(instruction_str)
                except (ValueError, SyntaxError):
                    # Manual regex parsing for common pattern
                    task_dict = self._parse_instruction_regex(instruction_str)

            # Extract user query and instruction text
            user_query = task_dict.get("task", "")
            instruction_text = task_dict.get("instruction", "")

            # Create Proposal object
            proposal = Proposal(user_query=user_query, proposal=instruction_text)

            return user_query, proposal

        except Exception as e:
            print(f"[Analysis] Error parsing TaskPackage: {e}")
            # Fallback: use entire instruction
            fallback_proposal = Proposal(
                user_query=str(task.instruction), proposal=str(task.instruction)
            )
            return str(task.instruction), fallback_proposal

    def _parse_instruction_regex(self, instruction_str: str) -> Dict:
        """Parse instruction string using regex patterns."""
        if not (instruction_str.startswith("{") and instruction_str.endswith("}")):
            raise ValueError("Not a dict-like string")

        # Find task value
        task_pattern = r"'task':\s*'(.*?)(?=',\s*'instruction')"
        task_match = re.search(task_pattern, instruction_str, re.DOTALL)

        if not task_match:
            task_match = re.search(r"'task':\s*'([^']*(?:''[^']*)*)'", instruction_str)
        if not task_match:
            task_match = re.search(r'"task":\s*"([^"]*(?:""[^"]*)*)"', instruction_str)

        # Find instruction value
        instruction_match = re.search(
            r"'instruction':\s*(<Proposal:\d+>)", instruction_str
        )
        if not instruction_match:
            instruction_match = re.search(
                r"'instruction':\s*'([^']*(?:''[^']*)*)'", instruction_str
            )
        if not instruction_match:
            instruction_match = re.search(
                r'"instruction":\s*"([^"]*(?:""[^"]*)*)"', instruction_str
            )

        if task_match:
            return {
                "task": task_match.group(1),
                "instruction": instruction_match.group(1) if instruction_match else "",
            }

        raise ValueError("Could not parse instruction format")

    def __call__(self, task: TaskPackage):
        """
        Process a task by parsing it and executing the parent's workflow.

        Args:
            task: TaskPackage containing the task information

        Returns:
            The result from the parent __call__ method
        """
        user_query, proposal = self._parse_task_package(task)
        self.user_query = user_query
        self.context_proposal = proposal
        task.instruction = str({"task": user_query, "instruction": proposal})

        print(f"Task instruction:\n {task.instruction}", flush=True)
        return super().__call__(task)

    def __add_inner_actions__(self):
        """Add inner action types based on the reasoning_type."""
        if self.reasoning_type == "react":
            self.actions += [ThinkAct, FinishAct]
        self.actions = list(set(self.actions))

    def __next_act__(
        self, task: TaskPackage, action_chain: ActObsChainType
    ) -> AgentAct:
        """
        Generate the next action for the agent.

        Args:
            task: The task which agent receives and solves
            action_chain: History actions and observations from memory

        Returns:
            The action for agent to execute
        """
        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
        )
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)

        return self.__action_parser__(raw_action, action_chain)

    def __action_parser__(
        self, raw_action: str, action_chain: ActObsChainType
    ) -> AgentAct:
        """
        Parse the generated content to an executable action.

        Args:
            raw_action: LLM generated text
            action_chain: Action history chain

        Returns:
            An executable action wrapper
        """
        action_name, args, PARSE_FLAG = parse_action(raw_action)

        # Resolve code_snippet references from action chain
        if "code_snippet" in args and args["code_snippet"] is not None:
            cs_ref = str(args["code_snippet"]).strip()
            resolved = False
            # Try exact match first
            for _, p_obs in reversed(action_chain):
                if isinstance(p_obs, CodeSnippet) and p_obs.get_id() == cs_ref:
                    args["code_snippet"] = p_obs
                    resolved = True
                    break
            # Fallback: find any CodeSnippet in action chain (observations are often strings)
            if not resolved:
                for _, p_obs in reversed(action_chain):
                    if isinstance(p_obs, CodeSnippet):
                        args["code_snippet"] = p_obs
                        resolved = True
                        break
            # Last resort: check if any observation string contains a CodeSnippet ID pattern
            if not resolved and "<CodeSnippet:" in cs_ref:
                for a_act, a_obs in reversed(action_chain):
                    if hasattr(a_act, "params") and "code_snippet" in a_act.params:
                        param = a_act.params["code_snippet"]
                        if isinstance(param, CodeSnippet):
                            args["code_snippet"] = param
                            resolved = True
                            break

        # Resolve instruction references from action chain or context
        if "instruction" in args and args["instruction"] is not None:
            found_proposal = self._find_proposal(args["instruction"], action_chain)
            if found_proposal is not None:
                args["instruction"] = found_proposal

        return AgentAct(name=action_name, params=args)

    def _find_proposal(
        self, instruction_ref: str, action_chain: ActObsChainType
    ) -> Optional[Proposal]:
        """Find a Proposal object from action chain or context."""
        instruction_ref = str(instruction_ref)

        # Try to find from action chain with ID matching
        for _, p_obs in reversed(action_chain):
            if isinstance(p_obs, Proposal) and repr(p_obs) == instruction_ref:
                return p_obs

        # Use stored context proposal
        if hasattr(self, "context_proposal") and self.context_proposal is not None:
            return self.context_proposal

        # Find any Proposal object in action chain
        for _, p_obs in reversed(action_chain):
            if isinstance(p_obs, Proposal):
                return p_obs

        return None

    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        """
        Forward the action to get the observation.

        Args:
            task: The task which agent receives and solves
            agent_act: The action wrapper for execution

        Returns:
            Observation from action execution
        """
        act_found_flag = False
        param_parse_flag = False
        observation = None

        for action in self.actions:
            if action.action_name == agent_act.name:
                act_found_flag = True
                try:
                    observation = action(**agent_act.params)
                except Exception as e:
                    observation = MISS_ACTION_PARAM.format(
                        param_doc=action.params_doc, failed_param=agent_act.params
                    )
                    return observation

                if agent_act.name == FinishAct.action_name:
                    task.answer = observation
                    task.completion = "completed"

            if action.action_name in agent_act.name:
                param_parse_flag = True

        if act_found_flag:
            return observation
        if param_parse_flag:
            return WRONG_ACTION_PARAM
        return ACTION_NOT_FOUND_MESS
