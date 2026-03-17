import json
import os
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

from .agent_llms import AgentLLM, LLMConfig


class ABCAgent:
    """Abstract base class for agents - placeholder for compatibility."""

    pass


class TaskPackage:
    """Wrapper for tasks passed to agents - equivalent to agentlite TaskPackage."""

    def __init__(self, data: Dict[str, Any] = None, task_id: str = None, **kwargs):
        if data is None:
            data = {}
        self.data = data
        self.task_id = task_id or str(datetime.now().timestamp())

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def __repr__(self):
        return f"TaskPackage(task_id={self.task_id}, data={self.data})"


class AgentAct:
    """Action wrapper - equivalent to agentlite AgentAct."""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    def __repr__(self):
        return f"AgentAct(name={self.name}, params={self.params})"


class ActObsChainType(List):
    """Action-observation chain - equivalent to agentlite ActObsChainType."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, action: AgentAct, observation: Any):
        self.append((action, observation))

    def get_last_observation(self) -> Any:
        if self:
            return self[-1][1]
        return None


class BaseAction:
    """Base class for actions - equivalent to agentlite BaseAction, adapted for LangChain Tools."""

    def __init__(
        self,
        action_name: str,
        action_desc: str,
        params_doc: Dict[str, str] = None,
        llm_provider: str = None,
        tmp: float = 0.4,
    ):
        self.action_name = action_name
        self.action_desc = action_desc
        self.params_doc = params_doc or {}
        self.llm_provider = llm_provider or os.getenv("BACKBONE_LLM")
        self.temperature = tmp
        self.llm = None

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement __call__")

    def to_langchain_tool(self) -> BaseTool:
        """Convert to LangChain Tool."""
        from langchain_core.tools import StructuredTool

        param_str = ", ".join([f"{k}: {v}" for k, v in self.params_doc.items()])

        return StructuredTool.from_function(
            name=self.action_name,
            description=f"{self.action_desc}. Parameters: {param_str}",
            func=self._run_tool,
        )

    def _run_tool(self, **kwargs) -> Any:
        return self(**kwargs)


class BaseAgent:
    """Base agent class - equivalent to agentlite BaseAgent, using LangChain."""

    def __init__(
        self,
        name: str,
        role: str,
        llm: AgentLLM = None,
        actions: List[BaseAction] = None,
        manager: Any = None,
        max_exec_steps: int = 60,
        logger: Any = None,
        reasoning_type: str = "react",
        constraint: str = "",
        instruction: str = "",
    ):
        self.name = name
        self.role = role
        self.constraint = constraint
        self.instruction = instruction
        self.llm = llm or AgentLLM(
            LLMConfig({"temperature": 0.4}), llm_name=os.getenv("BACKBONE_LLM")
        )
        self.actions = actions or []
        self.manager = manager
        self.max_exec_steps = max_exec_steps
        self.logger = logger
        self.reasoning_type = reasoning_type

        self.tools = [action.to_langchain_tool() for action in self.actions]
        self.langchain_llm = self._get_langchain_llm()
        self.llm_layer = self.llm  # Alias for backward compatibility

    def _get_langchain_llm(self):
        """Get the underlying LangChain LLM."""
        return self.llm.get_langchain_llm()

    def run(self, task: TaskPackage) -> Any:
        """
        Run the agent on a task using the agentlite-style execution loop.

        Agents with __next_act__ use a react-style loop:
          1. __next_act__ generates an action via LLM
          2. forward() executes the action
          3. Repeat until Finish action or max steps reached

        Falls back to LangChain's create_agent for agents without __next_act__.
        """
        # Use agentlite-style loop if the agent has __next_act__
        if hasattr(self, '__next_act__'):
            return self._run_agentlite_loop(task)

        # Fallback: LangChain create_agent
        from langchain.agents import create_agent

        task_input = task.get("input", str(task))
        system_prompt = self._create_prompt()

        agent = create_agent(
            model=self.langchain_llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        result = agent.invoke({"messages": [HumanMessage(content=task_input)]})

        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return str(result)

    def _run_agentlite_loop(self, task: TaskPackage) -> Any:
        """Execute the agentlite-style react loop: __next_act__ → forward → repeat."""
        # Initialize inner actions (ThinkAct, FinishAct, etc.) — only once
        if hasattr(self, '__add_inner_actions__') and not getattr(self, '_inner_actions_added', False):
            self.__add_inner_actions__()
            self._inner_actions_added = True

        action_chain = ActObsChainType()
        task.completion = "pending"

        for step in range(self.max_exec_steps):
            try:
                agent_act = self.__next_act__(task, action_chain)
            except Exception as e:
                print(f"[{self.name}] __next_act__ failed at step {step}: {type(e).__name__}: {e}", flush=True)
                break

            try:
                observation = self.forward(task, agent_act)
            except Exception as e:
                print(f"[{self.name}] forward() failed at step {step}: {type(e).__name__}: {e}", flush=True)
                observation = f"Action execution error: {e}"

            action_chain.add(agent_act, observation)

            # Check if task is completed (set by forward when FinishAct is called)
            if getattr(task, 'completion', None) == "completed":
                return getattr(task, 'answer', observation)

        # Max steps reached without completion
        print(f"[{self.name}] Reached max execution steps ({self.max_exec_steps})", flush=True)
        last_obs = action_chain.get_last_observation()
        return last_obs if last_obs is not None else None

    def __call__(self, task_input: Any) -> Any:
        """Allow calling the agent directly like a function."""
        if isinstance(task_input, TaskPackage):
            return self.run(task_input)
        if isinstance(task_input, str):
            task_input = {"input": task_input}
        task = TaskPackage(data=task_input)
        return self.run(task)

    def _create_prompt(self) -> str:
        """Create the agent prompt."""
        action_descriptions = "\n".join(
            [f"- {action.action_name}: {action.action_desc}" for action in self.actions]
        )

        return f"""You are {self.name}.

{self.role}

You have access to the following tools:
{action_descriptions}

Begin!"""

    def forward(self, task: TaskPackage, agent_act: AgentAct = None):
        """Forward pass - execute an action."""
        if agent_act is None:
            return self.run(task)

        action = act_match(agent_act.name, self.actions)
        if action is None:
            return f"Action {agent_act.name} not found"

        # Filter params to only include valid parameters for the action's __call__
        import inspect
        valid_params = agent_act.params
        if hasattr(action, '__call__'):
            sig = inspect.signature(action.__call__)
            accepted_keys = set(sig.parameters.keys()) - {'self'}
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_var_keyword and accepted_keys:
                valid_params = {
                    k: v for k, v in agent_act.params.items()
                    if k in accepted_keys
                }
        return action(**valid_params)


class AgentLogger:
    """Logger for agent execution - equivalent to agentlite AgentLogger."""

    def __init__(self, FLAG_PRINT: bool = True, PROMPT_DEBUG_FLAG: bool = False):
        self.FLAG_PRINT = FLAG_PRINT
        self.PROMPT_DEBUG_FLAG = PROMPT_DEBUG_FLAG
        self.history = []

    def get_prompt(self, prompt: str):
        if self.PROMPT_DEBUG_FLAG:
            print(f"[PROMPT] {prompt}")
        self.history.append({"type": "prompt", "content": prompt})

    def get_llm_output(self, output: str):
        if self.FLAG_PRINT:
            print(f"[LLM OUTPUT] {output}")
        self.history.append({"type": "llm_output", "content": output})

    def get_observation(self, observation: str):
        if self.FLAG_PRINT:
            print(f"[OBSERVATION] {observation}")
        self.history.append({"type": "observation", "content": observation})

    def info(self, message: str):
        if self.FLAG_PRINT:
            print(f"[INFO] {message}")
        self.history.append({"type": "info", "content": message})

    def error(self, message: str):
        if self.FLAG_PRINT:
            print(f"[ERROR] {message}")
        self.history.append({"type": "error", "content": message})


class BaseAgentLogger:
    """Base class for agent loggers."""

    def __init__(self):
        pass

    def log(self, message: str):
        raise NotImplementedError


class UILogger(BaseAgentLogger):
    """UI Logger for Streamlit integration."""

    def __init__(self):
        super().__init__()
        self.logs = []

    def log(self, message: str):
        self.logs.append(message)


class Proposal:
    """Proposal object for research planning."""

    _id_counter = 0

    def __init__(self, user_query: str, proposal: str = None):
        Proposal._id_counter += 1
        self._id = Proposal._id_counter
        self.user_query = user_query
        self.proposal = proposal
        self.status = "Draft"
        self.feedback = []
        self.id_feedback = []

    def get_id(self):
        return self._id

    def get_query(self):
        return self.user_query

    def get_proposal(self):
        return self.proposal

    def get_summary(self):
        return f"Proposal {self._id}: {self.proposal[:100]}..."

    def get_current_mapper_feedback(self):
        if self.id_feedback:
            return self.id_feedback[-1]
        return None

    def retrieve_mapper_feedback_trace(self):
        """Return (previous_feedback, current_feedback) tuple."""
        if len(self.id_feedback) >= 2:
            return self.id_feedback[-2], self.id_feedback[-1]
        elif len(self.id_feedback) == 1:
            return None, self.id_feedback[-1]
        return None, None

    def update_status(self, status: str):
        self.status = status

    def update_id_feedback(self, feedback):
        if isinstance(feedback, list):
            self.id_feedback.extend(feedback)
        else:
            self.id_feedback.append(feedback)

    def add_feedback(self, feedback: str):
        self.feedback.append(feedback)

    def get_status(self):
        return self.status

    def get_summary(self):
        summary = f"Proposal {self._id}: {self.proposal[:100]}..." if self.proposal else f"Proposal {self._id}"
        if self.feedback:
            summary += f"\nFeedback: {self.feedback[-1]}"
        if self.id_feedback:
            summary += f"\nID Mapping Feedback: {self.id_feedback[-1]}"
        return summary

    def log_summary(self):
        if self.status in ("Approved",):
            return f"<Proposal:{self._id}> approved, please do Finish action."
        return f"<Proposal:{self._id}> status: {self.status}"

    def __repr__(self):
        return f"<Proposal:{self._id}>"


ACTION_NOT_FOUND_MESS = "[Error] Action not found in action list."


# ---------------------------------------------------------------------------
# Built-in inner actions (equivalents of AgentLite's ThinkAct / PlanAct)
# These are added automatically by __add_inner_actions__ in each agent module.
# ---------------------------------------------------------------------------

class _ThinkAction(BaseAction):
    """Let the agent emit an intermediate thought without calling an external tool."""

    def __init__(self):
        super().__init__(
            action_name="Think",
            action_desc=(
                "Use this action to reason step-by-step before choosing the next tool. "
                "The thought is recorded but produces no external side-effect."
            ),
            params_doc={"thought": "Your reasoning or intermediate analysis (string)."},
        )

    def __call__(self, thought: str = "", **kwargs) -> str:
        return thought

    def __hash__(self):
        return hash(self.action_name)

    def __eq__(self, other):
        return isinstance(other, _ThinkAction)


class _PlanAction(BaseAction):
    """Let the agent emit a high-level plan before executing individual steps."""

    def __init__(self):
        super().__init__(
            action_name="Plan",
            action_desc=(
                "Use this action to outline a multi-step plan before starting execution. "
                "The plan is recorded but produces no external side-effect."
            ),
            params_doc={"plan": "Your step-by-step plan (string)."},
        )

    def __call__(self, plan: str = "", **kwargs) -> str:
        return plan

    def __hash__(self):
        return hash(self.action_name)

    def __eq__(self, other):
        return isinstance(other, _PlanAction)


# Module-level singletons — imported by research_planning, experiment_analysis,
# and literature_reasoning as `ThinkAct` and `PlanAct`.
ThinkAct = _ThinkAction()
PlanAct = _PlanAction()


def create_agent_executor(
    llm, tools: List[BaseTool], system_message: str, max_iterations: int = 60
):
    """Create a LangChain agent executor with tools."""
    from langchain.agents import create_agent

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
    )


def convert_to_langchain_tools(actions: List[BaseAction]) -> List[BaseTool]:
    """Convert BaseAction list to LangChain tools."""
    return [action.to_langchain_tool() for action in actions]


def act_match(action_name: str, action_list: List[BaseAction]) -> Optional[BaseAction]:
    """Match action name to action in list."""
    for action in action_list:
        if action.action_name == action_name:
            return action
    return None
