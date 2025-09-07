# Context for: agents

## Directory Structure

```
agents/
├── workflow
│   ├── __init__.py
│   ├── chain_agent.py
│   ├── evaluator_optimizer.py
│   ├── orchestrator_agent.py
│   ├── orchestrator_models.py
│   ├── orchestrator_prompts.py
│   ├── parallel_agent.py
│   └── router_agent.py
├── __init__.py
├── agent.py
└── base_agent.py
```
---

## File Contents

--- START OF FILE __init__.py ---

--- END OF FILE __init__.py ---


--- START OF FILE agent.py ---
"""
Agent implementation using the clean BaseAgent adapter.

This provides a streamlined implementation that adheres to AgentProtocol
while delegating LLM operations to an attached AugmentedLLMProtocol instance.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.interactive_prompt import InteractivePrompt
from mcp_agent.human_input.types import HumanInputCallback
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLMProtocol)


class Agent(BaseAgent):
    """
    An Agent is an entity that has access to a set of MCP servers and can interact with them.
    Each agent should have a purpose defined by its instruction.

    This implementation provides a clean adapter that adheres to AgentProtocol
    while delegating LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,  # Can be AgentConfig or backward compatible str name
        functions: Optional[List[Callable]] = None,
        connection_persistence: bool = True,
        human_input_callback: Optional[HumanInputCallback] = None,
        context: Optional["Context"] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        # Initialize with BaseAgent constructor
        super().__init__(
            config=config,
            functions=functions,
            connection_persistence=connection_persistence,
            human_input_callback=human_input_callback,
            context=context,
            **kwargs,
        )

    async def prompt(self, default_prompt: str = "", agent_name: Optional[str] = None) -> str:
        """
        Start an interactive prompt session with this agent.

        Args:
            default: Default message to use when user presses enter
            agent_name: Ignored for single agents, included for API compatibility

        Returns:
            The result of the interactive session
        """
        # Use the agent name as a string - ensure it's not the object itself
        agent_name_str = str(self.name)

        # Create agent_types dictionary with just this agent
        agent_types = {agent_name_str: self.agent_type.value}

        # Create the interactive prompt
        prompt = InteractivePrompt(agent_types=agent_types)

        # Define wrapper for send function
        async def send_wrapper(message, agent_name):
            return await self.send(message)

        # Start the prompt loop with just this agent
        return await prompt.prompt_loop(
            send_func=send_wrapper,
            default_agent=agent_name_str,
            available_agents=[agent_name_str],  # Only this agent
            prompt_provider=self,  # Pass self as the prompt provider since we implement the protocol
            default=default_prompt,
        )
--- END OF FILE agent.py ---


--- START OF FILE base_agent.py ---
"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached AugmentedLLMProtocol instance.
"""

import asyncio
import fnmatch
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ListToolsResult,
    PromptMessage,
    ReadResourceResult,
    TextContent,
    Tool,
)
from opentelemetry import trace
from pydantic import BaseModel

from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import PromptExitError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.human_input.types import (
    HUMAN_INPUT_SIGNAL_NAME,
    HumanInputCallback,
    HumanInputRequest,
    HumanInputResponse,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AgentProtocol, AugmentedLLMProtocol
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLMProtocol)

HUMAN_INPUT_TOOL_NAME = "__human_input__"
if TYPE_CHECKING:
    from mcp_agent.context import Context
    from mcp_agent.llm.usage_tracking import UsageAccumulator


DEFAULT_CAPABILITIES = AgentCapabilities(
    streaming=False, pushNotifications=False, stateTransitionHistory=False
)


class BaseAgent(MCPAggregator, AgentProtocol):
    """
    A base Agent class that implements the AgentProtocol interface.

    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        functions: Optional[List[Callable]] = None,
        connection_persistence: bool = True,
        human_input_callback: Optional[HumanInputCallback] = None,
        context: Optional["Context"] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.config = config

        super().__init__(
            context=context,
            server_names=self.config.servers,
            connection_persistence=connection_persistence,
            name=self.config.name,
            **kwargs,
        )

        self._context = context
        self.tracer = trace.get_tracer(__name__)
        self.name = self.config.name
        self.instruction = self.config.instruction
        self.functions = functions or []
        self.executor = self.context.executor if context and hasattr(context, "executor") else None
        self.logger = get_logger(f"{__name__}.{self.name}")

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # Initialize the LLM to None (will be set by attach_llm)
        self._llm: Optional[AugmentedLLMProtocol] = None

        # Map function names to tools
        self._function_tool_map: Dict[str, Any] = {}

        if not self.config.human_input:
            self.human_input_callback = None
        else:
            self.human_input_callback: Optional[HumanInputCallback] = human_input_callback
            if not human_input_callback and context and hasattr(context, "human_input_handler"):
                self.human_input_callback = context.human_input_handler

    async def initialize(self) -> None:
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await self.__aenter__()  # This initializes the connection manager and loads the servers

    async def attach_llm(
        self,
        llm_factory: Union[Type[AugmentedLLMProtocol], Callable[..., AugmentedLLMProtocol]],
        model: Optional[str] = None,
        request_params: Optional[RequestParams] = None,
        **additional_kwargs,
    ) -> AugmentedLLMProtocol:
        """
        Create and attach an LLM instance to this agent.

        Parameters have the following precedence (highest to lowest):
        1. Explicitly passed parameters to this method
        2. Agent's default_request_params
        3. LLM's default values

        Args:
            llm_factory: A class or callable that constructs an AugmentedLLM
            model: Optional model name override
            request_params: Optional request parameters override
            **additional_kwargs: Additional parameters passed to the LLM constructor

        Returns:
            The created LLM instance
        """
        # Start with agent's default params
        effective_params = (
            self._default_request_params.model_copy() if self._default_request_params else None
        )

        # Override with explicitly passed request_params
        if request_params:
            if effective_params:
                # Update non-None values
                for k, v in request_params.model_dump(exclude_unset=True).items():
                    if v is not None:
                        setattr(effective_params, k, v)
            else:
                effective_params = request_params

        # Override model if explicitly specified
        if model and effective_params:
            effective_params.model = model

        # Create the LLM instance
        self._llm = llm_factory(
            agent=self, request_params=effective_params, context=self._context, **additional_kwargs
        )

        return self._llm

    async def shutdown(self) -> None:
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await super().close()

    async def __call__(
        self,
        message: Union[str, PromptMessageMultipart] | None = None,
        agent_name: Optional[str] = None,
        default_prompt: str = "",
    ) -> str:
        """
        Make the agent callable to send messages or start an interactive prompt.

        Args:
            message: Optional message to send to the agent
            agent_name: Optional name of the agent (for consistency with DirectAgentApp)
            default: Default message to use in interactive prompt mode

        Returns:
            The agent's response as a string or the result of the interactive session
        """
        if message:
            return await self.send(message)
        return await self.prompt(default_prompt=default_prompt)

    async def generate_str(self, message: str, request_params: RequestParams | None) -> str:
        result: PromptMessageMultipart = await self.generate([Prompt.user(message)], request_params)
        return result.first_text()

    async def send(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: Message content in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly

        Returns:
            The agent's response as a string
        """
        # Convert the input to a PromptMessageMultipart
        prompt = self._normalize_message_input(message)

        # Use the LLM to generate a response
        response = await self.generate([prompt], None)
        return response.all_text()

    def _normalize_message_input(
        self, message: Union[str, PromptMessage, PromptMessageMultipart]
    ) -> PromptMessageMultipart:
        """
        Convert a message of any supported type to PromptMessageMultipart.

        Args:
            message: Message in various formats (string, PromptMessage, or PromptMessageMultipart)

        Returns:
            A PromptMessageMultipart object
        """
        # Handle single message
        if isinstance(message, str):
            return Prompt.user(message)
        elif isinstance(message, PromptMessage):
            return PromptMessageMultipart(role=message.role, content=[message.content])
        elif isinstance(message, PromptMessageMultipart):
            return message
        else:
            # Try to convert to string as fallback
            return Prompt.user(str(message))

    async def prompt(self, default_prompt: str = "") -> str:
        """
        Start an interactive prompt session with the agent.

        Args:
            default_prompt: The initial prompt to send to the agent

        Returns:
            The result of the interactive session
        """
        ...

    async def request_human_input(self, request: HumanInputRequest) -> str:
        """
        Request input from a human user. Pauses the workflow until input is received.

        Args:
            request: The human input request

        Returns:
            The input provided by the human

        Raises:
            TimeoutError: If the timeout is exceeded
        """
        if not self.human_input_callback:
            raise ValueError("Human input callback not set")

        # Generate a unique ID for this request to avoid signal collisions
        request_id = f"{HUMAN_INPUT_SIGNAL_NAME}_{self.name}_{uuid.uuid4()}"
        request.request_id = request_id
        # Use metadata as a dictionary to pass agent name
        request.metadata = {"agent_name": self.name}
        self.logger.debug("Requesting human input:", data=request)

        if not self.executor:
            raise ValueError("No executor available")

        async def call_callback_and_signal() -> None:
            try:
                assert self.human_input_callback is not None
                user_input = await self.human_input_callback(request)

                self.logger.debug("Received human input:", data=user_input)
                await self.executor.signal(signal_name=request_id, payload=user_input)
            except PromptExitError as e:
                # Propagate the exit error through the signal system
                self.logger.info("User requested to exit session")
                await self.executor.signal(
                    signal_name=request_id,
                    payload={"exit_requested": True, "error": str(e)},
                )
            except Exception as e:
                await self.executor.signal(
                    request_id, payload=f"Error getting human input: {str(e)}"
                )

        asyncio.create_task(call_callback_and_signal())

        self.logger.debug("Waiting for human input signal")

        # Wait for signal (workflow is paused here)
        result = await self.executor.wait_for_signal(
            signal_name=request_id,
            request_id=request_id,
            workflow_id=request.workflow_id,
            signal_description=request.description or request.prompt,
            timeout_seconds=request.timeout_seconds,
            signal_type=HumanInputResponse,
        )

        if isinstance(result, dict) and result.get("exit_requested", False):
            raise PromptExitError(result.get("error", "User requested to exit FastAgent session"))
        self.logger.debug("Received human input signal", data=result)
        return result

    def _matches_pattern(self, name: str, pattern: str, server_name: str) -> bool:
        """
        Check if a name matches a pattern for a specific server.

        Args:
            name: The name to match (could be tool name, resource URI, or prompt name)
            pattern: The pattern to match against (e.g., "add", "math*", "resource://math/*")
            server_name: The server name (used for tool name prefixing)

        Returns:
            True if the name matches the pattern
        """
        # For tools, build the full pattern with server prefix: server_name-pattern
        if name.startswith(f"{server_name}-"):
            full_pattern = f"{server_name}-{pattern}"
            return fnmatch.fnmatch(name, full_pattern)

        # For resources and prompts, match directly against the pattern
        return fnmatch.fnmatch(name, pattern)

    async def list_tools(self) -> ListToolsResult:
        """
        List all tools available to this agent, filtered by configuration.

        Returns:
            ListToolsResult with available tools
        """
        if not self.initialized:
            await self.initialize()

        # Get all tools from the parent class
        result = await super().list_tools()

        # Apply filtering if tools are specified in config
        if self.config.tools is not None:
            filtered_tools = []
            for tool in result.tools:
                # Extract server name from tool name (e.g., "mathematics-add" -> "mathematics")
                if "-" in tool.name:
                    server_name = tool.name.split("-", 1)[0]

                    # Check if this server has tool filters
                    if server_name in self.config.tools:
                        # Check if tool matches any pattern for this server
                        for pattern in self.config.tools[server_name]:
                            if self._matches_pattern(tool.name, pattern, server_name):
                                filtered_tools.append(tool)
                                break
            result.tools = filtered_tools

        if not self.human_input_callback:
            return result

        # Add a human_input_callback as a tool
        from mcp.server.fastmcp.tools import Tool as FastTool

        human_input_tool: FastTool = FastTool.from_function(self.request_human_input)
        result.tools.append(
            Tool(
                name=HUMAN_INPUT_TOOL_NAME,
                description=human_input_tool.description,
                inputSchema=human_input_tool.parameters,
            )
        )

        return result

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> CallToolResult:
        """
        Call a tool by name with the given arguments.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Result of the tool call
        """
        if name == HUMAN_INPUT_TOOL_NAME:
            # Call the human input tool
            return await self._call_human_input_tool(arguments)
        else:
            return await super().call_tool(name, arguments)

    async def _call_human_input_tool(
        self, arguments: Dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Handle human input request via tool calling.

        Args:
            arguments: Tool arguments

        Returns:
            Result of the human input request
        """
        # Handle human input request
        try:
            # Make sure arguments is not None
            if arguments is None:
                arguments = {}

            # Extract request data
            request_data = arguments.get("request")

            # Handle both string and dict request formats
            if isinstance(request_data, str):
                request = HumanInputRequest(prompt=request_data)
            elif isinstance(request_data, dict):
                request = HumanInputRequest(**request_data)
            else:
                # Fallback for invalid or missing request data
                request = HumanInputRequest(prompt="Please provide input:")

            result = await self.request_human_input(request=request)

            # Use response attribute if available, otherwise use the result directly
            response_text = (
                result.response if isinstance(result, HumanInputResponse) else str(result)
            )

            return CallToolResult(
                content=[TextContent(type="text", text=f"Human response: {response_text}")]
            )

        except PromptExitError:
            raise
        except asyncio.TimeoutError as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: Human input request timed out: {str(e)}",
                    )
                ],
            )
        except Exception as e:
            import traceback

            print(f"Error in _call_human_input_tool: {traceback.format_exc()}")

            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Error requesting human input: {str(e)}")],
            )

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult:
        """
        Get a prompt from a server.

        Args:
            prompt_name: Name of the prompt, optionally namespaced
            arguments: Optional dictionary of arguments to pass to the prompt template
            server_name: Optional name of the server to get the prompt from

        Returns:
            GetPromptResult containing the prompt information
        """
        return await super().get_prompt(prompt_name, arguments, server_name)

    async def apply_prompt(
        self,
        prompt: Union[str, GetPromptResult],
        arguments: Dict[str, str] | None = None,
        agent_name: str | None = None,
        server_name: str | None = None,
        as_template: bool = False,
    ) -> str:
        """
        Apply an MCP Server Prompt by name or GetPromptResult and return the assistant's response.
        Will search all available servers for the prompt if not namespaced and no server_name provided.

        If the last message in the prompt is from a user, this will automatically
        generate an assistant response to ensure we always end with an assistant message.

        Args:
            prompt: The name of the prompt to apply OR a GetPromptResult object
            arguments: Optional dictionary of string arguments to pass to the prompt template
            agent_name: Optional agent name (ignored at this level, used by multi-agent apps)
            server_name: Optional name of the server to get the prompt from
            as_template: If True, store as persistent template (always included in context)

        Returns:
            The assistant's response or error message
        """

        # Handle both string and GetPromptResult inputs
        if isinstance(prompt, str):
            prompt_name = prompt
            # Get the prompt - this will search all servers if needed
            self.logger.debug(f"Loading prompt '{prompt_name}'")
            prompt_result: GetPromptResult = await self.get_prompt(
                prompt_name, arguments, server_name
            )

            if not prompt_result or not prompt_result.messages:
                error_msg = f"Prompt '{prompt_name}' could not be found or contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Get the display name (namespaced version)
            namespaced_name = getattr(prompt_result, "namespaced_name", prompt_name)
        else:
            # prompt is a GetPromptResult object
            prompt_result = prompt
            if not prompt_result or not prompt_result.messages:
                error_msg = "Provided GetPromptResult contains no messages"
                self.logger.warning(error_msg)
                return error_msg

            # Use a reasonable display name
            namespaced_name = getattr(prompt_result, "namespaced_name", "provided_prompt")

        self.logger.debug(f"Using prompt '{namespaced_name}'")

        # Convert prompt messages to multipart format using the safer method
        multipart_messages = PromptMessageMultipart.from_get_prompt_result(prompt_result)

        if as_template:
            # Use apply_prompt_template to store as persistent prompt messages
            return await self.apply_prompt_template(prompt_result, namespaced_name)
        else:
            # Always call generate to ensure LLM implementations can handle prompt templates
            # This is critical for stateful LLMs like PlaybackLLM
            response = await self.generate(multipart_messages, None)
            return response.first_text()

    async def get_embedded_resources(
        self, resource_uri: str, server_name: str | None = None
    ) -> List[EmbeddedResource]:
        """
        Get a resource from an MCP server and return it as a list of embedded resources ready for use in prompts.

        Args:
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            List of EmbeddedResource objects ready to use in a PromptMessageMultipart

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        # Get the raw resource result
        result: ReadResourceResult = await self.get_resource(resource_uri, server_name)

        # Convert each resource content to an EmbeddedResource
        embedded_resources: List[EmbeddedResource] = []
        for resource_content in result.contents:
            embedded_resource = EmbeddedResource(
                type="resource", resource=resource_content, annotations=None
            )
            embedded_resources.append(embedded_resource)

        return embedded_resources

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageMultipart],
        resource_uri: str,
        server_name: str | None = None,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Content in various formats:
                - String: Converted to a user message with the text
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
            resource_uri: URI of the resource to retrieve
            server_name: Optional name of the MCP server to retrieve the resource from

        Returns:
            The agent's response as a string
        """
        # Get the embedded resources
        embedded_resources: List[EmbeddedResource] = await self.get_embedded_resources(
            resource_uri, server_name
        )

        # Create or update the prompt message
        prompt: PromptMessageMultipart
        if isinstance(prompt_content, str):
            # Create a new prompt with the text and resources
            content = [TextContent(type="text", text=prompt_content)]
            content.extend(embedded_resources)
            prompt = PromptMessageMultipart(role="user", content=content)
        elif isinstance(prompt_content, PromptMessage):
            # Convert PromptMessage to PromptMessageMultipart and add resources
            content = [prompt_content.content]
            content.extend(embedded_resources)
            prompt = PromptMessageMultipart(role=prompt_content.role, content=content)
        elif isinstance(prompt_content, PromptMessageMultipart):
            # Add resources to the existing prompt
            prompt = prompt_content
            prompt.content.extend(embedded_resources)
        else:
            raise TypeError(
                "prompt_content must be a string, PromptMessage, or PromptMessageMultipart"
            )

        response: PromptMessageMultipart = await self.generate([prompt], None)
        return response.first_text()

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Create a completion with the LLM using the provided messages.
        Delegates to the attached LLM.

        Args:
            multipart_messages: List of multipart messages to send to the LLM
            request_params: Optional parameters to configure the request

        Returns:
            The LLM's response as a PromptMessageMultipart
        """
        assert self._llm
        with self.tracer.start_as_current_span(f"Agent: '{self.name}' generate"):
            return await self._llm.generate(multipart_messages, request_params)

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template as persistent context that will be included in all future conversations.
        Delegates to the attached LLM.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated
        """
        assert self._llm
        with self.tracer.start_as_current_span(f"Agent: '{self.name}' apply_prompt_template"):
            return await self._llm.apply_prompt_template(prompt_result, prompt_name)

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Apply the prompt and return the result as a Pydantic model.
        Delegates to the attached LLM.

        Args:
            prompt: List of PromptMessageMultipart objects
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            An instance of the specified model, or None if coercion fails
        """
        assert self._llm
        with self.tracer.start_as_current_span(f"Agent: '{self.name}' structured"):
            return await self._llm.structured(multipart_messages, model, request_params)

    async def apply_prompt_messages(
        self, prompts: List[PromptMessageMultipart], request_params: RequestParams | None = None
    ) -> str:
        """
        Apply a list of prompt messages and return the result.

        Args:
            prompts: List of PromptMessageMultipart messages
            request_params: Optional request parameters

        Returns:
            The text response from the LLM
        """

        response = await self.generate(prompts, request_params)
        return response.first_text()

    async def list_prompts(self, server_name: str | None = None) -> Mapping[str, List[Prompt]]:
        """
        List all prompts available to this agent, filtered by configuration.

        Args:
            server_name: Optional server name to list prompts from

        Returns:
            Dictionary mapping server names to lists of Prompt objects
        """
        if not self.initialized:
            await self.initialize()

        # Get all prompts from the parent class
        result = await super().list_prompts(server_name)

        # Apply filtering if prompts are specified in config
        if self.config.prompts is not None:
            filtered_result = {}
            for server, prompts in result.items():
                # Check if this server has prompt filters
                if server in self.config.prompts:
                    filtered_prompts = []
                    for prompt in prompts:
                        # Check if prompt matches any pattern for this server
                        for pattern in self.config.prompts[server]:
                            if self._matches_pattern(prompt.name, pattern, server):
                                filtered_prompts.append(prompt)
                                break
                    if filtered_prompts:
                        filtered_result[server] = filtered_prompts
            result = filtered_result

        return result

    async def list_resources(self, server_name: str | None = None) -> Dict[str, List[str]]:
        """
        List all resources available to this agent, filtered by configuration.

        Args:
            server_name: Optional server name to list resources from

        Returns:
            Dictionary mapping server names to lists of resource URIs
        """
        if not self.initialized:
            await self.initialize()

        # Get all resources from the parent class
        result = await super().list_resources(server_name)

        # Apply filtering if resources are specified in config
        if self.config.resources is not None:
            filtered_result = {}
            for server, resources in result.items():
                # Check if this server has resource filters
                if server in self.config.resources:
                    filtered_resources = []
                    for resource in resources:
                        # Check if resource matches any pattern for this server
                        for pattern in self.config.resources[server]:
                            if self._matches_pattern(resource, pattern, server):
                                filtered_resources.append(resource)
                                break
                    if filtered_resources:
                        filtered_result[server] = filtered_resources
            result = filtered_result

        return result

    @property
    def agent_type(self) -> AgentType:
        """
        Return the type of this agent.
        """
        return AgentType.BASIC

    async def agent_card(self) -> AgentCard:
        """
        Return an A2A card describing this Agent
        """

        skills: List[AgentSkill] = []
        tools: ListToolsResult = await self.list_tools()
        for tool in tools.tools:
            skills.append(await self.convert(tool))

        return AgentCard(
            name=self.name,
            description=self.instruction,
            url=f"fast-agent://agents/{self.name}/",
            version="0.1",
            capabilities=DEFAULT_CAPABILITIES,
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            provider=None,
            documentationUrl=None,
            authentication=None,
            skills=skills,
        )

    async def convert(self, tool: Tool) -> AgentSkill:
        """
        Convert a Tool to an AgentSkill.
        """

        _, tool_without_namespace = await self._parse_resource_name(tool.name, "tool")
        return AgentSkill(
            id=tool.name,
            name=tool_without_namespace,
            description=tool.description,
            tags=["tool"],
            examples=None,
            inputModes=None,  # ["text/plain"],
            # cover TextContent | ImageContent ->
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/223
            # https://github.com/modelcontextprotocol/modelcontextprotocol/pull/93
            outputModes=None,  # ,["text/plain", "image/*"],
        )

    @property
    def message_history(self) -> List[PromptMessageMultipart]:
        """
        Return the agent's message history as PromptMessageMultipart objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageMultipart objects representing the conversation history
        """
        if self._llm:
            return self._llm.message_history
        return []

    @property
    def usage_accumulator(self) -> Optional["UsageAccumulator"]:
        """
        Return the usage accumulator for tracking token usage across turns.

        Returns:
            UsageAccumulator object if LLM is attached, None otherwise
        """
        if self._llm:
            return self._llm.usage_accumulator
        return None
--- END OF FILE base_agent.py ---


--- START OF FILE workflow/__init__.py ---
# Workflow agents module
--- END OF FILE workflow/__init__.py ---


--- START OF FILE workflow/chain_agent.py ---
"""
Chain workflow implementation using the clean BaseAgent adapter pattern.

This provides an implementation that delegates operations to a sequence of
other agents, chaining their outputs together.
"""

from typing import Any, List, Optional, Tuple, Type

from mcp.types import TextContent

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class ChainAgent(BaseAgent):
    """
    A chain agent that processes requests through a series of specialized agents in sequence.
    Passes the output of each agent to the next agent in the chain.
    """

    # TODO -- consider adding "repeat" mode
    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.CHAIN

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        cumulative: bool = False,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a ChainAgent.

        Args:
            config: Agent configuration or name
            agents: List of agents to chain together in sequence
            cumulative: Whether each agent sees all previous responses
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)
        self.agents = agents
        self.cumulative = cumulative

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Chain the request through multiple agents in sequence.

        Args:
            multipart_messages: Initial messages to send to the first agent
            request_params: Optional request parameters

        Returns:
            The response from the final agent in the chain
        """

        # # Get the original user message (last message in the list)
        user_message = multipart_messages[-1] if multipart_messages else None

        if not self.cumulative:
            response: PromptMessageMultipart = await self.agents[0].generate(multipart_messages)
            # Process the rest of the agents in the chain
            for agent in self.agents[1:]:
                next_message = Prompt.user(*response.content)
                response = await agent.generate([next_message])

            return response

        # Track all responses in the chain
        all_responses: List[PromptMessageMultipart] = []

        # Initialize list for storing formatted results
        final_results: List[str] = []

        # Add the original request with XML tag
        request_text = f"<fastagent:request>{user_message.all_text()}</fastagent:request>"
        final_results.append(request_text)

        # Process through each agent in sequence
        for i, agent in enumerate(self.agents):
            # In cumulative mode, include the original message and all previous responses
            chain_messages = multipart_messages.copy()
            chain_messages.extend(all_responses)
            current_response = await agent.generate(chain_messages, request_params)

            # Store the response
            all_responses.append(current_response)

            response_text = current_response.all_text()
            attributed_response = (
                f"<fastagent:response agent='{agent.name}'>{response_text}</fastagent:response>"
            )
            final_results.append(attributed_response)

            if i < len(self.agents) - 1:
                [Prompt.user(current_response.all_text())]

        # For cumulative mode, return the properly formatted output with XML tags
        response_text = "\n\n".join(final_results)
        return PromptMessageMultipart(
            role="assistant",
            content=[TextContent(type="text", text=response_text)],
        )

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Chain the request through multiple agents and parse the final response.

        Args:
            prompt: List of messages to send through the chain
            model: Pydantic model to parse the final response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the final agent, or None if parsing fails
        """
        # Generate response through the chain
        response = await self.generate(prompt, request_params)
        last_agent = self.agents[-1]
        try:
            return await last_agent.structured([response], model, request_params)
        except Exception as e:
            self.logger.warning(f"Failed to parse response from chain: {str(e)}")
            return None, Prompt.assistant("Failed to parse response from chain: {str(e)}")

    async def initialize(self) -> None:
        """
        Initialize the chain agent and all agents in the chain.
        """
        await super().initialize()

        # Initialize all agents in the chain if not already initialized
        for agent in self.agents:
            if not getattr(agent, "initialized", False):
                await agent.initialize()

    async def shutdown(self) -> None:
        """
        Shutdown the chain agent and all agents in the chain.
        """
        await super().shutdown()

        # Shutdown all agents in the chain
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down agent in chain: {str(e)}")
--- END OF FILE workflow/chain_agent.py ---


--- START OF FILE workflow/evaluator_optimizer.py ---
"""
Evaluator-Optimizer workflow implementation using the BaseAgent adapter pattern.

This workflow provides a mechanism for iterative refinement of responses through
evaluation and feedback cycles. It uses one agent to generate responses and another
to evaluate and provide feedback, continuing until a quality threshold is reached
or a maximum number of refinements is attempted.
"""

from enum import Enum
from typing import Any, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


class QualityRating(str, Enum):
    """Enum for evaluation quality ratings."""

    POOR = "POOR"  # Major improvements needed
    FAIR = "FAIR"  # Several improvements needed
    GOOD = "GOOD"  # Minor improvements possible
    EXCELLENT = "EXCELLENT"  # No improvements needed

    # Map string values to integer values for comparisons
    @property
    def value(self) -> int:
        """Convert string enum values to integers for comparison."""
        return {
            "POOR": 0,
            "FAIR": 1,
            "GOOD": 2,
            "EXCELLENT": 3,
        }[self._value_]


class EvaluationResult(BaseModel):
    """Model representing the evaluation result from the evaluator agent."""

    rating: QualityRating = Field(description="Quality rating of the response")
    feedback: str = Field(description="Specific feedback and suggestions for improvement")
    needs_improvement: bool = Field(description="Whether the output needs further improvement")
    focus_areas: List[str] = Field(
        default_factory=list, description="Specific areas to focus on in next iteration"
    )


class EvaluatorOptimizerAgent(BaseAgent):
    """
    An agent that implements the evaluator-optimizer workflow pattern.

    Uses one agent to generate responses and another to evaluate and provide feedback
    for refinement, continuing until a quality threshold is reached or a maximum
    number of refinement cycles is completed.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.EVALUATOR_OPTIMIZER

    def __init__(
        self,
        config: Agent,
        generator_agent: Agent,
        evaluator_agent: Agent,
        min_rating: QualityRating = QualityRating.GOOD,
        max_refinements: int = 3,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the evaluator-optimizer agent.

        Args:
            config: Agent configuration or name
            generator_agent: Agent that generates the initial and refined responses
            evaluator_agent: Agent that evaluates responses and provides feedback
            min_rating: Minimum acceptable quality rating to stop refinement
            max_refinements: Maximum number of refinement cycles to attempt
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)

        if not generator_agent:
            raise AgentConfigError("Generator agent must be provided")

        if not evaluator_agent:
            raise AgentConfigError("Evaluator agent must be provided")

        self.generator_agent = generator_agent
        self.evaluator_agent = evaluator_agent
        self.min_rating = min_rating
        self.max_refinements = max_refinements
        self.refinement_history = []

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Generate a response through evaluation-guided refinement.

        Args:
            multipart_messages: Messages to process
            request_params: Optional request parameters

        Returns:
            The optimized response after evaluation and refinement
        """
        # Initialize tracking variables
        refinement_count = 0
        best_response = None
        best_rating = QualityRating.POOR
        self.refinement_history = []

        # Extract the user request
        request = multipart_messages[-1].all_text() if multipart_messages else ""

        # Initial generation
        response = await self.generator_agent.generate(multipart_messages, request_params)
        best_response = response

        # Refinement loop
        while refinement_count < self.max_refinements:
            logger.debug(f"Evaluating response (iteration {refinement_count + 1})")

            # Evaluate current response
            eval_prompt = self._build_eval_prompt(
                request=request, response=response.all_text(), iteration=refinement_count
            )

            # Create evaluation message and get structured evaluation result
            eval_message = Prompt.user(eval_prompt)
            evaluation_result, _ = await self.evaluator_agent.structured(
                [eval_message], EvaluationResult, request_params
            )

            # If structured parsing failed, use default evaluation
            if evaluation_result is None:
                logger.warning("Structured parsing failed, using default evaluation")
                evaluation_result = EvaluationResult(
                    rating=QualityRating.POOR,
                    feedback="Failed to parse evaluation",
                    needs_improvement=True,
                    focus_areas=["Improve overall quality"],
                )

            # Track iteration
            self.refinement_history.append(
                {
                    "attempt": refinement_count + 1,
                    "response": response.all_text(),
                    "evaluation": evaluation_result.model_dump(),
                }
            )

            logger.debug(f"Evaluation result: {evaluation_result.rating}")

            # Track best response based on rating
            if evaluation_result.rating.value > best_rating.value:
                best_rating = evaluation_result.rating
                best_response = response
                logger.debug(f"New best response (rating: {best_rating})")

            # Check if we've reached acceptable quality
            if not evaluation_result.needs_improvement:
                logger.debug("Improvement not needed, stopping refinement")
                # When evaluator says no improvement needed, use the current response
                best_response = response
                break

            if evaluation_result.rating.value >= self.min_rating.value:
                logger.debug(f"Acceptable quality reached ({evaluation_result.rating})")
                break

            # Generate refined response
            refinement_prompt = self._build_refinement_prompt(
                request=request,
                response=response.all_text(),
                feedback=evaluation_result,
                iteration=refinement_count,
            )

            # Create refinement message and get refined response
            refinement_message = Prompt.user(refinement_prompt)
            response = await self.generator_agent.generate([refinement_message], request_params)

            refinement_count += 1

        return best_response

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Generate an optimized response and parse it into a structured format.

        Args:
            prompt: List of messages to process
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response, or None if parsing fails
        """
        # Generate optimized response
        response = await self.generate(prompt, request_params)

        # Delegate structured parsing to the generator agent
        structured_prompt = Prompt.user(response.all_text())
        return await self.generator_agent.structured([structured_prompt], model, request_params)

    async def initialize(self) -> None:
        """Initialize the agent and its generator and evaluator agents."""
        await super().initialize()

        # Initialize generator and evaluator agents if not already initialized
        if not getattr(self.generator_agent, "initialized", False):
            await self.generator_agent.initialize()

        if not getattr(self.evaluator_agent, "initialized", False):
            await self.evaluator_agent.initialize()

        self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the agent and its generator and evaluator agents."""
        await super().shutdown()

        # Shutdown generator and evaluator agents
        try:
            await self.generator_agent.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down generator agent: {str(e)}")

        try:
            await self.evaluator_agent.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down evaluator agent: {str(e)}")

    def _build_eval_prompt(self, request: str, response: str, iteration: int) -> str:
        """
        Build the evaluation prompt for the evaluator agent.

        Args:
            request: The original user request
            response: The current response to evaluate
            iteration: The current iteration number

        Returns:
            Formatted evaluation prompt
        """
        return f"""
You are an expert evaluator for content quality. Your task is to evaluate a response against the user's original request.

Evaluate the response for iteration {iteration + 1} and provide structured feedback on its quality and areas for improvement.

<fastagent:data>
<fastagent:request>
{request}
</fastagent:request>

<fastagent:response>
{response}
</fastagent:response>
</fastagent:data>

<fastagent:instruction>
Your response MUST be valid JSON matching this exact format (no other text, markdown, or explanation):

{{
  "rating": "RATING",
  "feedback": "DETAILED FEEDBACK",
  "needs_improvement": BOOLEAN,
  "focus_areas": ["FOCUS_AREA_1", "FOCUS_AREA_2", "FOCUS_AREA_3"]
}}

Where:
- RATING: Must be one of: "EXCELLENT", "GOOD", "FAIR", or "POOR"
  - EXCELLENT: No improvements needed
  - GOOD: Only minor improvements possible
  - FAIR: Several improvements needed
  - POOR: Major improvements needed
- DETAILED FEEDBACK: Specific, actionable feedback (as a single string)
- BOOLEAN: true or false (lowercase, no quotes) indicating if further improvement is needed
- FOCUS_AREAS: Array of 1-3 specific areas to focus on (empty array if no improvement needed)

Example of valid response (DO NOT include the triple backticks in your response):
{{
  "rating": "GOOD",
  "feedback": "The response is clear but could use more supporting evidence.",
  "needs_improvement": true,
  "focus_areas": ["Add more examples", "Include data points"]
}}

IMPORTANT: Your response should be ONLY the JSON object without any code fences, explanations, or other text.
</fastagent:instruction>
"""

    def _build_refinement_prompt(
        self,
        request: str,
        response: str,
        feedback: EvaluationResult,
        iteration: int,
    ) -> str:
        """
        Build the refinement prompt for the generator agent.

        Args:
            request: The original user request
            response: The current response to refine
            feedback: The evaluation feedback
            iteration: The current iteration number

        Returns:
            Formatted refinement prompt
        """
        focus_areas = ", ".join(feedback.focus_areas) if feedback.focus_areas else "None specified"

        return f"""
You are tasked with improving a response based on expert feedback. This is iteration {iteration + 1} of the refinement process.

Your goal is to address all feedback points while maintaining accuracy and relevance to the original request.

<fastagent:data>
<fastagent:request>
{request}
</fastagent:request>

<fastagent:previous-response>
{response}
</fastagent:previous-response>

<fastagent:feedback>
<rating>{feedback.rating}</rating>
<details>{feedback.feedback}</details>
<focus-areas>{focus_areas}</focus-areas>
</fastagent:feedback>
</fastagent:data>

<fastagent:instruction>
Create an improved version of the response that:
1. Directly addresses each point in the feedback
2. Focuses on the specific areas mentioned for improvement
3. Maintains all the strengths of the original response
4. Remains accurate and relevant to the original request

Provide your complete improved response without explanations or commentary.
</fastagent:instruction>
"""
--- END OF FILE workflow/evaluator_optimizer.py ---


--- START OF FILE workflow/orchestrator_agent.py ---
"""
OrchestratorAgent implementation using the BaseAgent adapter pattern.

This workflow provides an implementation that manages complex tasks by
dynamically planning, delegating to specialized agents, and synthesizing results.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from mcp.types import TextContent

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.agents.workflow.orchestrator_models import (
    NextStep,
    Plan,
    PlanResult,
    Step,
    TaskWithResult,
    format_plan_result,
    format_step_result_text,
)
from mcp_agent.agents.workflow.orchestrator_prompts import (
    FULL_PLAN_PROMPT_TEMPLATE,
    ITERATIVE_PLAN_PROMPT_TEMPLATE,
    SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE,
    SYNTHESIZE_PLAN_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
)
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    An agent that implements the orchestrator workflow pattern.

    Dynamically creates execution plans and delegates tasks
    to specialized worker agents, synthesizing their results into a cohesive output.
    Supports both full planning and iterative planning modes.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.ORCHESTRATOR

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        plan_type: Literal["full", "iterative"] = "full",
        plan_iterations: int = 5,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an OrchestratorAgent.

        Args:
            config: Agent configuration or name
            agents: List of specialized worker agents available for task execution
            plan_type: Planning mode ("full" or "iterative")
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)

        if not agents:
            raise AgentConfigError("At least one worker agent must be provided")

        self.plan_type = plan_type

        # Store agents by name for easier lookup
        self.agents: Dict[str, Agent] = {}
        for agent in agents:
            agent_name = agent.name
            self.logger.info(f"Adding agent '{agent_name}' to orchestrator")
            self.agents[agent_name] = agent
        self.plan_iterations = plan_iterations
        # For tracking state during execution
        self.plan_result: Optional[PlanResult] = None

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Execute an orchestrated plan to process the input.

        Args:
            multipart_messages: Messages to process
            request_params: Optional request parameters

        Returns:
            The final synthesized response from the orchestration
        """
        # Extract user request
        objective = multipart_messages[-1].all_text() if multipart_messages else ""

        # Initialize execution parameters
        params = self._merge_request_params(request_params)

        # Execute the plan
        plan_result = await self._execute_plan(objective, params)
        self.plan_result = plan_result

        # Return the result
        return PromptMessageMultipart(
            role="assistant",
            content=[TextContent(type="text", text=plan_result.result or "No result available")],
        )

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Execute an orchestration plan and parse the result into a structured format.

        Args:
            prompt: List of messages to process
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed final response, or None if parsing fails
        """
        # Generate orchestration result
        response = await self.generate(prompt, request_params)

        # Try to parse the response into the specified model
        try:
            result_text = response.all_text()
            prompt_message = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=result_text)]
            )
            assert self._llm
            return await self._llm.structured([prompt_message], model, request_params)
        except Exception as e:
            self.logger.warning(f"Failed to parse orchestration result: {str(e)}")
            return None, Prompt.assistant(f"Failed to parse orchestration result: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the orchestrator agent and worker agents."""
        await super().initialize()

        # Initialize all worker agents if not already initialized
        for agent_name, agent in self.agents.items():
            if not getattr(agent, "initialized", False):
                self.logger.debug(f"Initializing agent: {agent_name}")
                await agent.initialize()

        self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the orchestrator agent and worker agents."""
        await super().shutdown()

        # Shutdown all worker agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down agent {agent_name}: {str(e)}")

    async def _execute_plan(self, objective: str, request_params: RequestParams) -> PlanResult:
        """
        Execute a plan to achieve the given objective.

        Args:
            objective: The objective to achieve
            request_params: Request parameters for execution

        Returns:
            PlanResult containing execution results and final output
        """
        iterations = 0
        total_steps_executed = 0
        max_iterations = self.plan_iterations
        max_steps = getattr(request_params, "max_steps", max_iterations * 3)

        # Initialize plan result
        plan_result = PlanResult(objective=objective, step_results=[])
        plan_result.max_iterations_reached = False

        while iterations < max_iterations:
            # Generate plan based on planning mode
            if self.plan_type == "iterative":
                next_step = await self._get_next_step(objective, plan_result, request_params)
                if next_step is None:
                    self.logger.error("Failed to generate next step, ending iteration early")
                    plan_result.max_iterations_reached = True
                    break

                logger.debug(f"Iteration {iterations}: Iterative plan:", data=next_step)
                plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
            elif self.plan_type == "full":
                plan = await self._get_full_plan(objective, plan_result, request_params)
                if plan is None:
                    self.logger.error("Failed to generate full plan, ending iteration early")
                    plan_result.max_iterations_reached = True
                    break

                logger.debug(f"Iteration {iterations}: Full Plan:", data=plan)
            else:
                raise ValueError(f"Invalid plan type: {self.plan_type}")

            # Validate agent names early
            self._validate_agent_names(plan)

            # Store plan in result
            plan_result.plan = plan

            # Execute the steps in the plan
            for step in plan.steps:
                # Check if we've hit the step limit
                if total_steps_executed >= max_steps:
                    self.logger.warning(
                        f"Reached maximum step limit ({max_steps}) without completing objective"
                    )
                    plan_result.max_iterations_reached = True
                    break

                # Execute the step and collect results
                step_result = await self._execute_step(step, plan_result, request_params)

                plan_result.add_step_result(step_result)
                total_steps_executed += 1

            # Check if we need to break due to hitting max steps
            if getattr(plan_result, "max_iterations_reached", False):
                break

            # If the plan is marked complete, finalize the result
            if plan.is_complete:
                plan_result.is_complete = True
                break

            # Increment iteration counter
            iterations += 1

        # Generate final result based on execution status
        if iterations >= max_iterations and not plan_result.is_complete:
            self.logger.warning(f"Failed to complete in {max_iterations} iterations")
            plan_result.max_iterations_reached = True

            # Use incomplete plan template
            synthesis_prompt = SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE.format(
                plan_result=format_plan_result(plan_result), max_iterations=max_iterations
            )
        else:
            # Either plan is complete or we had other limits
            if not plan_result.is_complete:
                plan_result.is_complete = True

            # Use standard template
            synthesis_prompt = SYNTHESIZE_PLAN_PROMPT_TEMPLATE.format(
                plan_result=format_plan_result(plan_result)
            )

        # Generate final synthesis
        plan_result.result = await self._planner_generate_str(
            synthesis_prompt, request_params.model_copy(update={"max_iterations": 1})
        )

        return plan_result

    async def _execute_step(
        self, step: Step, previous_result: PlanResult, request_params: RequestParams
    ) -> Any:
        """
        Execute a single step from the plan.

        Args:
            step: The step to execute
            previous_result: Results of the plan execution so far
            request_params: Request parameters

        Returns:
            Result of executing the step
        """
        from mcp_agent.agents.workflow.orchestrator_models import StepResult

        # Initialize step result
        step_result = StepResult(step=step, task_results=[])

        # Format context for tasks
        context = format_plan_result(previous_result)

        # Execute all tasks in parallel
        futures = []
        error_tasks = []

        for task in step.tasks:
            # Check agent exists
            agent = self.agents.get(task.agent)
            if not agent:
                self.logger.error(
                    f"No agent found matching '{task.agent}'. Available agents: {list(self.agents.keys())}"
                )
                error_tasks.append(
                    (
                        task,
                        f"Error: Agent '{task.agent}' not found. Available agents: {', '.join(self.agents.keys())}",
                    )
                )
                continue

            # Prepare task prompt
            task_description = TASK_PROMPT_TEMPLATE.format(
                objective=previous_result.objective, task=task.description, context=context
            )

            # Queue task for execution
            futures.append(
                (
                    task,
                    agent.generate(
                        [
                            PromptMessageMultipart(
                                role="user",
                                content=[TextContent(type="text", text=task_description)],
                            )
                        ]
                    ),
                )
            )

        # Wait for all tasks
        task_results = []
        for future in futures:
            task, future_obj = future
            try:
                result = await future_obj
                result_text = result.all_text()

                # Create task result
                task_model = task.model_dump()
                task_result = TaskWithResult(
                    description=task_model["description"],
                    agent=task_model["agent"],
                    result=result_text,
                )
                task_results.append(task_result)
            except Exception as e:
                self.logger.error(f"Error executing task: {str(e)}")
                # Add error result
                task_model = task.model_dump()
                task_results.append(
                    TaskWithResult(
                        description=task_model["description"],
                        agent=task_model["agent"],
                        result=f"ERROR: {str(e)}",
                    )
                )

        # Add all task results to step result
        for task_result in task_results:
            step_result.add_task_result(task_result)

        # Add error task results
        for task, error_message in error_tasks:
            task_model = task.model_dump()
            step_result.add_task_result(
                TaskWithResult(
                    description=task_model["description"],
                    agent=task_model["agent"],
                    result=f"ERROR: {error_message}",
                )
            )

        # Format step result
        step_result.result = format_step_result_text(step_result)
        return step_result

    async def _get_full_plan(
        self, objective: str, plan_result: PlanResult, request_params: RequestParams
    ) -> Optional[Plan]:
        """
        Generate a full plan with all steps.

        Args:
            objective: The objective to achieve
            plan_result: Current plan execution state
            request_params: Request parameters

        Returns:
            Complete Plan with all steps, or None if parsing fails
        """
        # Format agent information for the prompt
        agent_formats = []
        for agent_name in self.agents.keys():
            formatted = self._format_agent_info(agent_name)
            agent_formats.append(formatted)

        agents = "\n".join(agent_formats)

        # Determine plan status
        if plan_result.is_complete:
            plan_status = "Plan Status: Complete"
        elif plan_result.step_results:
            plan_status = "Plan Status: In Progress"
        else:
            plan_status = "Plan Status: Not Started"

        # Calculate iteration information
        max_iterations = self.plan_iterations
        current_iteration = len(plan_result.step_results)
        current_iteration = min(current_iteration, max_iterations - 1)
        iterations_remaining = max(0, max_iterations - current_iteration - 1)
        iterations_info = f"Planning Budget: Iteration {current_iteration + 1} of {max_iterations} (with {iterations_remaining} remaining)"

        # Format the planning prompt
        prompt = FULL_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
            agents=agents,
        )

        # Get structured response from LLM
        try:
            plan_msg = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=prompt)]
            )
            plan, _ = await self._llm.structured([plan_msg], Plan, request_params)
            return plan
        except Exception as e:
            self.logger.error(f"Failed to parse plan: {str(e)}")
            return None

    async def _get_next_step(
        self, objective: str, plan_result: PlanResult, request_params: RequestParams
    ) -> Optional[NextStep]:
        """
        Generate just the next step for iterative planning.

        Args:
            objective: The objective to achieve
            plan_result: Current plan execution state
            request_params: Request parameters

        Returns:
            Next step to execute, or None if parsing fails
        """
        # Format agent information
        agents = "\n".join(
            [self._format_agent_info(agent_name) for agent_name in self.agents.keys()]
        )

        # Determine plan status
        if plan_result.is_complete:
            plan_status = "Plan Status: Complete"
        elif plan_result.step_results:
            plan_status = "Plan Status: In Progress"
        else:
            plan_status = "Plan Status: Not Started"

        # Calculate iteration information
        max_iterations = request_params.max_iterations
        current_iteration = len(plan_result.step_results)
        iterations_remaining = max_iterations - current_iteration
        iterations_info = (
            f"Planning Budget: {iterations_remaining} of {max_iterations} iterations remaining"
        )

        # Format the planning prompt
        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            plan_status=plan_status,
            iterations_info=iterations_info,
            agents=agents,
        )

        # Get structured response from LLM
        try:
            plan_msg = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=prompt)]
            )
            next_step, _ = await self._llm.structured([plan_msg], NextStep, request_params)
            return next_step
        except Exception as e:
            self.logger.error(f"Failed to parse next step: {str(e)}")
            return None

    def _validate_agent_names(self, plan: Plan) -> None:
        """
        Validate all agent names in a plan before execution.

        Args:
            plan: The plan to validate
        """
        if plan is None:
            self.logger.error("Cannot validate agent names: plan is None")
            return

        invalid_agents = []

        for step in plan.steps:
            for task in step.tasks:
                if task.agent not in self.agents:
                    invalid_agents.append(task.agent)

        if invalid_agents:
            available_agents = ", ".join(self.agents.keys())
            invalid_list = ", ".join(invalid_agents)
            self.logger.error(
                f"Plan contains invalid agent names: {invalid_list}. Available agents: {available_agents}"
            )

    def _format_agent_info(self, agent_name: str) -> str:
        """
        Format agent information for display in prompts.

        Args:
            agent_name: Name of the agent to format

        Returns:
            Formatted agent information string
        """
        agent = self.agents.get(agent_name)
        if not agent:
            self.logger.error(f"Agent '{agent_name}' not found in orchestrator agents")
            return ""

        # Get agent instruction or default description
        instruction = agent.instruction if agent.instruction else f"Agent '{agent_name}'"

        # Format with XML tags
        return f'<fastagent:agent name="{agent_name}">{instruction}</fastagent:agent>'

    async def _planner_generate_str(self, message: str, request_params: RequestParams) -> str:
        """
        Generate string response from the orchestrator's own LLM.

        Args:
            message: Message to send to the LLM
            request_params: Request parameters

        Returns:
            String response from the LLM
        """
        # Create prompt message
        prompt = PromptMessageMultipart(
            role="user", content=[TextContent(type="text", text=message)]
        )

        # Get response from LLM
        response = await self._llm.generate([prompt], request_params)
        return response.all_text()

    def _merge_request_params(self, request_params: Optional[RequestParams]) -> RequestParams:
        """
        Merge provided request parameters with defaults.

        Args:
            request_params: Optional request parameters to merge

        Returns:
            Merged request parameters
        """
        # Create orchestrator-specific defaults
        defaults = RequestParams(
            use_history=False,  # Orchestrator doesn't use history
            max_iterations=5,  # Default to 5 iterations
            maxTokens=8192,  # Higher limit for planning
            parallel_tool_calls=True,
        )

        # If base params provided, merge with defaults
        if request_params:
            # Create copy of defaults
            params = defaults.model_copy()
            # Update with provided params
            if isinstance(request_params, dict):
                params = params.model_copy(update=request_params)
            else:
                params = params.model_copy(update=request_params.model_dump())

            # Force specific settings
            params.use_history = False
            return params

        return defaults
--- END OF FILE workflow/orchestrator_agent.py ---


--- START OF FILE workflow/orchestrator_models.py ---
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from mcp_agent.agents.workflow.orchestrator_prompts import (
    PLAN_RESULT_TEMPLATE,
    STEP_RESULT_TEMPLATE,
    TASK_RESULT_TEMPLATE,
)


class Task(BaseModel):
    """An individual task that needs to be executed"""

    description: str = Field(description="Description of the task")


class ServerTask(Task):
    """An individual task that can be accomplished by one or more MCP servers"""

    servers: List[str] = Field(
        description="Names of MCP servers that the LLM has access to for this task",
        default_factory=list,
    )


class AgentTask(Task):
    """An individual task that can be accomplished by an Agent."""

    agent: str = Field(
        description="Name of Agent from given list of agents that the LLM has access to for this task",
    )


class Step(BaseModel):
    """A step containing independent tasks that can be executed in parallel"""

    description: str = Field(description="Description of the step")

    tasks: List[AgentTask] = Field(
        description="Subtasks that can be executed in parallel",
        default_factory=list,
    )


class Plan(BaseModel):
    """Plan generated by the orchestrator planner."""

    steps: List[Step] = Field(
        description="List of steps to execute sequentially",
        default_factory=list,
    )
    is_complete: bool = Field(description="Whether the overall plan objective is complete")


class TaskWithResult(Task):
    """An individual task with its result"""

    result: str = Field(description="Result of executing the task", default="Task completed")

    agent: str = Field(description="Name of the agent that executed this task", default="")

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class StepResult(BaseModel):
    """Result of executing a step"""

    step: Step = Field(description="The step that was executed", default_factory=Step)
    task_results: List[TaskWithResult] = Field(
        description="Results of executing each task", default_factory=list
    )
    result: str = Field(description="Result of executing the step", default="Step completed")

    def add_task_result(self, task_result: TaskWithResult) -> None:
        """Add a task result to this step"""
        if not isinstance(self.task_results, list):
            self.task_results = []
        self.task_results.append(task_result)


class PlanResult(BaseModel):
    """Results of executing a plan"""

    objective: str
    """Objective of the plan"""

    plan: Plan | None = None
    """The plan that was executed"""

    step_results: List[StepResult]
    """Results of executing each step"""

    is_complete: bool = False
    """Whether the overall plan objective is complete"""

    max_iterations_reached: bool = False
    """Whether the plan execution reached the maximum number of iterations without completing"""

    result: str | None = None
    """Result of executing the plan"""

    def add_step_result(self, step_result: StepResult) -> None:
        """Add a step result to this plan"""
        if not isinstance(self.step_results, list):
            self.step_results = []
        self.step_results.append(step_result)


class NextStep(Step):
    """Single next step in iterative planning"""

    is_complete: bool = Field(description="Whether the overall plan objective is complete")


def format_task_result_text(task_result: TaskWithResult) -> str:
    """Format a task result as plain text for display"""
    return TASK_RESULT_TEMPLATE.format(
        task_description=task_result.description, task_result=task_result.result
    )


def format_step_result_text(step_result: StepResult) -> str:
    """Format a step result as plain text for display"""
    tasks_str = "\n".join(
        f"  - {format_task_result_text(task)}" for task in step_result.task_results
    )
    return STEP_RESULT_TEMPLATE.format(
        step_description=step_result.step.description,
        step_result=step_result.result,
        tasks_str=tasks_str,
    )


def format_plan_result_text(plan_result: PlanResult) -> str:
    """Format the full plan execution state as plain text for display"""
    steps_str = (
        "\n\n".join(
            f"{i + 1}:\n{format_step_result_text(step)}"
            for i, step in enumerate(plan_result.step_results)
        )
        if plan_result.step_results
        else "No steps executed yet"
    )

    return PLAN_RESULT_TEMPLATE.format(
        plan_objective=plan_result.objective,
        steps_str=steps_str,
        plan_result=plan_result.result if plan_result.is_complete else "In Progress",
    )


def format_task_result_xml(task_result: TaskWithResult) -> str:
    """Format a task result with XML tags for better semantic understanding"""
    from mcp_agent.llm.prompt_utils import format_fastagent_tag

    return format_fastagent_tag(
        "task-result",
        f"\n<fastagent:description>{task_result.description}</fastagent:description>\n"
        f"<fastagent:result>{task_result.result}</fastagent:result>\n",
        {
            "description": task_result.description[:50] + "..."
            if len(task_result.description) > 50
            else task_result.description
        },
    )


def format_step_result_xml(step_result: StepResult) -> str:
    """Format a step result with XML tags for better semantic understanding"""
    from mcp_agent.llm.prompt_utils import format_fastagent_tag

    # Format each task result with XML
    task_results = []
    for task in step_result.task_results:
        task_results.append(format_task_result_xml(task))

    # Combine task results
    task_results_str = "\n".join(task_results)

    # Build step result with metadata and tasks
    step_content = (
        f"<fastagent:description>{step_result.step.description}</fastagent:description>\n"
        f"<fastagent:summary>{step_result.result}</fastagent:summary>\n"
        f"<fastagent:task-results>\n{task_results_str}\n</fastagent:task-results>\n"
    )

    return format_fastagent_tag("step-result", step_content)


def format_plan_result(plan_result: PlanResult) -> str:
    """Format the full plan execution state with XML for better semantic understanding"""
    from mcp_agent.llm.prompt_utils import format_fastagent_tag

    # Format objective
    objective_tag = format_fastagent_tag("objective", plan_result.objective)

    # Format step results
    step_results = []
    for step in plan_result.step_results:
        step_results.append(format_step_result_xml(step))

    # Build progress section
    if step_results:
        steps_content = "\n".join(step_results)
        progress_content = (
            f"{objective_tag}\n"
            f"<fastagent:steps>\n{steps_content}\n</fastagent:steps>\n"
            f"<fastagent:status>{plan_result.result if plan_result.is_complete else 'In Progress'}</fastagent:status>\n"
        )
    else:
        # No steps executed yet
        progress_content = (
            f"{objective_tag}\n"
            f"<fastagent:steps>No steps executed yet</fastagent:steps>\n"
            f"<fastagent:status>Not Started</fastagent:status>\n"
        )

    return format_fastagent_tag("progress", progress_content)
--- END OF FILE workflow/orchestrator_models.py ---


--- START OF FILE workflow/orchestrator_prompts.py ---
"""
Prompt templates used by the Orchestrator workflow.
"""

# Templates for formatting results
TASK_RESULT_TEMPLATE = """Task: {task_description}
Result: {task_result}"""

STEP_RESULT_TEMPLATE = """Step: {step_description}
Step Subtasks:
{tasks_str}"""

PLAN_RESULT_TEMPLATE = """Plan Objective: {plan_objective}

Progress So Far (steps completed):
{steps_str}

Result: {plan_result}"""

FULL_PLAN_PROMPT_TEMPLATE = """You are tasked with orchestrating a plan to complete an objective.
You can analyze results from the previous steps already executed to decide if the objective is complete.

<fastagent:data>
<fastagent:objective>
{objective}
</fastagent:objective>

<fastagent:available-agents>
{agents}
</fastagent:available-agents>

<fastagent:progress>
{plan_result}
</fastagent:progress>

<fastagent:status>
{plan_status}
{iterations_info}
</fastagent:status>
</fastagent:data>

Your plan must be structured in sequential steps, with each step containing independent parallel subtasks.
If the previous results achieve the objective, return is_complete=True.
Otherwise, generate remaining steps needed.

<fastagent:instruction>
You are operating in "full plan" mode, where you generate a complete plan with ALL remaining steps needed.
After receiving your plan, the system will execute ALL steps in your plan before asking for your input again.
If the plan needs multiple iterations, you'll be called again with updated results.

Generate a plan with all remaining steps needed.
Steps are sequential, but each Step can have parallel subtasks.
For each Step, specify a description of the step and independent subtasks that can run in parallel.
For each subtask specify:
    1. Clear description of the task that an LLM can execute  
    2. Name of 1 Agent from the available agents list above
    
CRITICAL: You MUST ONLY use agent names that are EXACTLY as they appear in <fastagent:available-agents> above.
Do NOT invent new agents. Do NOT modify agent names. The plan will FAIL if you use an agent that doesn't exist.

Return your response in the following JSON structure:
    {{
        "steps": [
            {{
                "description": "Description of step 1",
                "tasks": [
                    {{
                        "description": "Description of task 1",
                        "agent": "agent_name"  // agent MUST be exactly one of the agent names listed above
                    }},
                    {{
                        "description": "Description of task 2", 
                        "agent": "agent_name2"  // agent MUST be exactly one of the agent names listed above
                    }}
                ]
            }}
        ],
        "is_complete": false
    }}

Set "is_complete" to true when ANY of these conditions are met:
1. The objective has been achieved in full or substantively
2. The remaining work is minor or trivial compared to what's been accomplished
3. Additional steps provide minimal value toward the core objective
4. The plan has gathered sufficient information to answer the original request

Be decisive - avoid excessive planning steps that add little value. It's better to complete a plan early than to continue with marginal improvements. Focus on the core intent of the objective, not perfection.

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences.
</fastagent:instruction>
"""

ITERATIVE_PLAN_PROMPT_TEMPLATE = """You are tasked with determining only the next step in a plan
needed to complete an objective. You must analyze the current state and progress from previous steps 
to decide what to do next.

<fastagent:data>
<fastagent:objective>
{objective}
</fastagent:objective>

<fastagent:available-agents>
{agents}
</fastagent:available-agents>

<fastagent:progress>
{plan_result}
</fastagent:progress>

<fastagent:status>
{plan_status}
{iterations_info}
</fastagent:status>
</fastagent:data>

A Step must be sequential in the plan, but can have independent parallel subtasks. Only return a single Step.
If the previous results achieve the objective, return is_complete=True.
Otherwise, generate the next Step.

<fastagent:instruction>
You are operating in "iterative plan" mode, where you generate ONLY ONE STEP at a time.
After each step is executed, you'll be called again to determine the next step based on updated results.

Generate the next step, by specifying a description of the step and independent subtasks that can run in parallel:
For each subtask specify:
    1. Clear description of the task that an LLM can execute  
    2. Name of 1 Agent from the available agents list above

CRITICAL: You MUST ONLY use agent names that are EXACTLY as they appear in <fastagent:available-agents> above.
Do NOT invent new agents. Do NOT modify agent names. The plan will FAIL if you use an agent that doesn't exist.

Return your response in the following JSON structure:
    {{
        "description": "Description of step 1",
        "tasks": [
            {{
                "description": "Description of task 1",
                "agent": "agent_name"  // agent MUST be exactly one of the agent names listed above
            }}
        ],
        "is_complete": false
    }}

Set "is_complete" to true when ANY of these conditions are met:
1. The objective has been achieved in full or substantively
2. The remaining work is minor or trivial compared to what's been accomplished
3. Additional steps provide minimal value toward the core objective
4. The plan has gathered sufficient information to answer the original request

Be decisive - avoid excessive planning steps that add little value. It's better to complete a plan early than to continue with marginal improvements. Focus on the core intent of the objective, not perfection.

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences.
</fastagent:instruction>
"""

TASK_PROMPT_TEMPLATE = """You are part of a larger workflow to achieve an objective.

<fastagent:data>
<fastagent:objective>
{objective}
</fastagent:objective>

<fastagent:task>
{task}
</fastagent:task>

<fastagent:context>
{context}
</fastagent:context>
</fastagent:data>

<fastagent:instruction>
Your job is to accomplish only the task specified above.
Use the context from previous steps to inform your approach.
The context contains structured XML with the results from previous steps - pay close attention to:
- The objective in <fastagent:objective>
- Previous step results in <fastagent:steps>
- Task results and their attribution in <fastagent:task-result>

Provide a direct, focused response that addresses the task.
</fastagent:instruction>
"""

SYNTHESIZE_STEP_PROMPT_TEMPLATE = """You need to synthesize the results of parallel tasks into a cohesive result.

<fastagent:data>
<fastagent:step-results>
{step_result}
</fastagent:step-results>
</fastagent:data>

<fastagent:instruction>
Analyze the results from all tasks in this step.
Each task was executed by a specific agent (finder, writer, etc.)
Consider the expertise of each agent when weighing their results.
Combine the information into a coherent, unified response.
Focus on key insights and important outcomes.
Resolve any conflicting information if present.
</fastagent:instruction>
"""

SYNTHESIZE_PLAN_PROMPT_TEMPLATE = """You need to synthesize the results of all completed plan steps into a final response.

<fastagent:data>
<fastagent:plan-results>
{plan_result}
</fastagent:plan-results>
</fastagent:data>

<fastagent:instruction>
Create a comprehensive final response that addresses the original objective.
Integrate all the information gathered across all plan steps.
Provide a clear, complete answer that achieves the objective.
Focus on delivering value through your synthesis, not just summarizing.

If the plan was marked as incomplete but the maximum number of iterations was reached,
make sure to state clearly what was accomplished and what remains to be done.
</fastagent:instruction>
"""

# New template for incomplete plans due to iteration limits
SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE = """You need to synthesize the results of all completed plan steps into a final response.

<fastagent:data>
<fastagent:plan-results>
{plan_result}
</fastagent:plan-results>
</fastagent:data>

<fastagent:status>
The maximum number of iterations ({max_iterations}) was reached before the objective could be completed.
</fastagent:status>

<fastagent:instruction>
Create a comprehensive response that summarizes what was accomplished so far.
The objective was NOT fully completed due to reaching the maximum number of iterations.

In your response:
1. Clearly state that the objective was not fully completed
2. Summarize what WAS accomplished across all the executed steps
3. Identify what remains to be done to complete the objective
4. Organize the information to provide maximum value despite being incomplete

Focus on being transparent about the incomplete status while providing as much value as possible.
</fastagent:instruction>
"""
--- END OF FILE workflow/orchestrator_prompts.py ---


--- START OF FILE workflow/parallel_agent.py ---
import asyncio
from typing import Any, List, Optional, Tuple

from mcp.types import TextContent
from opentelemetry import trace

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class ParallelAgent(BaseAgent):
    """
    LLMs can sometimes work simultaneously on a task (fan-out)
    and have their outputs aggregated programmatically (fan-in).
    This workflow performs both the fan-out and fan-in operations using LLMs.
    From the user's perspective, an input is specified and the output is returned.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.PARALLEL

    def __init__(
        self,
        config: AgentConfig,
        fan_in_agent: Agent,
        fan_out_agents: List[Agent],
        include_request: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize a ParallelLLM agent.

        Args:
            config: Agent configuration or name
            fan_in_agent: Agent that aggregates results from fan-out agents
            fan_out_agents: List of agents to execute in parallel
            include_request: Whether to include the original request in the aggregation
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, **kwargs)
        self.fan_in_agent = fan_in_agent
        self.fan_out_agents = fan_out_agents
        self.include_request = include_request

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Execute fan-out agents in parallel and aggregate their results with the fan-in agent.

        Args:
            multipart_messages: List of messages to send to the fan-out agents
            request_params: Optional parameters to configure the request

        Returns:
            The aggregated response from the fan-in agent
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Parallel: '{self.name}' generate"):
            # Execute all fan-out agents in parallel
            responses: List[PromptMessageMultipart] = await asyncio.gather(
                *[
                    agent.generate(multipart_messages, request_params)
                    for agent in self.fan_out_agents
                ]
            )

            # Extract the received message from the input
            received_message: Optional[str] = (
                multipart_messages[-1].all_text() if multipart_messages else None
            )

            # Convert responses to strings for aggregation
            string_responses = []
            for response in responses:
                string_responses.append(response.all_text())

            # Format the responses and send to the fan-in agent
            aggregated_prompt = self._format_responses(string_responses, received_message)

            # Create a new multipart message with the formatted responses
            formatted_prompt = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=aggregated_prompt)]
            )

            # Use the fan-in agent to aggregate the responses
            return await self.fan_in_agent.generate([formatted_prompt], request_params)

    def _format_responses(self, responses: List[Any], message: Optional[str] = None) -> str:
        """
        Format a list of responses for the fan-in agent.

        Args:
            responses: List of responses from fan-out agents
            message: Optional original message that was sent to the agents

        Returns:
            Formatted string with responses
        """
        formatted = []

        # Include the original message if specified
        if self.include_request and message:
            formatted.append("The following request was sent to the agents:")
            formatted.append(f"<fastagent:request>\n{message}\n</fastagent:request>")

        # Format each agent's response
        for i, response in enumerate(responses):
            agent_name = self.fan_out_agents[i].name
            formatted.append(
                f'<fastagent:response agent="{agent_name}">\n{response}\n</fastagent:response>'
            )
        return "\n\n".join(formatted)

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Apply the prompt and return the result as a Pydantic model.

        This implementation delegates to the fan-in agent's structured method.

        Args:
            prompt: List of PromptMessageMultipart objects
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            An instance of the specified model, or None if coercion fails
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Parallel: '{self.name}' generate"):
            # Generate parallel responses first
            responses: List[PromptMessageMultipart] = await asyncio.gather(
                *[
                    agent.generate(multipart_messages, request_params)
                    for agent in self.fan_out_agents
                ]
            )

            # Extract the received message
            received_message: Optional[str] = (
                multipart_messages[-1].all_text() if multipart_messages else None
            )

            # Convert responses to strings
            string_responses = [response.all_text() for response in responses]

            # Format the responses for the fan-in agent
            aggregated_prompt = self._format_responses(string_responses, received_message)

            # Create a multipart message
            formatted_prompt = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text=aggregated_prompt)]
            )

            # Use the fan-in agent to parse the structured output
            return await self.fan_in_agent.structured([formatted_prompt], model, request_params)

    async def initialize(self) -> None:
        """
        Initialize the agent and its fan-in and fan-out agents.
        """
        await super().initialize()

        # Initialize fan-in and fan-out agents if not already initialized
        if not getattr(self.fan_in_agent, "initialized", False):
            await self.fan_in_agent.initialize()

        for agent in self.fan_out_agents:
            if not getattr(agent, "initialized", False):
                await agent.initialize()

    async def shutdown(self) -> None:
        """
        Shutdown the agent and its fan-in and fan-out agents.
        """
        await super().shutdown()

        # Shutdown fan-in and fan-out agents
        try:
            await self.fan_in_agent.shutdown()
        except Exception as e:
            self.logger.warning(f"Error shutting down fan-in agent: {str(e)}")

        for agent in self.fan_out_agents:
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down fan-out agent {agent.name}: {str(e)}")
--- END OF FILE workflow/parallel_agent.py ---


--- START OF FILE workflow/router_agent.py ---
"""
Router agent implementation using the BaseAgent adapter pattern.

This provides a simplified implementation that routes messages to agents
by determining the best agent for a request and dispatching to it.
"""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type

from opentelemetry import trace
from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol, ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from a2a.types import AgentCard

    from mcp_agent.context import Context

logger = get_logger(__name__)

# Simple system instruction for the router
ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate agent.
Analyze each request and determine which specialized agent would be best suited to handle it based on their capabilities.

Follow these guidelines:
- Carefully match the request's needs with each agent's capabilities and description
- Select the single most appropriate agent for the request
- Provide your confidence level (high, medium, low) and brief reasoning for your selection
"""

# Default routing instruction with placeholders for context (AgentCard JSON)
DEFAULT_ROUTING_INSTRUCTION = """
Select from the following agents to handle the request:
<fastagent:agents>
[
{context}
]
</fastagent:agents>

You must respond with the 'name' of one of the agents listed above.

"""


class RoutingResponse(BaseModel):
    """Model for the structured routing response from the LLM."""

    agent: str
    confidence: str
    reasoning: str | None = None


class RouterAgent(BaseAgent):
    """
    A simplified router that uses an LLM to determine the best agent for a request,
    then dispatches the request to that agent and returns the response.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.ROUTER

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        routing_instruction: Optional[str] = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a RouterAgent.

        Args:
            config: Agent configuration or name
            agents: List of agents to route between
            routing_instruction: Optional custom routing instruction
            context: Optional application context
            default_request_params: Optional default request parameters
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config=config, context=context, **kwargs)

        if not agents:
            raise AgentConfigError("At least one agent must be provided")

        self.agents = agents
        self.routing_instruction = routing_instruction
        self.agent_map = {agent.name: agent for agent in agents}

        # Set up base router request parameters
        base_params = {"systemPrompt": ROUTING_SYSTEM_INSTRUCTION, "use_history": False}

        if default_request_params:
            merged_params = default_request_params.model_copy(update=base_params)
        else:
            merged_params = RequestParams(**base_params)

        self._default_request_params = merged_params

    async def initialize(self) -> None:
        """Initialize the router and all agents."""
        if not self.initialized:
            await super().initialize()

            # Initialize all agents if not already initialized
            for agent in self.agents:
                if not getattr(agent, "initialized", False):
                    await agent.initialize()

            self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the router and all agents."""
        await super().shutdown()

        # Shutdown all agents
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent: {str(e)}")

    async def attach_llm(
        self,
        llm_factory: type[AugmentedLLMProtocol] | Callable[..., AugmentedLLMProtocol],
        model: str | None = None,
        request_params: RequestParams | None = None,
        **additional_kwargs,
    ) -> AugmentedLLMProtocol:
        return await super().attach_llm(
            llm_factory, model, request_params, verb="Routing", **additional_kwargs
        )

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Route the request to the most appropriate agent and return its response.

        Args:
            multipart_messages: Messages to route
            request_params: Optional request parameters

        Returns:
            The response from the selected agent
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Routing: '{self.name}' generate"):
            route, warn = await self._route_request(multipart_messages[-1])

            if not route:
                return Prompt.assistant(warn or "No routing result or warning received")

            # Get the selected agent
            agent: Agent = self.agent_map[route.agent]

            # Dispatch the request to the selected agent
            return await agent.generate(multipart_messages, request_params)

    async def structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Route the request to the most appropriate agent and parse its response.

        Args:
            prompt: Messages to route
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the selected agent, or None if parsing fails
        """

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"Routing: '{self.name}' structured"):
            route, warn = await self._route_request(multipart_messages[-1])

            if not route:
                return None, Prompt.assistant(
                    warn or "No routing result or warning received (structured)"
                )

            # Get the selected agent
            agent: Agent = self.agent_map[route.agent]

            # Dispatch the request to the selected agent
            return await agent.structured(multipart_messages, model, request_params)

    async def _route_request(
        self, message: PromptMessageMultipart
    ) -> Tuple[RoutingResponse | None, str | None]:
        """
        Determine which agent to route the request to.

        Args:
            request: The request to route

        Returns:
            RouterResult containing the selected agent, or None if no suitable agent was found
        """
        if not self.agents:
            logger.error("No agents available for routing")
            raise AgentConfigError("No agents available for routing - fatal error")

        # If only one agent is available, use it directly
        if len(self.agents) == 1:
            return RoutingResponse(
                agent=self.agents[0].name, confidence="high", reasoning="Only one agent available"
            ), None

        # Generate agent descriptions for the context
        agent_descriptions = []
        for agent in self.agents:
            agent_card: AgentCard = await agent.agent_card()
            agent_descriptions.append(
                agent_card.model_dump_json(
                    include={"name", "description", "skills"}, exclude_none=True
                )
            )

        context = ",\n".join(agent_descriptions)

        # Format the routing prompt
        routing_instruction = self.routing_instruction or DEFAULT_ROUTING_INSTRUCTION
        routing_instruction = routing_instruction.format(context=context)

        assert self._llm
        mutated = message.model_copy(deep=True)
        mutated.add_text(routing_instruction)
        response, _ = await self._llm.structured(
            [mutated],
            RoutingResponse,
            self._default_request_params,
        )

        warn: str | None = None
        if not response:
            warn = "No routing response received from LLM"
        elif response.agent not in self.agent_map:
            warn = f"A response was received, but the agent {response.agent} was not known to the Router"

        if warn:
            logger.warning(warn)
            return None, warn
        else:
            assert response
            logger.info(
                f"Routing structured request to agent: {response.agent or 'error'} (confidence: {response.confidence or ''})"
            )

            return response, None
--- END OF FILE workflow/router_agent.py ---


