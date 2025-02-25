import asyncio
from contextlib import AsyncExitStack

from dhenara.ai.types import AIModelCallConfig, AIModelCallResponse, AIModelEndpoint

from .factory import AIModelClientFactory


class AIModelClient:
    """
    A high-level client for making AI model calls with automatic resource management.

    This client handles:
    - Connection lifecycle management
    - Automatic retries with exponential backoff
    - Request timeouts
    - Resource cleanup

    Attributes:
        model_endpoint (AIModelEndpoint): The AI model endpoint configuration
        config (AIModelCallConfig): Configuration for API calls including timeouts and retries
    """

    def __init__(
        self,
        model_endpoint: AIModelEndpoint,
        config: AIModelCallConfig | None = None,
    ):
        """
        Initialize the AI Model Client.

        Args:
            model_endpoint: Configuration for the AI model endpoint
            config: Optional configuration for API calls. If not provided, default config is used.
        """
        self.model_endpoint = model_endpoint
        self.config = config or AIModelCallConfig()
        self._provider_client = None
        self._client_stack = AsyncExitStack()

    async def __aenter__(self):
        """Initialize and return the client in a context manager."""
        self._provider_client = await self._client_stack.enter_async_context(
            AIModelClientFactory.create_provider_client(
                self.model_endpoint,
                self.config,
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context manager."""
        await self._client_stack.aclose()
        self._provider_client = None

    async def generate(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        """
        Generate a response from the AI model with automatic context management.

        This method handles the complete lifecycle of the request including connection
        management, retries, and timeouts.

        Args:
            prompt: The primary input prompt for the AI model
            context: Optional list of previous conversation context
            instructions: Optional system instructions for the AI model

        Returns:
            AIModelCallResponse: The generated response from the AI model

        Raises:
            TimeoutError: If the request exceeds the configured timeout
            RuntimeError: If all retry attempts fail
        """
        async with self as client:  # noqa: F841
            return await self._execute_with_retry(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )

    async def generate_with_existing_connection(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        """
        Generate a response using an existing connection or create a new one.

        This method is useful for making multiple calls efficiently when you want to
        reuse the same connection. Note that you're responsible for cleaning up
        resources when using this method.

        Args:
            prompt: The primary input prompt for the AI model
            context: Optional list of previous conversation context
            instructions: Optional system instructions for the AI model

        Returns:
            AIModelCallResponse: The generated response from the AI model

        Note:
            This method doesn't automatically clean up resources. Use `cleanup()`
            when you're done making calls.
        """
        if not self._provider_client:
            self._provider_client = await self._client_stack.enter_async_context(
                AIModelClientFactory.create_provider_client(
                    self.model_endpoint,
                    self.config,
                ),
            )
        return await self._execute_with_retry(
            prompt=prompt,
            context=context,
            instructions=instructions,
        )

    async def cleanup(self) -> None:
        """
        Clean up resources manually.

        Call this method when you're done using generate_with_existing_connection()
        to ensure proper resource cleanup.
        """
        await self._client_stack.aclose()
        self._provider_client = None

    async def _execute_with_retry(
        self,
        prompt: dict,
        context: list[dict] | None,
        instructions: list[str] | None,
    ) -> AIModelCallResponse:
        """Execute the request with retry logic."""
        last_exception = None
        for attempt in range(self.config.retries):
            try:
                return await self._execute_with_timeout(
                    prompt=prompt,
                    context=context,
                    instructions=instructions,
                )
            except asyncio.TimeoutError as e:  # noqa: PERF203
                last_exception = e
                if attempt == self.config.retries - 1:
                    break
                delay = min(
                    self.config.retry_delay * (2**attempt),
                    self.config.max_retry_delay,
                )
                await asyncio.sleep(delay)

        raise last_exception or RuntimeError("All retry attempts failed")

    async def _execute_with_timeout(
        self,
        prompt: dict,
        context: list[dict] | None,
        instructions: list[str] | None,
    ) -> AIModelCallResponse:
        """Execute the request with timeout handling."""
        async with AsyncExitStack() as stack:
            if self.config.timeout:
                await stack.enter_async_context(asyncio.timeout(self.config.timeout))
            return await self._provider_client._validate_and_generate_response(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )
