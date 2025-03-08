import asyncio
import signal
import time
from contextlib import AsyncExitStack, ExitStack, contextmanager

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
        is_async (bool): Async client or not
    """

    def __init__(
        self,
        model_endpoint: AIModelEndpoint,
        config: AIModelCallConfig | None = None,
        is_async: bool = True,
    ):
        self.model_endpoint = model_endpoint
        self.config = config or AIModelCallConfig()
        self.is_async = is_async
        self._provider_client = None
        self._client_stack = AsyncExitStack() if is_async else ExitStack()

    async def _async_context(self):
        try:
            self._provider_client = await self._client_stack.enter_async_context(
                AIModelClientFactory.create_provider_client(
                    self.model_endpoint,
                    self.config,
                    is_async=True,
                ),
            )
            yield self
        finally:
            await self._client_stack.aclose()
            self._provider_client = None

    @contextmanager
    def _sync_context(self):
        try:
            self._provider_client = self._client_stack.enter_context(
                AIModelClientFactory.create_provider_client(
                    self.model_endpoint,
                    self.config,
                    is_async=False,
                ),
            )
            yield self
        finally:
            self._client_stack.close()
            self._provider_client = None

    def __enter__(self):
        if self.is_async:
            raise RuntimeError("Use 'async with' for async client")
        # Store the context manager instance
        self._sync_ctx = self._sync_context()
        return self._sync_ctx.__enter__()

    def __exit__(self, *exc):
        if self.is_async:
            raise RuntimeError(
                f"This client is created with is_async={self.is_async}. Use 'async with' for async client"
            )
        # Use the stored context manager instance
        return self._sync_ctx.__exit__(*exc)

    async def __aenter__(self):
        if not self.is_async:
            raise RuntimeError(f"This client is created with is_async={self.is_async}. Use 'with' for sync client")
        self._async_ctx = self._async_context()
        return await anext(self._async_ctx.__aiter__())

    async def __aexit__(self, *exc):
        if not self.is_async:
            raise RuntimeError("Use 'with' for sync client")
        try:
            await self._async_ctx.aclose()
        except:
            pass

    def _execute_with_retry_sync(self, *args, **kwargs) -> AIModelCallResponse:
        """Synchronous retry logic"""
        last_exception = None
        for attempt in range(self.config.retries):
            try:
                return self._execute_with_timeout_sync(*args, **kwargs)
            except TimeoutError as e:  # noqa: PERF203 # TODO
                last_exception = e
                if attempt == self.config.retries - 1:
                    break
                delay = min(
                    self.config.retry_delay * (2**attempt),
                    self.config.max_retry_delay,
                )
                time.sleep(delay)

        raise last_exception or RuntimeError("All retry attempts failed")

    async def _execute_with_retry_async(self, *args, **kwargs) -> AIModelCallResponse:
        """Asynchronous retry logic"""
        last_exception = None
        for attempt in range(self.config.retries):
            try:
                return await self._execute_with_timeout_async(*args, **kwargs)
            except asyncio.TimeoutError as e:  # noqa: PERF203 # TODO
                last_exception = e
                if attempt == self.config.retries - 1:
                    break
                delay = min(
                    self.config.retry_delay * (2**attempt),
                    self.config.max_retry_delay,
                )
                await asyncio.sleep(delay)

        raise last_exception or RuntimeError("All retry attempts failed")

    def _execute_with_timeout_sync(self, *args, **kwargs) -> AIModelCallResponse:
        """Synchronous timeout handling"""
        with ExitStack() as stack:  # noqa: F841
            if self.config.timeout:
                signal.alarm(self.config.timeout)
            try:
                return self._provider_client._validate_and_generate_response_sync(*args, **kwargs)
            finally:
                if self.config.timeout:
                    signal.alarm(0)

    async def _execute_with_timeout_async(self, *args, **kwargs) -> AIModelCallResponse:
        """Asynchronous timeout handling"""
        async with AsyncExitStack() as stack:
            if self.config.timeout:
                await stack.enter_async_context(asyncio.timeout(self.config.timeout))
            return await self._provider_client._validate_and_generate_response_async(*args, **kwargs)

    # Genereate Response Fns
    def generate(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        """Synchronous generate method"""
        if self.is_async:
            raise RuntimeError(
                f"This client is created with is_async={self.is_async}. Use generate_async for async client"
            )

        with self as client:  # noqa: F841
            return self._execute_with_retry_sync(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )

    async def generate_async(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        """Asynchronous generate method"""
        if not self.is_async:
            raise RuntimeError(f"This client is created with is_async={self.is_async}. Use generate for sync client")

        async with self as client:  # noqa: F841
            return await self._execute_with_retry_async(
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
        if self.is_async:
            return self.generate_with_existing_connection_async(
                prompt=prompt, context=context, instructions=instructions
            )
        else:
            return self.generate_with_existing_connection_sync(
                prompt=prompt, context=context, instructions=instructions
            )

    def generate_with_existing_connection_sync(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
        if not self._provider_client:
            self._provider_client = self._client_stack.enter_async_context(
                AIModelClientFactory.create_provider_client(
                    self.model_endpoint,
                    self.config,
                ),
            )
        return self._execute_with_retry(
            prompt=prompt,
            context=context,
            instructions=instructions,
        )

    def cleanup_sync(self) -> None:
        """
        Clean up resources manually.

        Call this method when you're done using generate_with_existing_connection()
        to ensure proper resource cleanup.
        """
        self._client_stack.close()
        self._provider_client = None

    async def generate_with_existing_connection_async(
        self,
        prompt: dict,
        context: list[dict] | None = None,
        instructions: list[str] | None = None,
    ) -> AIModelCallResponse:
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

    async def cleanup_async(self) -> None:
        """
        Clean up resources manually.

        Call this method when you're done using generate_with_existing_connection()
        to ensure proper resource cleanup.
        """
        await self._client_stack.aclose()
        self._provider_client = None
