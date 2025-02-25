import concurrent.futures
import functools
import logging
from typing import Optional

from dhenara.ai.types import AIModelCallConfig, AIModelCallResponse, AIModelEndpoint

from ._decorators import AsyncToSyncWrapper, SyncWrapperConfig, sync_wrapper
from .ai_client import AIModelClient

logger = logging.getLogger(__name__)


class AIModelClientSync:
    """
    Synchronous wrapper for AIModelClient that provides a simplified interface
    while maintaining all the context management of the async implementation.
    """

    def __init__(
        self,
        model_endpoint: AIModelEndpoint,
        config: Optional[AIModelCallConfig] = None,
    ):
        self.async_client = AIModelClient(model_endpoint, config)
        self._config = config
        self._wrapper = AsyncToSyncWrapper(SyncWrapperConfig(default_timeout=config.timeout if config else None))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _handle_sync_errors(self, func):
        """Decorator to handle synchronous operation errors"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Operation {func.__name__} timed out after {self._config.timeout if self._config else 'default'} seconds")
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {type(e).__name__}: {e}", exc_info=True)
                raise

        return wrapper

    def generate(
        self,
        prompt: dict,
        context: Optional[list[dict]] = None,
        instructions: Optional[list[str]] = None,
    ) -> AIModelCallResponse:
        """
        Synchronous version of generate that handles context automatically
        """

        @self._handle_sync_errors
        @sync_wrapper
        async def _generate():
            async with self.async_client as client:
                return await client.generate(
                    prompt=prompt,
                    context=context,
                    instructions=instructions,
                )

        return _generate()

    async def generate_with_existing_connection(
        self,
        prompt: dict,
        context: Optional[list[dict]] = None,
        instructions: Optional[list[str]] = None,
    ) -> AIModelCallResponse:
        """
        Synchronous version of generate_with_existing_connection
        """

        @self._handle_sync_errors
        @sync_wrapper
        async def _generate_with_existing_connection():
            return await self.async_client.generate_with_existing_connection(
                prompt=prompt,
                context=context,
                instructions=instructions,
            )

        return _generate_with_existing_connection()

    def cleanup(self) -> None:
        """Clean up resources"""
        self._wrapper.cleanup()
