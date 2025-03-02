import asyncio
import concurrent.futures
import functools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class SyncWrapperConfig:
    max_workers: int = 10
    default_timeout: float | None = None
    thread_name_prefix: str = "async_wrapper_"
    shutdown_timeout: float = 60.0


class AsyncToSyncWrapper:
    """
    Handles conversion of async functions to sync functions with proper
    thread and event loop management.
    """

    def __init__(self, config: SyncWrapperConfig | None = None):
        self._config = config or SyncWrapperConfig()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def __call__(self, async_func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(async_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                return self._run_in_new_loop(async_func, *args, **kwargs)
            return self._run_async_func(async_func, *args, **kwargs)

        return wrapper

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._config.max_workers,
                    thread_name_prefix=self._config.thread_name_prefix,
                )
            return self._executor

    def _run_in_new_loop(
        self,
        async_func,
        *args,
        timeout: float | None = None,
        **kwargs,
    ):
        future = concurrent.futures.Future()
        timeout = timeout or self._config.default_timeout

        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(async_func(*args, **kwargs))
                    future.set_result(result)
                finally:
                    loop.close()
            except Exception as e:
                logger.exception("Error in async execution")
                future.set_exception(e)

        self._get_executor().submit(run_in_thread)
        return future.result(timeout=timeout)

    def _run_async_func(self, async_func, *args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))

    def cleanup(self):
        with self._lock:
            if self._executor:
                try:
                    self._executor.shutdown(wait=True, timeout=self._config.shutdown_timeout)
                except Exception:
                    logger.exception("Error during executor shutdown")
                finally:
                    self._executor = None


sync_wrapper = AsyncToSyncWrapper()
# def sync_wrapper(async_func):
#    """Decorator to convert async functions to sync functions"""
#
#    @wraps(async_func)
#    def sync_func(*args, **kwargs):
#        return asyncio.run(async_func(*args, **kwargs))
#
#    return sync_func
