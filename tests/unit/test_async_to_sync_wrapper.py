# Provenance: Added to improve decorator coverage (2026-01-20)

from __future__ import annotations

import asyncio

import pytest

from dhenara.ai.ai_client._decorators import AsyncToSyncWrapper, SyncWrapperConfig

pytestmark = [pytest.mark.unit]


@pytest.mark.case_id("DAI-037")
def test_dai_037_runs_async_from_sync():
    """GIVEN an async function
    WHEN it is wrapped via AsyncToSyncWrapper and called from sync code
    THEN the result is returned without requiring an event loop.
    """

    async def plus_one(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    wrapper = AsyncToSyncWrapper(SyncWrapperConfig(default_timeout=1.0))
    sync_plus_one = wrapper(plus_one)

    assert sync_plus_one(41) == 42

    wrapper.cleanup()


@pytest.mark.case_id("DAI-038")
def test_dai_038_rejects_running_loop():
    """GIVEN an AsyncToSyncWrapper wrapped function
    WHEN it is called from inside an active asyncio event loop
    THEN it raises to avoid blocking the event loop thread.
    """

    async def plus_one(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    wrapper = AsyncToSyncWrapper()
    sync_plus_one = wrapper(plus_one)

    async def _inner() -> None:
        with pytest.raises(RuntimeError, match=r"running event loop"):
            sync_plus_one(1)

    asyncio.run(_inner())

    wrapper.cleanup()


@pytest.mark.case_id("DAI-055")
def test_dai_055_run_in_new_loop_and_cleanup():
    """GIVEN an AsyncToSyncWrapper
    WHEN _run_in_new_loop is invoked directly
    THEN it executes the coroutine in a background loop and cleanup clears the executor.
    """

    async def ok() -> int:
        await asyncio.sleep(0)
        return 7

    async def boom() -> int:
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    wrapper = AsyncToSyncWrapper(SyncWrapperConfig(default_timeout=1.0, max_workers=1))

    assert wrapper._run_in_new_loop(ok) == 7
    with pytest.raises(RuntimeError, match="boom"):
        wrapper._run_in_new_loop(boom)

    # Ensure executor is created and then cleared
    _ = wrapper._get_executor()
    assert wrapper._executor is not None
    wrapper.cleanup()
    assert wrapper._executor is None
