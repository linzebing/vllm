# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for multi-turn conversation caching."""

import time

import pytest

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


@pytest.fixture
def vllm_config():
    """Create a minimal VllmConfig for testing."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space_bytes=4 * 1024 * 1024 * 1024,  # 4GB
        cache_dtype="auto",
        num_gpu_blocks=1000,
        num_cpu_blocks=1000,
        enable_prefix_caching=True,
    )

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        max_model_len=2048,
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA not available",
)
class TestMultiTurnConversationIntegration:
    """Integration tests for multi-turn conversation caching."""

    def test_conversation_kv_cache_persistence(self, vllm_config):
        """Test that KV cache persists across conversation turns."""
        # Create scheduler (which will create KVCacheManager)
        scheduler = Scheduler(
            vllm_config=vllm_config,
            output_proc_callback=lambda x: None,
        )

        conversation_id = "test-conv-123"

        # Turn 0: Initial prompt
        request_0 = Request(
            request_id="req-0",
            prompt="Hello, how are you?",
            prompt_token_ids=[1, 2, 3, 4, 5],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id=conversation_id,
            turn_number=0,
            is_conversation_end=False,
        )

        # Add request to scheduler
        scheduler.add_request(request_0)

        # Schedule and execute (simplified - in real system, executor runs)
        scheduled_requests = scheduler.schedule()

        # Verify request was scheduled
        assert len(scheduled_requests.scheduled_new_reqs) > 0

        # Simulate completion of turn 0
        request_0.status = "finished"
        scheduler.finish_request(request_0.request_id, True)

        # Verify conversation was suspended (not freed)
        conv_state = scheduler.conversation_manager.get_conversation(conversation_id)
        assert conv_state is not None
        assert conv_state.is_active is False  # Suspended
        assert conv_state.turn_number == 0

        # Turn 1: Continuation prompt
        request_1 = Request(
            request_id="req-1",
            prompt="I'm doing well, thanks!",
            prompt_token_ids=[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],  # Includes previous + new tokens
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id=conversation_id,
            turn_number=1,
            is_conversation_end=False,
        )

        # Add continuation request
        scheduler.add_request(request_1)

        # Schedule - should resume conversation
        scheduled_requests = scheduler.schedule()

        # Verify request was scheduled with resumed KV cache
        assert len(scheduled_requests.scheduled_new_reqs) > 0

        # Verify conversation was resumed and updated
        conv_state = scheduler.conversation_manager.get_conversation(conversation_id)
        assert conv_state is not None
        assert conv_state.is_active is True  # Active again
        assert conv_state.turn_number == 1

        # Verify num_computed_tokens was transferred
        assert request_1.num_computed_tokens > 0

    def test_conversation_end_frees_kv_cache(self, vllm_config):
        """Test that KV cache is freed when conversation ends."""
        scheduler = Scheduler(
            vllm_config=vllm_config,
            output_proc_callback=lambda x: None,
        )

        conversation_id = "test-conv-end"

        # Turn 0
        request_0 = Request(
            request_id="req-end-0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id=conversation_id,
            turn_number=0,
            is_conversation_end=False,
        )

        scheduler.add_request(request_0)
        scheduler.schedule()

        # Complete turn 0
        request_0.status = "finished"
        scheduler.finish_request(request_0.request_id, True)

        # Verify conversation exists
        assert (
            scheduler.conversation_manager.get_conversation(conversation_id) is not None
        )

        # Turn 1 (final turn)
        request_1 = Request(
            request_id="req-end-1",
            prompt="Goodbye",
            prompt_token_ids=[1, 2, 3, 4, 5],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id=conversation_id,
            turn_number=1,
            is_conversation_end=True,  # Mark as final turn
        )

        scheduler.add_request(request_1)
        scheduler.schedule()

        # Complete turn 1
        request_1.status = "finished"
        scheduler.finish_request(request_1.request_id, True)

        # Verify conversation was ended and removed
        assert scheduler.conversation_manager.get_conversation(conversation_id) is None

    def test_multiple_concurrent_conversations(self, vllm_config):
        """Test handling multiple concurrent conversations."""
        scheduler = Scheduler(
            vllm_config=vllm_config,
            output_proc_callback=lambda x: None,
        )

        # Create 3 different conversations
        conv_ids = ["conv-A", "conv-B", "conv-C"]

        for i, conv_id in enumerate(conv_ids):
            request = Request(
                request_id=f"req-{conv_id}",
                prompt=f"Conversation {i}",
                prompt_token_ids=[1, 2, 3, i],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=10),
                eos_token_id=2,
                arrival_time=time.time(),
                lora_request=None,
                conversation_id=conv_id,
                turn_number=0,
                is_conversation_end=False,
            )
            scheduler.add_request(request)

        # Schedule all conversations
        scheduler.schedule()

        # Complete all conversations
        for conv_id in conv_ids:
            scheduler.finish_request(f"req-{conv_id}", True)

        # Verify all conversations are suspended
        for conv_id in conv_ids:
            conv_state = scheduler.conversation_manager.get_conversation(conv_id)
            assert conv_state is not None
            assert conv_state.is_active is False

        # Continue one of the conversations (conv-B)
        request_b1 = Request(
            request_id="req-conv-B-turn1",
            prompt="Continuation B",
            prompt_token_ids=[1, 2, 3, 1, 4, 5],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id="conv-B",
            turn_number=1,
            is_conversation_end=False,
        )

        scheduler.add_request(request_b1)
        scheduler.schedule()

        # Verify conv-B is active, others still suspended
        assert (
            scheduler.conversation_manager.get_conversation("conv-B").is_active is True
        )
        assert (
            scheduler.conversation_manager.get_conversation("conv-A").is_active is False
        )
        assert (
            scheduler.conversation_manager.get_conversation("conv-C").is_active is False
        )

    def test_conversation_timeout_cleanup_in_scheduler(self, vllm_config):
        """Test that scheduler cleans up timed-out conversations."""
        # Create manager with short timeout
        scheduler = Scheduler(
            vllm_config=vllm_config,
            output_proc_callback=lambda x: None,
        )
        scheduler.conversation_manager.conversation_timeout_seconds = 2.0

        conversation_id = "test-conv-timeout"

        # Create and complete a conversation turn
        request = Request(
            request_id="req-timeout",
            prompt="Test",
            prompt_token_ids=[1, 2, 3],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id=conversation_id,
            turn_number=0,
            is_conversation_end=False,
        )

        scheduler.add_request(request)
        scheduler.schedule()
        scheduler.finish_request(request.request_id, True)

        # Verify conversation exists
        assert (
            scheduler.conversation_manager.get_conversation(conversation_id) is not None
        )

        # Wait for timeout
        time.sleep(3)

        # Trigger cleanup (would normally happen during schedule())
        timed_out = scheduler.conversation_manager.cleanup_timed_out_conversations()

        # Verify conversation was cleaned up
        assert len(timed_out) == 1
        assert timed_out[0].conversation_id == conversation_id
        assert scheduler.conversation_manager.get_conversation(conversation_id) is None

    def test_non_conversation_requests_unchanged(self, vllm_config):
        """Test that normal requests (without conversation_id) work as before."""
        scheduler = Scheduler(
            vllm_config=vllm_config,
            output_proc_callback=lambda x: None,
        )

        # Create normal request (no conversation_id)
        request = Request(
            request_id="req-normal",
            prompt="Normal request",
            prompt_token_ids=[1, 2, 3, 4, 5],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=SamplingParams(max_tokens=10),
            eos_token_id=2,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id=None,  # No conversation ID
            turn_number=0,
            is_conversation_end=False,
        )

        scheduler.add_request(request)
        scheduler.schedule()

        # Complete request
        request.status = "finished"
        scheduler.finish_request(request.request_id, True)

        # Verify no conversation was created
        stats = scheduler.conversation_manager.get_stats()
        assert stats["total_conversations"] == 0


@pytest.mark.skip(reason="Requires full model loading and GPU execution")
class TestMultiTurnConversationEndToEnd:
    """End-to-end tests with actual model execution (expensive)."""

    def test_full_multi_turn_conversation(self):
        """
        Full end-to-end test with model loading and execution.
        This test is expensive and should only be run manually.
        """
        # TODO: Implement full e2e test with:
        # 1. Load actual model (e.g., facebook/opt-125m)
        # 2. Process multi-turn conversation
        # 3. Verify TTFT is faster on subsequent turns
        # 4. Verify output correctness
        pass

    def test_conversation_caching_performance_improvement(self):
        """
        Performance benchmark comparing with/without conversation caching.
        Should demonstrate 2-3x speedup for subsequent turns.
        """
        # TODO: Implement benchmark:
        # 1. Run multi-turn conversation with caching enabled
        # 2. Run same conversation with caching disabled
        # 3. Compare TTFT for turn 1, 2, 3...
        # 4. Verify speedup is 2-3x for turns > 0
        pass
