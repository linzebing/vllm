# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ConversationContextManager."""

import time

import pytest

from vllm.v1.conversation_context import (
    ConversationContextManager,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlocks
from vllm.v1.request import Request


@pytest.fixture
def conversation_manager():
    """Create a ConversationContextManager for testing."""
    return ConversationContextManager(
        enable_conversation_caching=True,
        conversation_timeout_seconds=5.0,  # Short timeout for testing
        max_active_conversations=10,
    )


@pytest.fixture
def sample_request():
    """Create a sample request for testing."""
    return Request(
        request_id="test-req-1",
        prompt="Hello world",
        prompt_token_ids=[1, 2, 3, 4],
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=None,
        eos_token_id=0,
        arrival_time=time.time(),
        lora_request=None,
        conversation_id="conv-123",
        turn_number=0,
        is_conversation_end=False,
    )


@pytest.fixture
def sample_kv_blocks():
    """Create sample KV cache blocks for testing."""
    # Create a mock KVCacheBlocks object
    # In practice, this would contain actual GPU block allocations
    return KVCacheBlocks(blocks=[[0, 1, 2]], num_computed_tokens=[4])


class TestConversationContextManager:
    """Test suite for ConversationContextManager."""

    def test_create_conversation(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test creating a new conversation."""
        conv_state = conversation_manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        assert conv_state.conversation_id == "conv-123"
        assert conv_state.turn_number == 0
        assert conv_state.num_computed_tokens == 4
        assert conv_state.all_token_ids == [1, 2, 3, 4]
        assert conv_state.is_active is True
        assert len(conv_state.kv_cache_blocks.blocks[0]) == 3

    def test_update_conversation(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test updating an existing conversation."""
        # First create the conversation
        conversation_manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        # Now update it with a new turn
        new_request = Request(
            request_id="test-req-2",
            prompt="How are you?",
            prompt_token_ids=[1, 2, 3, 4, 5, 6],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=None,
            eos_token_id=0,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id="conv-123",
            turn_number=1,
            is_conversation_end=False,
        )

        updated_state = conversation_manager.update_conversation(
            conversation_id="conv-123",
            request=new_request,
            turn_number=1,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=6,
            all_token_ids=[1, 2, 3, 4, 5, 6],
        )

        assert updated_state.turn_number == 1
        assert updated_state.num_computed_tokens == 6
        assert len(updated_state.all_token_ids) == 6
        assert updated_state.last_request.request_id == "test-req-2"

    def test_suspend_and_resume_conversation(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test suspending and resuming a conversation."""
        # Create conversation
        conversation_manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        # Suspend it
        conversation_manager.suspend_conversation("conv-123")
        conv_state = conversation_manager.get_conversation("conv-123")
        assert conv_state.is_active is False

        # Resume it
        resumed_state = conversation_manager.resume_conversation("conv-123")
        assert resumed_state is not None
        assert resumed_state.is_active is True
        assert resumed_state.conversation_id == "conv-123"

    def test_end_conversation(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test ending a conversation."""
        # Create conversation
        conversation_manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        # End it
        ended_state = conversation_manager.end_conversation("conv-123")
        assert ended_state is not None
        assert ended_state.conversation_id == "conv-123"

        # Should no longer exist
        assert conversation_manager.get_conversation("conv-123") is None

    def test_get_active_conversations(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test getting list of active conversations."""
        # Create multiple conversations
        for i in range(3):
            conv_id = f"conv-{i}"
            req = Request(
                request_id=f"req-{i}",
                prompt=f"Prompt {i}",
                prompt_token_ids=[1, 2, 3],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=None,
                eos_token_id=0,
                arrival_time=time.time(),
                lora_request=None,
                conversation_id=conv_id,
                turn_number=0,
                is_conversation_end=False,
            )
            conversation_manager.create_conversation(
                conversation_id=conv_id,
                request=req,
                kv_cache_blocks=sample_kv_blocks,
                num_computed_tokens=3,
                all_token_ids=[1, 2, 3],
                turn_number=0,
            )

        active = conversation_manager.get_active_conversations()
        assert len(active) == 3
        assert all(conv.is_active for conv in active)

    def test_conversation_timeout_cleanup(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test that timed out conversations are cleaned up."""
        # Create conversation
        conversation_manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        # Suspend it
        conversation_manager.suspend_conversation("conv-123")

        # Wait for timeout (5 seconds in our test fixture)
        time.sleep(6)

        # Run cleanup
        timed_out = conversation_manager.cleanup_timed_out_conversations()
        assert len(timed_out) == 1
        assert timed_out[0].conversation_id == "conv-123"

        # Should no longer exist
        assert conversation_manager.get_conversation("conv-123") is None

    def test_max_conversations_limit(self, sample_request, sample_kv_blocks):
        """Test that max conversations limit is enforced."""
        manager = ConversationContextManager(
            enable_conversation_caching=True,
            conversation_timeout_seconds=300.0,
            max_active_conversations=3,
        )

        # Create 3 conversations (should succeed)
        for i in range(3):
            conv_id = f"conv-{i}"
            req = Request(
                request_id=f"req-{i}",
                prompt=f"Prompt {i}",
                prompt_token_ids=[1, 2, 3],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=None,
                eos_token_id=0,
                arrival_time=time.time(),
                lora_request=None,
                conversation_id=conv_id,
                turn_number=0,
                is_conversation_end=False,
            )
            result = manager.create_conversation(
                conversation_id=conv_id,
                request=req,
                kv_cache_blocks=sample_kv_blocks,
                num_computed_tokens=3,
                all_token_ids=[1, 2, 3],
                turn_number=0,
            )
            assert result is not None

        # Try to create 4th conversation (should return None due to limit)
        req_4 = Request(
            request_id="req-4",
            prompt="Prompt 4",
            prompt_token_ids=[1, 2, 3],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=None,
            eos_token_id=0,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id="conv-4",
            turn_number=0,
            is_conversation_end=False,
        )

        # The manager will evict LRU conversation
        result = manager.create_conversation(
            conversation_id="conv-4",
            request=req_4,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=3,
            all_token_ids=[1, 2, 3],
            turn_number=0,
        )
        assert result is not None  # Should succeed by evicting LRU

        # Should have exactly 3 conversations (conv-0 was evicted)
        assert len(manager._conversations) == 3
        assert manager.get_conversation("conv-0") is None  # LRU was evicted
        assert manager.get_conversation("conv-4") is not None  # New one exists

    def test_resume_nonexistent_conversation(self, conversation_manager):
        """Test that resuming a non-existent conversation returns None."""
        result = conversation_manager.resume_conversation("nonexistent-conv")
        assert result is None

    def test_end_nonexistent_conversation(self, conversation_manager):
        """Test that ending a non-existent conversation returns None."""
        result = conversation_manager.end_conversation("nonexistent-conv")
        assert result is None

    def test_disabled_conversation_caching(self, sample_request, sample_kv_blocks):
        """Test that manager does nothing when caching is disabled."""
        manager = ConversationContextManager(enable_conversation_caching=False)

        result = manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        # Should return None when disabled
        assert result is None
        assert len(manager._conversations) == 0

    def test_conversation_statistics(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test getting conversation statistics."""
        # Create some conversations
        for i in range(3):
            conv_id = f"conv-{i}"
            req = Request(
                request_id=f"req-{i}",
                prompt=f"Prompt {i}",
                prompt_token_ids=[1, 2, 3],
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=None,
                eos_token_id=0,
                arrival_time=time.time(),
                lora_request=None,
                conversation_id=conv_id,
                turn_number=0,
                is_conversation_end=False,
            )
            conversation_manager.create_conversation(
                conversation_id=conv_id,
                request=req,
                kv_cache_blocks=sample_kv_blocks,
                num_computed_tokens=3,
                all_token_ids=[1, 2, 3],
                turn_number=0,
            )

        stats = conversation_manager.get_stats()
        assert stats["total_conversations"] == 3
        assert stats["active_conversations"] == 3
        assert stats["suspended_conversations"] == 0

        # Suspend one
        conversation_manager.suspend_conversation("conv-0")
        stats = conversation_manager.get_stats()
        assert stats["active_conversations"] == 2
        assert stats["suspended_conversations"] == 1

    def test_conversation_token_accumulation(
        self, conversation_manager, sample_request, sample_kv_blocks
    ):
        """Test that tokens accumulate correctly across conversation turns."""
        # Create initial conversation with 4 tokens
        conversation_manager.create_conversation(
            conversation_id="conv-123",
            request=sample_request,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=4,
            all_token_ids=[1, 2, 3, 4],
            turn_number=0,
        )

        # Update with turn 1: add tokens 5, 6, 7
        new_request = Request(
            request_id="test-req-2",
            prompt="Turn 1",
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7],
            mm_inputs=None,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=None,
            eos_token_id=0,
            arrival_time=time.time(),
            lora_request=None,
            conversation_id="conv-123",
            turn_number=1,
            is_conversation_end=False,
        )

        updated = conversation_manager.update_conversation(
            conversation_id="conv-123",
            request=new_request,
            turn_number=1,
            kv_cache_blocks=sample_kv_blocks,
            num_computed_tokens=7,
            all_token_ids=[1, 2, 3, 4, 5, 6, 7],
        )

        assert updated.num_computed_tokens == 7
        assert updated.all_token_ids == [1, 2, 3, 4, 5, 6, 7]
        assert updated.turn_number == 1
