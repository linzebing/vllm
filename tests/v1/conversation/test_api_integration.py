# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for multi-turn conversation caching through OpenAI API."""

import pytest

from vllm.entrypoints.openai.protocol import ChatCompletionRequest


class TestConversationAPIProtocol:
    """Test conversation fields in OpenAI API protocol."""

    def test_chat_completion_request_with_conversation_id(self):
        """Test ChatCompletionRequest accepts conversation_id."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            conversation_id="conv-123",
        )

        assert request.conversation_id == "conv-123"
        assert request.end_conversation is False

    def test_chat_completion_request_with_end_conversation(self):
        """Test ChatCompletionRequest accepts end_conversation."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Goodbye"}],
            model="gpt-3.5-turbo",
            conversation_id="conv-123",
            end_conversation=True,
        )

        assert request.conversation_id == "conv-123"
        assert request.end_conversation is True

    def test_chat_completion_request_without_conversation_fields(self):
        """Test ChatCompletionRequest works without conversation fields."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
        )

        assert request.conversation_id is None
        assert request.end_conversation is False

    def test_conversation_id_validation(self):
        """Test that conversation_id must be string if provided."""
        # Valid string conversation_id
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            conversation_id="valid-conv-id",
        )
        assert request.conversation_id == "valid-conv-id"

        # None is also valid (no conversation)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            conversation_id=None,
        )
        assert request.conversation_id is None

    def test_end_conversation_validation(self):
        """Test that end_conversation must be boolean."""
        # True
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            conversation_id="conv-123",
            end_conversation=True,
        )
        assert request.end_conversation is True

        # False
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            conversation_id="conv-123",
            end_conversation=False,
        )
        assert request.end_conversation is False

    def test_conversation_fields_in_request_dict(self):
        """Test that conversation fields appear in request serialization."""
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            conversation_id="conv-123",
            end_conversation=True,
        )

        request_dict = request.model_dump()
        assert "conversation_id" in request_dict
        assert "end_conversation" in request_dict
        assert request_dict["conversation_id"] == "conv-123"
        assert request_dict["end_conversation"] is True


@pytest.mark.skip(reason="Requires running API server")
class TestConversationAPIEndToEnd:
    """
    End-to-end API tests for conversation caching.
    These tests require a running vLLM server and should be run separately.
    """

    def test_multi_turn_conversation_through_api(self):
        """
        Test multi-turn conversation through OpenAI-compatible API.

        This test should:
        1. Start a conversation with conversation_id
        2. Send multiple turns with same conversation_id
        3. Verify response times improve for subsequent turns
        4. End conversation with end_conversation=True
        5. Verify conversation is cleaned up
        """
        # Example API usage (pseudo-code):
        # client = OpenAI(base_url="http://localhost:8000/v1")
        #
        # # Turn 0
        # response0 = client.chat.completions.create(
        #     model="facebook/opt-125m",
        #     messages=[{"role": "user", "content": "Hello"}],
        #     extra_body={"conversation_id": "conv-test-123"}
        # )
        #
        # # Turn 1 (should be faster)
        # response1 = client.chat.completions.create(
        #     model="facebook/opt-125m",
        #     messages=[
        #         {"role": "user", "content": "Hello"},
        #         {"role": "assistant", "content": response0.choices[0].message.content},
        #         {"role": "user", "content": "How are you?"}
        #     ],
        #     extra_body={"conversation_id": "conv-test-123"}
        # )
        #
        # # Turn 2 (final turn)
        # response2 = client.chat.completions.create(
        #     model="facebook/opt-125m",
        #     messages=[...],
        #     extra_body={
        #         "conversation_id": "conv-test-123",
        #         "end_conversation": True
        #     }
        # )
        pass

    def test_conversation_timeout_through_api(self):
        """
        Test that conversations timeout correctly through API.

        Should verify:
        1. Create conversation
        2. Wait longer than timeout
        3. Try to continue conversation
        4. Verify it's treated as new conversation (no speedup)
        """
        pass

    def test_multiple_concurrent_conversations_through_api(self):
        """
        Test handling multiple concurrent conversations through API.

        Should verify:
        1. Create multiple conversations with different IDs
        2. Interleave requests from different conversations
        3. Verify each maintains independent KV cache
        4. End conversations individually
        """
        pass
