# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversation context management for multi-turn conversations with persistent KV cache."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ConversationState:
    """State of an active conversation.

    This tracks the KV cache blocks and request state for a conversation
    that spans multiple turns, enabling incremental decoding without
    recomputing previous turns.
    """

    conversation_id: str
    # The request object from the most recent turn
    last_request: "Request"
    # KV cache blocks allocated for this conversation
    kv_cache_blocks: "KVCacheBlocks"
    # Number of tokens computed so far (across all turns)
    num_computed_tokens: int
    # All token IDs across all turns
    all_token_ids: list[int]
    # Turn number (incremented with each new request)
    turn_number: int
    # Timestamp of last activity (for cleanup)
    last_activity_time: float = field(default_factory=time.time)
    # Whether conversation is currently being processed
    is_active: bool = False


class ConversationContextManager:
    """Manages conversation contexts for multi-turn conversations.

    This enables incremental decoding by maintaining KV cache blocks
    across conversation turns, avoiding the need to recompute previous
    context on each turn.
    """

    def __init__(
        self,
        enable_conversation_caching: bool = True,
        conversation_timeout_seconds: float = 300.0,
        max_active_conversations: int | None = None,
    ):
        """Initialize the conversation context manager.

        Args:
            enable_conversation_caching: Whether to enable conversation-level
                KV cache persistence.
            conversation_timeout_seconds: Time in seconds after which inactive
                conversations are cleaned up.
            max_active_conversations: Maximum number of active conversations.
                If exceeded, oldest conversations are cleaned up. None for unlimited.
        """
        self.enable_conversation_caching = enable_conversation_caching
        self.conversation_timeout_seconds = conversation_timeout_seconds
        self.max_active_conversations = max_active_conversations

        # Active conversations: conversation_id -> ConversationState
        self._conversations: dict[str, ConversationState] = {}

        # Track conversation creation order for LRU cleanup
        self._conversation_order: list[str] = []

    def is_conversation_active(self, conversation_id: str) -> bool:
        """Check if a conversation is currently active."""
        return conversation_id in self._conversations

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        """Get the state of an active conversation."""
        if conversation_id in self._conversations:
            conv_state = self._conversations[conversation_id]
            # Update last activity time
            conv_state.last_activity_time = time.time()
            return conv_state
        return None

    def create_conversation(
        self,
        conversation_id: str,
        request: "Request",
        kv_cache_blocks: "KVCacheBlocks",
        num_computed_tokens: int,
        all_token_ids: list[int],
    ) -> ConversationState:
        """Create a new conversation context.

        Args:
            conversation_id: Unique identifier for the conversation.
            request: The initial request object.
            kv_cache_blocks: KV cache blocks allocated for this request.
            num_computed_tokens: Number of tokens computed.
            all_token_ids: All token IDs (prompt + generated tokens).

        Returns:
            The newly created ConversationState.
        """
        if not self.enable_conversation_caching:
            raise ValueError(
                "Conversation caching is disabled. "
                "Enable it with enable_conversation_caching=True"
            )

        # Clean up old conversations if needed
        self._maybe_cleanup_conversations()

        conv_state = ConversationState(
            conversation_id=conversation_id,
            last_request=request,
            kv_cache_blocks=kv_cache_blocks,
            num_computed_tokens=num_computed_tokens,
            all_token_ids=all_token_ids.copy(),
            turn_number=1,
            last_activity_time=time.time(),
            is_active=True,
        )

        self._conversations[conversation_id] = conv_state
        self._conversation_order.append(conversation_id)

        logger.debug(
            f"Created conversation context {conversation_id} "
            f"with {num_computed_tokens} tokens"
        )

        return conv_state

    def update_conversation(
        self,
        conversation_id: str,
        request: "Request",
        kv_cache_blocks: "KVCacheBlocks",
        num_computed_tokens: int,
        all_token_ids: list[int],
    ) -> ConversationState:
        """Update an existing conversation with a new turn.

        Args:
            conversation_id: The conversation to update.
            request: The new request object.
            kv_cache_blocks: Updated KV cache blocks.
            num_computed_tokens: Updated number of computed tokens.
            all_token_ids: Updated token IDs.

        Returns:
            The updated ConversationState.
        """
        if conversation_id not in self._conversations:
            raise ValueError(
                f"Conversation {conversation_id} does not exist. "
                "Create it first with create_conversation()"
            )

        conv_state = self._conversations[conversation_id]
        conv_state.last_request = request
        conv_state.kv_cache_blocks = kv_cache_blocks
        conv_state.num_computed_tokens = num_computed_tokens
        conv_state.all_token_ids = all_token_ids.copy()
        conv_state.turn_number += 1
        conv_state.last_activity_time = time.time()
        conv_state.is_active = True

        logger.debug(
            f"Updated conversation {conversation_id} to turn {conv_state.turn_number} "
            f"with {num_computed_tokens} total tokens"
        )

        return conv_state

    def suspend_conversation(self, conversation_id: str) -> None:
        """Mark a conversation as suspended (not actively being processed).

        The KV cache blocks remain allocated but the conversation is not
        actively running. This allows other requests to be scheduled while
        keeping the conversation context alive for future turns.
        """
        if conversation_id in self._conversations:
            self._conversations[conversation_id].is_active = False
            logger.debug(f"Suspended conversation {conversation_id}")

    def resume_conversation(self, conversation_id: str) -> ConversationState | None:
        """Resume a suspended conversation for a new turn.

        Returns:
            The ConversationState if found, None otherwise.
        """
        if conversation_id in self._conversations:
            conv_state = self._conversations[conversation_id]
            conv_state.is_active = True
            conv_state.last_activity_time = time.time()
            logger.debug(f"Resumed conversation {conversation_id}")
            return conv_state
        return None

    def end_conversation(self, conversation_id: str) -> ConversationState | None:
        """Explicitly end a conversation and return its state for cleanup.

        Returns:
            The ConversationState if found (for cleanup), None otherwise.
        """
        if conversation_id in self._conversations:
            conv_state = self._conversations.pop(conversation_id)
            if conversation_id in self._conversation_order:
                self._conversation_order.remove(conversation_id)
            logger.debug(
                f"Ended conversation {conversation_id} after "
                f"{conv_state.turn_number} turns"
            )
            return conv_state
        return None

    def cleanup_timed_out_conversations(self) -> list[ConversationState]:
        """Clean up conversations that have timed out.

        Returns:
            List of ConversationStates that were cleaned up (for KV cache freeing).
        """
        if not self.enable_conversation_caching:
            return []

        current_time = time.time()
        timed_out_convs = []

        for conv_id in list(self._conversations.keys()):
            conv_state = self._conversations[conv_id]
            time_since_activity = current_time - conv_state.last_activity_time

            if time_since_activity > self.conversation_timeout_seconds:
                logger.info(
                    f"Cleaning up conversation {conv_id} due to timeout "
                    f"({time_since_activity:.1f}s > "
                    f"{self.conversation_timeout_seconds}s)"
                )
                removed_state = self.end_conversation(conv_id)
                if removed_state:
                    timed_out_convs.append(removed_state)

        return timed_out_convs

    def _maybe_cleanup_conversations(self) -> None:
        """Clean up old conversations if we've exceeded the maximum."""
        if (
            self.max_active_conversations is not None
            and len(self._conversations) >= self.max_active_conversations
        ):
            # Remove oldest conversation (LRU)
            if self._conversation_order:
                oldest_conv_id = self._conversation_order[0]
                logger.info(
                    f"Removing conversation {oldest_conv_id} to make room "
                    f"(max_active_conversations={self.max_active_conversations})"
                )
                # Note: The caller should handle freeing KV cache blocks
                self.end_conversation(oldest_conv_id)

    def get_num_active_conversations(self) -> int:
        """Get the number of currently active conversations."""
        return len(self._conversations)

    def get_conversation_stats(self) -> dict:
        """Get statistics about active conversations."""
        if not self._conversations:
            return {
                "num_conversations": 0,
                "avg_turns": 0.0,
                "avg_tokens": 0.0,
                "max_turns": 0,
                "max_tokens": 0,
            }

        turns = [conv.turn_number for conv in self._conversations.values()]
        tokens = [conv.num_computed_tokens for conv in self._conversations.values()]

        return {
            "num_conversations": len(self._conversations),
            "avg_turns": sum(turns) / len(turns),
            "avg_tokens": sum(tokens) / len(tokens),
            "max_turns": max(turns),
            "max_tokens": max(tokens),
        }
