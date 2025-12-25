"""Conversation history management and relevance filtering."""

from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter
import re
from .config import HISTORY_RECENCY_WINDOW, HISTORY_MAX_EXCHANGES, HISTORY_KEYWORD_THRESHOLD


@dataclass
class ConversationExchange:
    """Represents a single Q&A exchange."""
    user_query: str
    assistant_response: str  # Stage 3 final answer


def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate keyword-based similarity using simple word overlap.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0 and 1
    """
    # Normalize: lowercase and split into words
    words1 = re.findall(r'\w+', text1.lower())
    words2 = re.findall(r'\w+', text2.lower())

    if not words1 or not words2:
        return 0.0

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'would',
                  'should', 'will', 'shall', 'may', 'might', 'must', 'that', 'this',
                  'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}

    words1_filtered = [w for w in words1 if w not in stop_words]
    words2_filtered = [w for w in words2 if w not in stop_words]

    if not words1_filtered or not words2_filtered:
        return 0.0

    # Calculate word frequency
    counter1 = Counter(words1_filtered)
    counter2 = Counter(words2_filtered)

    # Calculate overlap (Jaccard-like similarity with frequency weighting)
    common_words = set(counter1.keys()) & set(counter2.keys())

    if not common_words:
        return 0.0

    # Weight by frequency
    overlap_score = sum(min(counter1[word], counter2[word]) for word in common_words)
    total_words = sum(counter1.values()) + sum(counter2.values())

    return (2 * overlap_score) / total_words if total_words > 0 else 0.0


def select_relevant_history(
    conversation_messages: List[Dict[str, Any]],
    current_query: str,
    recency_window: int = HISTORY_RECENCY_WINDOW,
    max_history_exchanges: int = HISTORY_MAX_EXCHANGES,
    keyword_threshold: float = HISTORY_KEYWORD_THRESHOLD
) -> List[ConversationExchange]:
    """
    Select relevant conversation history for the current query.

    Uses a two-tiered approach:
    1. Always include last N exchanges (recency-based)
    2. Include older exchanges with high keyword similarity (relevance-based)

    Args:
        conversation_messages: Full message history from storage
        current_query: The current user question
        recency_window: Always include last N exchanges
        max_history_exchanges: Maximum exchanges to include
        keyword_threshold: Minimum keyword overlap score (0-1)

    Returns:
        List of relevant exchanges, ordered chronologically
    """
    # Parse messages into exchanges (user + assistant pairs)
    exchanges = []
    i = 0
    while i < len(conversation_messages):
        msg = conversation_messages[i]

        # Look for user message
        if msg.get('role') == 'user':
            user_query = msg.get('content', '')

            # Look for following assistant message
            if i + 1 < len(conversation_messages):
                next_msg = conversation_messages[i + 1]
                if next_msg.get('role') == 'assistant':
                    # Extract Stage 3 final answer
                    stage3 = next_msg.get('stage3', {})
                    assistant_response = stage3.get('content', '') if isinstance(stage3, dict) else str(stage3)

                    exchanges.append(ConversationExchange(
                        user_query=user_query,
                        assistant_response=assistant_response
                    ))
                    i += 2  # Skip both messages
                    continue

        i += 1

    if not exchanges:
        return []

    # Tier 1: Always include recent exchanges
    recent_count = min(recency_window, len(exchanges))
    recent_exchanges = exchanges[-recent_count:]
    selected_indices = set(range(len(exchanges) - recent_count, len(exchanges)))

    # Tier 2: Add relevant older exchanges based on keyword similarity
    if len(exchanges) > recent_count and len(selected_indices) < max_history_exchanges:
        older_exchanges = exchanges[:-recent_count]

        # Calculate similarity scores for older exchanges
        scored_exchanges = []
        for idx, exchange in enumerate(older_exchanges):
            # Compare current query with both old query and old response
            query_sim = calculate_keyword_similarity(current_query, exchange.user_query)
            response_sim = calculate_keyword_similarity(current_query, exchange.assistant_response)
            max_sim = max(query_sim, response_sim)

            if max_sim >= keyword_threshold:
                scored_exchanges.append((idx, max_sim, exchange))

        # Sort by similarity (highest first)
        scored_exchanges.sort(key=lambda x: x[1], reverse=True)

        # Add top scoring exchanges up to max limit
        remaining_slots = max_history_exchanges - len(selected_indices)
        for idx, sim, exchange in scored_exchanges[:remaining_slots]:
            selected_indices.add(idx)

    # Collect all selected exchanges in chronological order
    selected_exchanges = [exchanges[i] for i in sorted(selected_indices)]

    return selected_exchanges


def truncate_response(text: str, max_tokens: int = 500) -> str:
    """
    Truncate response to approximate token limit.

    Uses rough estimate: 1 token ≈ 4 characters

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text with ellipsis if needed
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def format_history_for_stage1(
    history: List[ConversationExchange],
    current_query: str
) -> str:
    """
    Format history for Stage 1 prompts (individual model responses).

    Provides full conversational context.

    Args:
        history: List of relevant exchanges
        current_query: Current user question (for context, not included)

    Returns:
        Formatted history string
    """
    if not history:
        return ""

    parts = ["[Previous conversation for context]\n"]

    for i, exchange in enumerate(history, 1):
        # Truncate long responses to save tokens
        response = truncate_response(exchange.assistant_response, max_tokens=400)

        parts.append(f"\nUser: {exchange.user_query}")
        parts.append(f"Assistant: {response}")

    parts.append("\n[End of previous conversation]\n")
    parts.append("\nPlease answer the following question, keeping the conversation context in mind:")

    return "\n".join(parts)


def format_history_for_stage2(
    history: List[ConversationExchange],
    current_query: str
) -> str:
    """
    Format history for Stage 2 prompts (peer ranking).

    Provides condensed context focusing on the conversation flow.

    Args:
        history: List of relevant exchanges
        current_query: Current user question

    Returns:
        Formatted history string
    """
    if not history:
        return ""

    parts = ["Context from previous conversation:\n"]

    for exchange in history:
        # More aggressive truncation for ranking stage
        response = truncate_response(exchange.assistant_response, max_tokens=200)
        parts.append(f"Q: {exchange.user_query}")
        parts.append(f"A: {response}\n")

    return "\n".join(parts)


def format_history_for_stage3(
    history: List[ConversationExchange],
    current_query: str
) -> str:
    """
    Format history for Stage 3 prompt (chairman synthesis).

    Provides comprehensive context for final answer synthesis.

    Args:
        history: List of relevant exchanges
        current_query: Current user question

    Returns:
        Formatted history string
    """
    if not history:
        return ""

    parts = ["Conversation history:\n"]

    for exchange in history:
        # Moderate truncation - chairman needs more context
        response = truncate_response(exchange.assistant_response, max_tokens=350)
        parts.append(f"\nUser previously asked: {exchange.user_query}")
        parts.append(f"Council answered: {response}")

    parts.append("\n---\n")

    return "\n".join(parts)
