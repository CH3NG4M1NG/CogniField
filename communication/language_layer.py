"""
communication/language_layer.py
================================
Language Layer — Semantic Message Encoding / Decoding

Sits between raw beliefs and the communication bus. It:

  1. ENCODES structured beliefs into compact semantic messages
     - extracts subject, predicate, value, confidence
     - assigns a semantic token (shared vocabulary)
     - compresses into a canonical form

  2. DECODES incoming messages back into belief updates
     - maps semantic tokens to belief keys
     - adjusts confidence by source trust
     - resolves ambiguous tokens via context

  3. EVOLVES vocabulary over time
     - frequently used phrase → shorter token
     - rare tokens eventually pruned
     - agents that communicate more develop richer shared vocabulary

This gives the system emergent communication efficiency: agents that
interact often compress their common beliefs into high-confidence
shorthand, while rare concepts get verbose descriptions.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .communication_module import Message, MessageType


# ---------------------------------------------------------------------------
# Semantic token
# ---------------------------------------------------------------------------

@dataclass
class SemanticToken:
    """A named concept in the shared vocabulary."""
    token:       str          # short identifier, e.g. "edible_true"
    subject:     str          # e.g. "apple"
    predicate:   str          # e.g. "edible"
    value:       Any          # e.g. True
    usage_count: int   = 0
    confidence:  float = 0.5  # how reliably this token is interpreted
    created_at:  float = field(default_factory=time.time)
    last_used:   float = field(default_factory=time.time)

    def use(self) -> None:
        self.usage_count += 1
        self.last_used = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ---------------------------------------------------------------------------
# Encoded message
# ---------------------------------------------------------------------------

@dataclass
class EncodedMessage:
    """A semantically encoded message ready for transmission."""
    sender_id:   str
    token:       str
    subject:     str
    predicate:   str
    value:       Any
    confidence:  float
    msg_type:    MessageType
    raw_content: str     = ""   # human-readable form
    timestamp:   float   = field(default_factory=time.time)

    def to_message(self) -> Message:
        """Convert back to a CommunicationModule Message."""
        if self.msg_type in (MessageType.BELIEF, MessageType.WARNING):
            return Message.belief_msg(
                self.sender_id, self.subject, self.predicate,
                self.value, self.confidence,
            )
        elif self.msg_type == MessageType.OBSERVATION:
            return Message.observation_msg(
                self.sender_id, self.predicate, self.subject,
                str(self.value), float(self.confidence),
            )
        else:
            m = Message(
                sender_id=self.sender_id, receiver_id=None,
                msg_type=self.msg_type,
                content={
                    "subject":   self.subject,
                    "predicate": self.predicate,
                    "value":     self.value,
                    "token":     self.token,
                },
                confidence=self.confidence,
            )
            return m


# ---------------------------------------------------------------------------
# Language Layer
# ---------------------------------------------------------------------------

class LanguageLayer:
    """
    Semantic encoding/decoding layer for inter-agent communication.

    Parameters
    ----------
    agent_id          : The agent this layer belongs to.
    vocab_max         : Maximum vocabulary size before pruning.
    min_usage_to_keep : Tokens used fewer than this are pruned.
    evolve_threshold  : Usage count above which a token is "established".
    """

    def __init__(
        self,
        agent_id:           str,
        vocab_max:          int   = 500,
        min_usage_to_keep:  int   = 2,
        evolve_threshold:   int   = 5,
    ) -> None:
        self.agent_id          = agent_id
        self.vocab_max         = vocab_max
        self.min_usage         = min_usage_to_keep
        self.evolve_threshold  = evolve_threshold

        # Vocabulary: token → SemanticToken
        self._vocab:    Dict[str, SemanticToken] = {}
        # Reverse index: (subject, predicate) → token
        self._index:    Dict[Tuple[str, str], str] = {}
        # Decode failures log
        self._decode_failures: int = 0
        # Message stats
        self._encoded_count = 0
        self._decoded_count = 0

        # Seed vocabulary with common predicates
        self._bootstrap_vocabulary()

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def _bootstrap_vocabulary(self) -> None:
        """Seed common tokens so agents start with shared primitives."""
        seeds = [
            ("edible",   "true",    True,  "edible_yes"),
            ("edible",   "false",   False, "edible_no"),
            ("fragile",  "true",    True,  "fragile"),
            ("heavy",    "true",    True,  "heavy"),
            ("category", "food",    "food","is_food"),
            ("category", "material","material","is_material"),
            ("outcome",  "success", "success", "succeeded"),
            ("outcome",  "failure", "failure", "failed"),
        ]
        for pred, val_str, val, token in seeds:
            key = (pred, val_str)
            st  = SemanticToken(
                token=token, subject="*", predicate=pred,
                value=val, usage_count=1, confidence=0.9,
            )
            self._vocab[token] = st
            self._index[key]   = token

    def _make_token(self, subject: str, predicate: str, value: Any) -> str:
        """Generate a compact token for a (subject, predicate, value) triple."""
        val_short = str(value).lower()[:8].replace(" ", "_")
        pred_short = predicate[:8]
        subj_short = subject[:6]
        return f"{subj_short}_{pred_short}_{val_short}"

    def register_token(
        self,
        subject:   str,
        predicate: str,
        value:     Any,
        confidence: float = 0.7,
    ) -> SemanticToken:
        """Add or retrieve a token for a belief triple."""
        key = (predicate, str(value).lower())
        if key in self._index:
            token_str = self._index[key]
            st = self._vocab[token_str]
            st.use()
            return st

        # Wildcard match: same predicate+value across any subject
        for existing_token, existing_st in self._vocab.items():
            if (existing_st.predicate == predicate
                    and str(existing_st.value).lower() == str(value).lower()
                    and existing_st.subject == "*"):
                existing_st.use()
                self._index[(predicate, str(value).lower())] = existing_token
                return existing_st

        # Create new token
        token_str = self._make_token(subject, predicate, value)
        st = SemanticToken(
            token=token_str, subject=subject,
            predicate=predicate, value=value,
            confidence=confidence,
        )
        self._vocab[token_str]  = st
        self._index[(predicate, str(value).lower())] = token_str
        self._maybe_prune()
        return st

    def _maybe_prune(self) -> None:
        """Remove low-usage tokens if vocabulary exceeds max size."""
        if len(self._vocab) <= self.vocab_max:
            return
        # Keep: high usage, high confidence, recently used
        def keep_score(st: SemanticToken) -> float:
            return st.usage_count * st.confidence + 1 / (st.age_seconds + 1)

        sorted_tokens = sorted(self._vocab.values(), key=keep_score)
        n_remove = len(self._vocab) - self.vocab_max
        for st in sorted_tokens[:n_remove]:
            if st.usage_count < self.min_usage:
                self._vocab.pop(st.token, None)
                key = (st.predicate, str(st.value).lower())
                self._index.pop(key, None)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        subject:   str,
        predicate: str,
        value:     Any,
        confidence: float = 0.5,
        msg_type:  MessageType = MessageType.BELIEF,
    ) -> EncodedMessage:
        """
        Encode a belief into a semantic message.

        Parameters
        ----------
        subject    : Concept being described (e.g. "apple").
        predicate  : Property (e.g. "edible").
        value      : Value (e.g. True).
        confidence : Sender's confidence.
        msg_type   : Type of message to create.

        Returns
        -------
        EncodedMessage ready for transmission.
        """
        st = self.register_token(subject, predicate, value, confidence)
        raw = f"{subject}.{predicate}={value}({confidence:.2f})"
        self._encoded_count += 1

        return EncodedMessage(
            sender_id=self.agent_id,
            token=st.token,
            subject=subject,
            predicate=predicate,
            value=value,
            confidence=confidence,
            msg_type=msg_type,
            raw_content=raw,
        )

    def encode_from_message(self, msg: Message) -> Optional[EncodedMessage]:
        """Encode an existing Message into its semantic form."""
        subject   = msg.content.get("subject", "")
        predicate = msg.content.get("predicate", "")
        value     = msg.content.get("value")
        if not subject or not predicate or value is None:
            return None
        return self.encode(subject, predicate, value, msg.confidence, msg.msg_type)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        encoded: EncodedMessage,
        sender_trust: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Decode a semantic message into a belief update dict.

        Returns
        -------
        Dict with keys: subject, predicate, value, effective_confidence, token.
        """
        self._decoded_count += 1

        # Look up token for context enrichment
        st = self._vocab.get(encoded.token)
        if st:
            st.use()
            # If token confidence differs significantly, blend
            blended_conf = 0.7 * encoded.confidence + 0.3 * st.confidence
        else:
            blended_conf = encoded.confidence
            self._decode_failures += 1

        # Apply trust discount
        effective_conf = blended_conf * sender_trust

        return {
            "subject":              encoded.subject,
            "predicate":            encoded.predicate,
            "value":                encoded.value,
            "effective_confidence": float(np.clip(effective_conf, 0.0, 0.95)),
            "token":                encoded.token,
            "raw":                  encoded.raw_content,
        }

    def decode_message(
        self,
        msg:          Message,
        sender_trust: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """Decode a raw Message into a belief update dict."""
        encoded = self.encode_from_message(msg)
        if encoded is None:
            return None
        return self.decode(encoded, sender_trust)

    # ------------------------------------------------------------------
    # Vocabulary evolution
    # ------------------------------------------------------------------

    def merge_vocabulary(self, other_vocab: Dict[str, SemanticToken]) -> int:
        """
        Merge another agent's vocabulary into this one.
        High-usage foreign tokens enrich our understanding.
        Returns count of new tokens adopted.
        """
        new_count = 0
        for token_str, st in other_vocab.items():
            if st.usage_count < self.evolve_threshold:
                continue
            if token_str not in self._vocab:
                self._vocab[token_str] = SemanticToken(
                    token=st.token, subject=st.subject,
                    predicate=st.predicate, value=st.value,
                    confidence=st.confidence * 0.8,   # slight discount for foreign
                    usage_count=1,
                )
                key = (st.predicate, str(st.value).lower())
                if key not in self._index:
                    self._index[key] = token_str
                new_count += 1
        return new_count

    def get_shared_tokens(
        self,
        other_vocab: Dict[str, SemanticToken],
    ) -> List[str]:
        """Return tokens that both this agent and another agent know."""
        return [t for t in self._vocab if t in other_vocab]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def vocab_size(self) -> int:
        return len(self._vocab)

    def established_tokens(self) -> List[SemanticToken]:
        """Tokens used enough to be considered established vocabulary."""
        return [st for st in self._vocab.values()
                if st.usage_count >= self.evolve_threshold]

    def summary(self) -> Dict:
        return {
            "agent_id":        self.agent_id,
            "vocab_size":      self.vocab_size(),
            "established":     len(self.established_tokens()),
            "encoded":         self._encoded_count,
            "decoded":         self._decoded_count,
            "decode_failures": self._decode_failures,
        }

    def __repr__(self) -> str:
        return (f"LanguageLayer(agent={self.agent_id}, "
                f"vocab={self.vocab_size()}, "
                f"established={len(self.established_tokens())})")
