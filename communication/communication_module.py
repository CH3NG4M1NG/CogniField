"""
communication/communication_module.py
=======================================
Inter-Agent Communication System

Agents exchange structured messages on a shared bus. Each message
carries not just content but metadata: sender, confidence, type, and
a timestamp so receivers can weight older messages less.

Message Types
-------------
  BELIEF       – "I believe apple.edible=True with conf=0.87"
  OBSERVATION  – "I just saw the stone get picked up"
  WARNING      – "Do NOT eat the glowing_cube — I got hurt (reward=-0.4)"
  SUGGESTION   – "I think you should inspect the purple_berry first"
  EXPERIMENT   – "I'm testing object X with action Y, join?"
  RESULT       – "Experiment result: X.edible=True"
  CONSENSUS    – "The group has agreed: stone.edible=False"

Communication Flow
------------------
                 Agent A
                    │  send(msg, to=B) or broadcast(msg)
                    ▼
            [Message Bus]
                    │  deliver(msg) to queue
                    ▼
                 Agent B
                    │  receive() → List[Message]
                    │  evaluate_message(msg) → accept/ignore
                    ▼
               Belief Update

Filtering
---------
Agents filter incoming messages by:
  - trust score of sender (low trust → downweight)
  - message age (old messages → downweight)
  - type priority (WARNING > BELIEF > SUGGESTION)
  - redundancy (already know this? → skip)
"""

from __future__ import annotations

import time
import uuid
import enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class MessageType(str, enum.Enum):
    BELIEF       = "belief"
    OBSERVATION  = "observation"
    WARNING      = "warning"
    SUGGESTION   = "suggestion"
    EXPERIMENT   = "experiment"
    RESULT       = "result"
    CONSENSUS    = "consensus"
    QUESTION     = "question"


# Priority of message types (higher = more important)
MESSAGE_PRIORITY: Dict[MessageType, int] = {
    MessageType.WARNING:     5,
    MessageType.CONSENSUS:   4,
    MessageType.RESULT:      3,
    MessageType.BELIEF:      2,
    MessageType.OBSERVATION: 2,
    MessageType.EXPERIMENT:  1,
    MessageType.SUGGESTION:  0,
}


@dataclass
class Message:
    """A single inter-agent message."""
    id:         str
    sender_id:  str
    receiver_id: Optional[str]    # None = broadcast
    msg_type:   MessageType
    content:    Dict[str, Any]    # flexible payload
    confidence: float             # sender's confidence in this message [0,1]
    timestamp:  float = field(default_factory=time.time)
    ttl:        float = 120.0     # time-to-live in seconds

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        return self.age > self.ttl

    @property
    def priority(self) -> int:
        return MESSAGE_PRIORITY.get(self.msg_type, 0)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @staticmethod
    def belief_msg(
        sender_id:   str,
        subject:     str,
        predicate:   str,
        value:       Any,
        confidence:  float,
        receiver_id: Optional[str] = None,
    ) -> "Message":
        return Message(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=receiver_id,
            msg_type=MessageType.BELIEF,
            content={"subject": subject, "predicate": predicate,
                     "value": value, "key": f"{subject}.{predicate}"},
            confidence=confidence,
        )

    @staticmethod
    def warning_msg(
        sender_id:   str,
        subject:     str,
        predicate:   str,
        value:       Any,
        confidence:  float,
        reason:      str = "",
        receiver_id: Optional[str] = None,
    ) -> "Message":
        return Message(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=receiver_id,
            msg_type=MessageType.WARNING,
            content={"subject": subject, "predicate": predicate,
                     "value": value, "key": f"{subject}.{predicate}",
                     "reason": reason},
            confidence=confidence,
        )

    @staticmethod
    def observation_msg(
        sender_id:  str,
        action:     str,
        target:     str,
        outcome:    str,
        reward:     float,
    ) -> "Message":
        return Message(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=None,
            msg_type=MessageType.OBSERVATION,
            content={"action": action, "target": target,
                     "outcome": outcome, "reward": reward},
            confidence=1.0,
        )

    @staticmethod
    def result_msg(
        sender_id:   str,
        experiment:  str,
        subject:     str,
        predicate:   str,
        value:       Any,
        confidence:  float,
    ) -> "Message":
        return Message(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=None,
            msg_type=MessageType.RESULT,
            content={"experiment": experiment, "subject": subject,
                     "predicate": predicate, "value": value,
                     "key": f"{subject}.{predicate}"},
            confidence=confidence,
        )

    @staticmethod
    def question_msg(
        sender_id:   str,
        subject:     str,
        predicate:   str,
        receiver_id: str = None,
    ) -> "Message":
        return Message(
            id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=receiver_id,
            msg_type=MessageType.QUESTION,
            content={"subject": subject, "predicate": predicate},
            confidence=1.0,
        )

    def __repr__(self) -> str:
        return (f"Message({self.msg_type.value}, "
                f"from={self.sender_id}, "
                f"conf={self.confidence:.2f}, "
                f"content={list(self.content.keys())})")


class CommunicationModule:
    """
    Shared message bus for all agents.

    Each agent has its own incoming queue. Messages are routed
    by receiver_id (direct) or broadcast (None receiver → all).

    Parameters
    ----------
    max_queue : Maximum messages per agent queue.
    """

    def __init__(self, max_queue: int = 200) -> None:
        self.max_queue = max_queue
        self._queues:  Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_queue)
        )
        self._all_agents: List[str] = []
        self._history:    List[Message] = []
        self._send_count  = 0
        self._drop_count  = 0

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register(self, agent_id: str) -> None:
        if agent_id not in self._all_agents:
            self._all_agents.append(agent_id)
            if agent_id not in self._queues:
                self._queues[agent_id] = deque(maxlen=self.max_queue)

    def deregister(self, agent_id: str) -> None:
        if agent_id in self._all_agents:
            self._all_agents.remove(agent_id)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send(
        self,
        msg: Message,
        drop_expired: bool = True,
    ) -> bool:
        """
        Route a message to its recipient(s).
        Returns True if delivered to at least one agent.
        """
        if msg.is_expired:
            self._drop_count += 1
            return False

        self._send_count += 1
        self._history.append(msg)

        if msg.receiver_id is None:
            # Broadcast: send to everyone except sender
            delivered = False
            for agent_id in self._all_agents:
                if agent_id != msg.sender_id:
                    self._queues[agent_id].append(msg)
                    delivered = True
            return delivered
        else:
            # Direct message
            if msg.receiver_id in self._queues:
                self._queues[msg.receiver_id].append(msg)
                return True
            return False

    def broadcast(self, msg: Message) -> int:
        """Broadcast a message to all agents. Returns count of recipients."""
        msg.receiver_id = None
        self.send(msg)
        return len(self._all_agents) - 1

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def receive(
        self,
        agent_id:   str,
        max_msgs:   int = 20,
        types:      Optional[List[MessageType]] = None,
        min_conf:   float = 0.0,
        sort_by_priority: bool = True,
    ) -> List[Message]:
        """
        Drain the queue for agent_id and return messages.

        Parameters
        ----------
        max_msgs         : Max messages to return.
        types            : Filter by message type (None = all).
        min_conf         : Minimum confidence threshold.
        sort_by_priority : Return higher-priority messages first.

        Returns
        -------
        List[Message]  — non-expired, filtered, sorted.
        """
        queue   = self._queues.get(agent_id, deque())
        result  = []

        while queue and len(result) < max_msgs:
            msg = queue.popleft()
            if msg.is_expired:
                continue
            if min_conf > 0 and msg.confidence < min_conf:
                continue
            if types and msg.msg_type not in types:
                queue.append(msg)  # put back, not for this filter
                # avoid infinite loop
                if len(result) == 0 and len(queue) == 0:
                    break
                continue
            result.append(msg)

        if sort_by_priority:
            result.sort(key=lambda m: (-m.priority, m.timestamp))

        return result

    def peek(self, agent_id: str) -> int:
        """Return number of pending messages without draining."""
        return len(self._queues.get(agent_id, deque()))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        return {
            "registered_agents": len(self._all_agents),
            "total_sent":        self._send_count,
            "total_dropped":     self._drop_count,
            "queue_sizes":       {a: len(q) for a, q in self._queues.items()},
            "history_size":      len(self._history),
        }

    def message_log(self, last_n: int = 10) -> List[Dict]:
        return [
            {"id": m.id, "type": m.msg_type.value,
             "from": m.sender_id, "to": m.receiver_id or "all",
             "conf": round(m.confidence, 3), "age": round(m.age, 1)}
            for m in self._history[-last_n:]
        ]

    def __repr__(self) -> str:
        return (f"CommunicationModule("
                f"agents={len(self._all_agents)}, "
                f"sent={self._send_count})")
