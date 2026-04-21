"""
core/event_bus.py
==================
Event Bus — Publish / Subscribe System

Agents and subsystems fire events; interested parties react instantly
without being tightly coupled to the source.

Event Types
-----------
BELIEF_UPDATED    – a belief changed (subject, predicate, value, confidence)
CONSENSUS_REACHED – global consensus formed on a key
CONFLICT_DETECTED – two agents hold contradictory beliefs
GOAL_COMPLETED    – an agent finished a goal
GOAL_FAILED       – an agent failed a goal
TASK_ASSIGNED     – a cooperative task was assigned
TASK_COMPLETED    – a cooperative task finished
AGENT_JOINED      – a new agent joined the system
AGENT_LEFT        – an agent left the system
EXPERIMENT_DONE   – an experiment produced a result
WARNING_ISSUED    – a danger was detected
KNOWLEDGE_SHARED  – a belief was written to shared memory
ROLE_CHANGED      – an agent's role evolved

Delivery
--------
Synchronous fan-out: all subscribers for an event type are called
immediately when the event fires.  If a subscriber raises, it is
caught and logged; other subscribers still receive the event.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    BELIEF_UPDATED    = "belief_updated"
    CONSENSUS_REACHED = "consensus_reached"
    CONFLICT_DETECTED = "conflict_detected"
    GOAL_COMPLETED    = "goal_completed"
    GOAL_FAILED       = "goal_failed"
    TASK_ASSIGNED     = "task_assigned"
    TASK_COMPLETED    = "task_completed"
    AGENT_JOINED      = "agent_joined"
    AGENT_LEFT        = "agent_left"
    EXPERIMENT_DONE   = "experiment_done"
    WARNING_ISSUED    = "warning_issued"
    KNOWLEDGE_SHARED  = "knowledge_shared"
    ROLE_CHANGED      = "role_changed"
    ROUND_COMPLETE    = "round_complete"


@dataclass
class Event:
    """An event fired on the bus."""
    event_type: EventType
    source:     str           # who fired the event
    payload:    Dict[str, Any]= field(default_factory=dict)
    timestamp:  float         = field(default_factory=time.time)
    event_id:   int           = 0

    def __repr__(self) -> str:
        return (f"Event({self.event_type.value}, "
                f"src={self.source}, "
                f"keys={list(self.payload.keys())})")


# Handler signature: (Event) -> None
Handler = Callable[[Event], None]


class EventBus:
    """
    Central publish/subscribe bus.

    Usage
    -----
    # Subscribe
    bus.subscribe(EventType.BELIEF_UPDATED, my_handler)
    bus.subscribe_all(agent.on_any_event)

    # Publish
    bus.publish(Event(EventType.BELIEF_UPDATED, "agent_A",
                      {"key": "apple.edible", "value": True}))
    """

    def __init__(self, max_log: int = 5_000) -> None:
        self._handlers:    Dict[EventType, List[Handler]] = defaultdict(list)
        self._all_handlers: List[Handler] = []
        self._event_log:   List[Event]    = []
        self._error_log:   List[str]      = []
        self._max_log      = max_log
        self._counter      = 0
        self._counts:      Dict[EventType, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Subscribe handler to a specific event type."""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe handler to ALL event types."""
        if handler not in self._all_handlers:
            self._all_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a specific subscription."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def unsubscribe_all(self, handler: Handler) -> None:
        if handler in self._all_handlers:
            self._all_handlers.remove(handler)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> int:
        """
        Fire an event. Delivers to all matching + all-handlers.
        Returns number of handlers called.
        """
        self._counter += 1
        event.event_id = self._counter
        self._counts[event.event_type] += 1

        if len(self._event_log) < self._max_log:
            self._event_log.append(event)

        called = 0

        # Type-specific handlers
        for handler in list(self._handlers.get(event.event_type, [])):
            try:
                handler(event)
                called += 1
            except Exception as e:
                self._error_log.append(
                    f"Handler {handler.__name__} for {event.event_type}: "
                    f"{type(e).__name__}: {e}"
                )

        # All-event handlers
        for handler in list(self._all_handlers):
            try:
                handler(event)
                called += 1
            except Exception as e:
                self._error_log.append(
                    f"All-handler {handler.__name__}: {e}"
                )

        return called

    def fire(
        self,
        event_type: EventType,
        source:     str,
        **payload: Any,
    ) -> int:
        """Convenience: fire an event by type + keyword payload."""
        return self.publish(Event(event_type, source, dict(payload)))

    # ------------------------------------------------------------------
    # Querying the log
    # ------------------------------------------------------------------

    def recent_events(self, n: int = 20) -> List[Event]:
        return self._event_log[-n:]

    def events_of_type(self, event_type: EventType) -> List[Event]:
        return [e for e in self._event_log if e.event_type == event_type]

    def events_from(self, source: str) -> List[Event]:
        return [e for e in self._event_log if e.source == source]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "total_events":   self._counter,
            "by_type":        {k.value: v for k, v in self._counts.items()},
            "handlers":       {k.value: len(v) for k, v in self._handlers.items()},
            "all_handlers":   len(self._all_handlers),
            "errors":         len(self._error_log),
            "log_size":       len(self._event_log),
        }

    def __repr__(self) -> str:
        return (f"EventBus(events={self._counter}, "
                f"types={len(self._counts)}, "
                f"errors={len(self._error_log)})")
