"""
planning/cooperation_engine.py
================================
Cooperation Engine

Divides goals into sub-tasks and assigns them to the most capable agents.
Think of this as the "team coordination" layer of the multi-agent system.

Cooperation Patterns
--------------------
1. PARALLEL EXPLORATION
   Multiple agents investigate different unknowns simultaneously.
   Agent A: test purple_berry.edible
   Agent B: test glowing_cube.fragile
   → results shared; twice the information in same number of steps

2. SEQUENTIAL PIPELINE
   Agents hand off results:
   Agent A (Explorer): discovers + experiments
   → Agent B (Analyst): validates A's findings
   → Agent C (Planner): uses validated knowledge to plan

3. VOTING CONSENSUS
   All agents act independently on the same question,
   then vote to form a shared belief.

4. SPECIALISED TASK ASSIGNMENT
   Tasks are routed to the agent with the best skill for that task:
   edibility questions → Explorer (lowest risk tolerance = knows most about food)
   safety warnings     → Risk Manager
   multi-step plans    → Planner

Task Assignment Algorithm
--------------------------
1. Score each agent for each task:
   fitness = role_fit × (1 - workload) × belief_confidence × trust

2. Assign to highest-fitness available agent.

3. Track completion; redistribute if agent fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class TaskType(str, Enum):
    EXPLORE      = "explore"
    VERIFY       = "verify"
    WARN         = "warn"
    PLAN         = "plan"
    NEGOTIATE    = "negotiate"
    EXPERIMENT   = "experiment"
    CONSOLIDATE  = "consolidate"


class TaskStatus(str, Enum):
    PENDING    = "pending"
    ASSIGNED   = "assigned"
    IN_PROGRESS= "in_progress"
    COMPLETED  = "completed"
    FAILED     = "failed"


@dataclass
class CoopTask:
    """A task to be distributed to an agent."""
    task_id:     str
    task_type:   TaskType
    description: str
    target:      str            # object/concept this task is about
    priority:    float          = 0.5
    status:      TaskStatus     = TaskStatus.PENDING
    assigned_to: Optional[str]  = None
    result:      Optional[Any]  = None
    created_at:  float          = field(default_factory=time.time)
    completed_at:Optional[float]= None
    retries:     int            = 0
    max_retries: int            = 2

    def assign(self, agent_id: str) -> None:
        self.assigned_to = agent_id
        self.status      = TaskStatus.ASSIGNED

    def complete(self, result: Any = None) -> None:
        self.status       = TaskStatus.COMPLETED
        self.result       = result
        self.completed_at = time.time()

    def fail(self) -> None:
        self.retries += 1
        self.status = TaskStatus.FAILED if self.retries >= self.max_retries else TaskStatus.PENDING
        self.assigned_to = None


@dataclass
class CoopPlan:
    """A cooperative plan: set of tasks with assignments."""
    goal:         str
    tasks:        List[CoopTask]
    pattern:      str          # "parallel" | "pipeline" | "voting"
    created_at:   float = field(default_factory=time.time)

    @property
    def n_pending(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)

    @property
    def n_completed(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        return self.n_completed / len(self.tasks)


# Role-to-task fitness matrix (higher = better fit)
ROLE_TASK_FITNESS: Dict[str, Dict[TaskType, float]] = {
    "explorer":      {TaskType.EXPLORE: 0.9, TaskType.EXPERIMENT: 0.8,
                      TaskType.VERIFY: 0.5,  TaskType.PLAN: 0.3,
                      TaskType.WARN: 0.4,    TaskType.NEGOTIATE: 0.5,
                      TaskType.CONSOLIDATE: 0.3},
    "analyst":       {TaskType.VERIFY: 0.9,  TaskType.EXPERIMENT: 0.7,
                      TaskType.EXPLORE: 0.5, TaskType.PLAN: 0.6,
                      TaskType.WARN: 0.6,    TaskType.NEGOTIATE: 0.7,
                      TaskType.CONSOLIDATE: 0.7},
    "risk_manager":  {TaskType.WARN: 0.9,    TaskType.VERIFY: 0.8,
                      TaskType.EXPERIMENT: 0.5, TaskType.PLAN: 0.4,
                      TaskType.EXPLORE: 0.3, TaskType.NEGOTIATE: 0.6,
                      TaskType.CONSOLIDATE: 0.5},
    "planner":       {TaskType.PLAN: 0.9,    TaskType.NEGOTIATE: 0.7,
                      TaskType.CONSOLIDATE: 0.8, TaskType.VERIFY: 0.6,
                      TaskType.EXPLORE: 0.4, TaskType.EXPERIMENT: 0.4,
                      TaskType.WARN: 0.5},
    "generalist":    {t: 0.5 for t in TaskType},
}


class CooperationEngine:
    """
    Assigns cooperative tasks to the best-suited agents.

    Parameters
    ----------
    agent_registry : {agent_id: {"role": str, "busy": bool, "workload": int}}
    """

    def __init__(self) -> None:
        self._agents:  Dict[str, Dict] = {}   # agent_id → info
        self._tasks:   List[CoopTask]  = []
        self._plans:   List[CoopPlan]  = []
        self._task_counter = 0

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        role:     str,
        capabilities: Optional[Dict[str, float]] = None,
    ) -> None:
        """Register an agent with its role and optional capability scores."""
        self._agents[agent_id] = {
            "role":         role,
            "workload":     0,       # current task count
            "capabilities": capabilities or {},
            "completed":    0,
            "failed":       0,
        }

    def update_workload(self, agent_id: str, delta: int) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["workload"] = max(
                0, self._agents[agent_id]["workload"] + delta
            )

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def create_task(
        self,
        task_type:   TaskType,
        description: str,
        target:      str = "",
        priority:    float = 0.5,
    ) -> CoopTask:
        self._task_counter += 1
        task = CoopTask(
            task_id=f"task_{self._task_counter:04d}",
            task_type=task_type,
            description=description,
            target=target,
            priority=priority,
        )
        self._tasks.append(task)
        return task

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------

    def assign(
        self,
        task:         CoopTask,
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Assign a task to the most suitable available agent.
        Returns assigned agent_id, or None if no agents available.
        """
        if not self._agents:
            return None

        best_agent = None
        best_score = -1.0

        for agent_id, info in self._agents.items():
            if info["workload"] >= 3:
                continue   # too busy

            role    = info["role"]
            fitness = ROLE_TASK_FITNESS.get(role, {}).get(task.task_type, 0.5)
            wl_pen  = info["workload"] * 0.1
            trust   = (trust_scores or {}).get(agent_id, 0.5)
            score   = fitness * trust * (1 - wl_pen)

            if score > best_score:
                best_score = score
                best_agent = agent_id

        if best_agent:
            task.assign(best_agent)
            self.update_workload(best_agent, +1)

        return best_agent

    # ------------------------------------------------------------------
    # Cooperative plans
    # ------------------------------------------------------------------

    def plan_parallel(
        self,
        goal:    str,
        targets: List[str],
        task_type: TaskType = TaskType.EXPLORE,
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> CoopPlan:
        """
        Create a parallel plan: investigate multiple targets simultaneously.
        Each target gets its own task assigned to the best available agent.
        """
        tasks = []
        for target in targets:
            task = self.create_task(
                task_type,
                description=f"{task_type.value}: {target}",
                target=target,
                priority=0.7,
            )
            self.assign(task, trust_scores)
            tasks.append(task)

        plan = CoopPlan(goal=goal, tasks=tasks, pattern="parallel")
        self._plans.append(plan)
        return plan

    def plan_pipeline(
        self,
        goal:      str,
        stages:    List[Tuple[TaskType, str]],   # (type, description)
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> CoopPlan:
        """
        Sequential pipeline: each stage must complete before the next starts.
        Stage assignment happens upfront; execution is caller's responsibility.
        """
        tasks = []
        for i, (ttype, desc) in enumerate(stages):
            task = self.create_task(
                ttype, description=desc,
                priority=1.0 - i * 0.1
            )
            self.assign(task, trust_scores)
            tasks.append(task)

        plan = CoopPlan(goal=goal, tasks=tasks, pattern="pipeline")
        self._plans.append(plan)
        return plan

    def plan_vote(
        self,
        goal:    str,
        target:  str,
        task_type: TaskType = TaskType.VERIFY,
        n_voters:  int = 3,
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> CoopPlan:
        """
        All available agents investigate the same target and vote.
        """
        tasks = []
        agents = list(self._agents.keys())[:n_voters]
        for agent_id in agents:
            task = self.create_task(
                task_type,
                description=f"Vote on {target} (voter={agent_id})",
                target=target,
                priority=0.8,
            )
            task.assign(agent_id)
            self.update_workload(agent_id, +1)
            tasks.append(task)

        plan = CoopPlan(goal=goal, tasks=tasks, pattern="voting")
        self._plans.append(plan)
        return plan

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed."""
        for task in self._tasks:
            if task.task_id == task_id:
                agent_id = task.assigned_to
                task.complete(result)
                if agent_id and agent_id in self._agents:
                    self.update_workload(agent_id, -1)
                    self._agents[agent_id]["completed"] += 1
                return True
        return False

    def fail_task(self, task_id: str) -> bool:
        """Mark a task as failed (may be retried)."""
        for task in self._tasks:
            if task.task_id == task_id:
                agent_id = task.assigned_to
                if agent_id and agent_id in self._agents:
                    self.update_workload(agent_id, -1)
                    self._agents[agent_id]["failed"] += 1
                task.fail()
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_tasks_for(self, agent_id: str) -> List[CoopTask]:
        """Return all assigned-but-not-complete tasks for an agent."""
        return [t for t in self._tasks
                if t.assigned_to == agent_id
                and t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)]

    def redistribute_stuck_tasks(
        self,
        max_age_seconds: float = 60.0,
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Reassign tasks that have been assigned too long without completion."""
        reassigned = []
        now = time.time()
        for task in self._tasks:
            if task.status != TaskStatus.ASSIGNED:
                continue
            age = now - task.created_at
            if age > max_age_seconds:
                old_agent = task.assigned_to
                if old_agent and old_agent in self._agents:
                    self.update_workload(old_agent, -1)
                task.status      = TaskStatus.PENDING
                task.assigned_to = None
                new_agent = self.assign(task, trust_scores)
                if new_agent:
                    reassigned.append(task.task_id)
        return reassigned

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        n_tasks    = len(self._tasks)
        n_complete = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        n_failed   = sum(1 for t in self._tasks if t.status == TaskStatus.FAILED)
        return {
            "agents":     len(self._agents),
            "plans":      len(self._plans),
            "tasks":      n_tasks,
            "completed":  n_complete,
            "failed":     n_failed,
            "completion_rate": round(n_complete / max(n_tasks, 1), 3),
            "agent_loads": {aid: info["workload"]
                            for aid, info in self._agents.items()},
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"CooperationEngine(agents={s['agents']}, "
                f"tasks={s['tasks']}, "
                f"completion={s['completion_rate']:.0%})")
