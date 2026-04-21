"""
language/structure_checker.py
==============================
Language Structure Checker

Returns a probability score of correctness rather than a binary pass/fail.
Detects:
  - Grammar issues (via heuristic pattern analysis)
  - Semantic mismatch (via latent space geometry)
  - Structural incoherence (word order, missing content)

Design: No external NLP library required.
Uses regex-based heuristics + vector-space geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StructureReport:
    """Assessment of a text string's structural quality."""
    text:             str
    grammar_score:    float   # [0,1] — 1 = well-formed
    semantic_score:   float   # [0,1] — 1 = coherent
    overall_score:    float   # weighted combination
    issues:           List[str]
    suggestions:      List[str]

    @property
    def is_valid(self) -> bool:
        return self.overall_score >= 0.6


class StructureChecker:
    """
    Probabilistic language structure assessment.

    Parameters
    ----------
    grammar_weight  : Weight of grammar score in the overall score.
    semantic_weight : Weight of semantic score in the overall score.
    """

    # Common English stop words
    _STOP = set("the a an is are was were be been being have has had do does did "
                "will would could should may might shall can i you he she it we "
                "they this that these those of in on at to for with from by about".split())

    # Part-of-speech proxies (very rough heuristics)
    _VERB_SUFFIXES   = ("ate", "ize", "ise", "ify", "ing", "ed", "s", "es")
    _NOUN_SUFFIXES   = ("tion", "ness", "ment", "ity", "er", "or", "ist",
                        "ism", "age", "ance", "ence")
    _ADJ_SUFFIXES    = ("ful", "less", "ous", "ive", "al", "ible", "able",
                        "ic", "ical", "ish")

    # Sentence patterns that suggest grammatical issues
    _DOUBLE_DET   = re.compile(r"\b(the|a|an)\s+(the|a|an)\b", re.IGNORECASE)
    _DOUBLE_VERB  = re.compile(r"\b(is|are|was|were)\s+(is|are|was|were)\b",
                                re.IGNORECASE)
    _REPEATED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    _ENDS_ABRUPTLY = re.compile(r"\b(and|or|but|the|a|an|in|on|of)\s*$",
                                 re.IGNORECASE)
    _MISSING_SUBJECT = re.compile(r"^(runs|walks|eats|flies|falls|sits)\b",
                                   re.IGNORECASE)

    def __init__(
        self,
        grammar_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        self.gw = grammar_weight
        self.sw = semantic_weight

    # ------------------------------------------------------------------
    # Grammar checks
    # ------------------------------------------------------------------

    def _grammar_issues(self, text: str) -> Tuple[List[str], float]:
        """
        Heuristic grammar checks. Returns (issues, score ∈ [0,1]).
        """
        issues = []
        penalties = 0.0
        t = text.strip()

        if not t:
            return ["Empty input"], 0.0

        # Double determiner
        if self._DOUBLE_DET.search(t):
            issues.append("Double determiner detected (e.g. 'the the')")
            penalties += 0.25

        # Double auxiliary
        if self._DOUBLE_VERB.search(t):
            issues.append("Double auxiliary verb detected")
            penalties += 0.20

        # Repeated word
        m = self._REPEATED_WORD.search(t)
        if m:
            issues.append(f"Repeated word: '{m.group(1)}'")
            penalties += 0.15

        # Ends abruptly
        if self._ENDS_ABRUPTLY.search(t):
            issues.append("Sentence ends with a function word — likely incomplete")
            penalties += 0.20

        # Missing subject (starts with bare verb)
        if self._MISSING_SUBJECT.match(t):
            issues.append("Sentence may be missing a subject")
            penalties += 0.10

        # Very short (< 2 tokens) — likely incomplete
        words = t.split()
        if len(words) < 2:
            issues.append("Very short input — may be incomplete")
            penalties += 0.15

        # Very long without punctuation — likely run-on
        if len(words) > 30 and not re.search(r"[.!?,;]", t):
            issues.append("Long sentence without punctuation — possible run-on")
            penalties += 0.10

        score = max(0.0, 1.0 - penalties)
        return issues, score

    # ------------------------------------------------------------------
    # Semantic checks (geometry-based)
    # ------------------------------------------------------------------

    def _semantic_issues(
        self,
        text: str,
        vec: Optional[np.ndarray] = None,
        context_vecs: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[str], float]:
        """
        Semantic coherence checks.
        If vec and context_vecs provided: checks geometric coherence.
        Otherwise: falls back to lexical heuristics.
        """
        issues    = []
        penalties = 0.0

        if vec is not None and context_vecs:
            # Geometric coherence: how aligned is vec with context?
            from ..latent_space.frequency_space import FrequencySpace
            sims = [FrequencySpace.similarity(vec, cv) for cv in context_vecs]
            avg_sim = float(np.mean(sims))
            if avg_sim < 0.1:
                issues.append("Vector is far from all context vectors — "
                               "possible semantic mismatch")
                penalties += 0.4
            elif avg_sim < 0.25:
                issues.append("Low semantic alignment with context")
                penalties += 0.2

        else:
            # Lexical heuristics
            words = re.findall(r"\b[a-z]+\b", text.lower())
            content = [w for w in words if w not in self._STOP]

            if not content:
                issues.append("No content words found — likely all function words")
                penalties += 0.4

            # Check for contradictory word pairs
            contradictions = [
                ("hot", "cold"), ("big", "small"), ("alive", "dead"),
                ("full", "empty"), ("open", "closed"),
            ]
            for w1, w2 in contradictions:
                if w1 in content and w2 in content:
                    issues.append(f"Potentially contradictory terms: '{w1}' and '{w2}'")
                    penalties += 0.15

            # Check topic coherence: content words should share a semantic field
            # (Proxy: do any two content words share characters?)
            if len(content) >= 3:
                n_unique = len(set(content))
                if n_unique / len(content) < 0.5:
                    issues.append("Many repeated content words — may be incoherent")
                    penalties += 0.1

        score = max(0.0, 1.0 - penalties)
        return issues, score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        text: str,
        vec: Optional[np.ndarray] = None,
        context_vecs: Optional[List[np.ndarray]] = None,
    ) -> StructureReport:
        """
        Assess the structural quality of a text string.

        Parameters
        ----------
        text         : Input text.
        vec          : Latent vector of *text* (for geometric checks).
        context_vecs : Expected context vectors.

        Returns
        -------
        StructureReport
        """
        g_issues, g_score = self._grammar_issues(text)
        s_issues, s_score = self._semantic_issues(text, vec, context_vecs)

        overall = self.gw * g_score + self.sw * s_score
        all_issues = g_issues + s_issues

        suggestions = self._suggest(all_issues)

        return StructureReport(
            text=text,
            grammar_score=g_score,
            semantic_score=s_score,
            overall_score=overall,
            issues=all_issues,
            suggestions=suggestions,
        )

    def score(self, text: str, **kwargs) -> float:
        """Quick score without full report."""
        return self.check(text, **kwargs).overall_score

    def _suggest(self, issues: List[str]) -> List[str]:
        """Map detected issues to human-readable suggestions."""
        suggestions = []
        for issue in issues:
            if "determiner" in issue:
                suggestions.append("Remove the extra 'the' or 'a'.")
            elif "auxiliary" in issue:
                suggestions.append("Use only one auxiliary verb (is/was/were).")
            elif "Repeated word" in issue:
                suggestions.append("Remove the duplicate word.")
            elif "incomplete" in issue or "abruptly" in issue:
                suggestions.append("Complete the sentence — it seems cut off.")
            elif "missing a subject" in issue:
                suggestions.append("Add a subject before the verb.")
            elif "contradictory" in issue:
                suggestions.append("Check for conflicting descriptors.")
            elif "content words" in issue:
                suggestions.append("Add nouns and verbs to convey meaning.")
            elif "far from all context" in issue:
                suggestions.append("The input may be off-topic relative to context.")
        return suggestions

    def batch_check(self, texts: List[str]) -> List[StructureReport]:
        return [self.check(t) for t in texts]

    def __repr__(self) -> str:
        return (f"StructureChecker(grammar_weight={self.gw}, "
                f"semantic_weight={self.sw})")
