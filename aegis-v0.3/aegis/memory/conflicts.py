from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .models import Memory


class MemoryRelationship(str, Enum):
    """
    Relationship between an existing memory and a candidate memory.
    """

    DUPLICATE = "duplicate"
    RELATED = "related"
    CONFLICT = "conflict"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class ConflictResult:
    """
    Result returned by the deterministic memory relationship detector.

    The detector does not modify either memory.
    """

    relationship: MemoryRelationship
    score: float
    matched_terms: tuple[str, ...]
    reason: str


class MemoryConflictDetector:
    """
    Deterministic first-generation memory relationship detector.

    This component intentionally does not use an LLM.

    It provides predictable and explainable classification for:

        DUPLICATE
        RELATED
        CONFLICT
        UNRELATED

    Later A.E.G.I.S. versions can combine this symbolic detector
    with embeddings and LLM reasoning.
    """

    # =========================================================
    # TEXT CONFIGURATION
    # =========================================================

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "uses",
        "using",
        "with",
    }

    CHANGE_WORDS = {
        "now",
        "currently",
        "changed",
        "change",
        "switched",
        "switch",
        "replaced",
        "replace",
        "updated",
        "update",
        "new",
        "instead",
        "previously",
        "formerly",
        "before",
        "after",
    }

    NEGATION_WORDS = {
        "not",
        "never",
        "no",
        "dont",
        "doesnt",
        "isnt",
        "wasnt",
        "cannot",
        "cant",
        "wont",
        "without",
    }

    # =========================================================
    # PUBLIC API
    # =========================================================

    def compare(
        self,
        existing: Memory,
        candidate: Memory,
    ) -> ConflictResult:
        """
        Compare an existing memory with a candidate memory.

        Classification order:

            1. DUPLICATE
            2. CONFLICT
            3. RELATED
            4. UNRELATED

        The detector never modifies either memory.
        """

        existing_terms = self._terms(
            existing.content
        )

        candidate_terms = self._terms(
            candidate.content
        )

        # -----------------------------------------------------
        # Empty content
        # -----------------------------------------------------

        if not existing_terms or not candidate_terms:
            return ConflictResult(
                relationship=MemoryRelationship.UNRELATED,
                score=0.0,
                matched_terms=(),
                reason=(
                    "One or both memories contain no "
                    "meaningful terms."
                ),
            )

        # -----------------------------------------------------
        # Term overlap
        # -----------------------------------------------------

        matched = sorted(
            existing_terms.intersection(
                candidate_terms
            )
        )

        union = (
            existing_terms
            | candidate_terms
        )

        similarity = (
            len(matched) / len(union)
            if union
            else 0.0
        )

        # -----------------------------------------------------
        # Exact normalized duplicate
        # -----------------------------------------------------

        existing_normalized = self._normalize(
            existing.content
        )

        candidate_normalized = self._normalize(
            candidate.content
        )

        if (
            existing_normalized
            == candidate_normalized
        ):
            return ConflictResult(
                relationship=MemoryRelationship.DUPLICATE,
                score=1.0,
                matched_terms=tuple(matched),
                reason=(
                    "The normalized memory contents "
                    "are identical."
                ),
            )

        # -----------------------------------------------------
        # Detect change indicators
        # -----------------------------------------------------

        existing_change_terms = (
            existing_terms
            & self.CHANGE_WORDS
        )

        candidate_change_terms = (
            candidate_terms
            & self.CHANGE_WORDS
        )

        # -----------------------------------------------------
        # Detect negation
        # -----------------------------------------------------

        existing_negated = bool(
            existing_terms
            & self.NEGATION_WORDS
        )

        candidate_negated = bool(
            candidate_terms
            & self.NEGATION_WORDS
        )

        negation_changed = (
            existing_negated
            != candidate_negated
        )

        # -----------------------------------------------------
        # Conflict detection
        # -----------------------------------------------------
        #
        # A candidate containing explicit change language
        # should be treated as a conflict when it shares a
        # meaningful subject with the existing memory.
        #
        # Example:
        #
        #   Existing:
        #   "AEGIS uses Qwen 7B."
        #
        #   Candidate:
        #   "AEGIS now uses Gemma 4."
        #
        # Shared subject:
        #   AEGIS
        #
        # Change signal:
        #   now
        #
        # Therefore:
        #   CONFLICT
        # -----------------------------------------------------

        if (
            candidate_change_terms
            and matched
        ):
            score = max(
                similarity,
                0.50,
            )

            score = min(
                1.0,
                score + 0.25,
            )

            return ConflictResult(
                relationship=MemoryRelationship.CONFLICT,
                score=score,
                matched_terms=tuple(matched),
                reason=(
                    "The candidate shares a subject with "
                    "the existing memory and contains explicit "
                    "change or replacement language."
                ),
            )

        # -----------------------------------------------------
        # Existing memory already describes a change
        # -----------------------------------------------------
        #
        # Example:
        #
        #   Existing:
        #   "I currently use Gemma 4."
        #
        #   Candidate:
        #   "I use Qwen 7B."
        #
        # This is weaker than an explicit candidate change
        # statement, so only classify it as conflict when
        # there is substantial overlap.
        # -----------------------------------------------------

        if (
            existing_change_terms
            and similarity >= 0.40
        ):
            return ConflictResult(
                relationship=MemoryRelationship.CONFLICT,
                score=min(
                    1.0,
                    similarity + 0.20,
                ),
                matched_terms=tuple(matched),
                reason=(
                    "The existing memory contains change "
                    "language and the candidate substantially "
                    "overlaps with the same subject."
                ),
            )

        # -----------------------------------------------------
        # Negation conflict
        # -----------------------------------------------------

        if (
            negation_changed
            and similarity >= 0.15
        ):
            return ConflictResult(
                relationship=MemoryRelationship.CONFLICT,
                score=min(
                    1.0,
                    similarity + 0.25,
                ),
                matched_terms=tuple(matched),
                reason=(
                    "The memories share a subject but "
                    "differ in negation."
                ),
            )

        # -----------------------------------------------------
        # Strong similarity without contradiction
        # -----------------------------------------------------

        if similarity >= 0.75:
            return ConflictResult(
                relationship=MemoryRelationship.DUPLICATE,
                score=similarity,
                matched_terms=tuple(matched),
                reason=(
                    "The memories contain substantial "
                    "overlapping information without an "
                    "explicit contradiction."
                ),
            )

        # -----------------------------------------------------
        # Moderate similarity
        # -----------------------------------------------------

        if similarity >= 0.40:
            return ConflictResult(
                relationship=MemoryRelationship.RELATED,
                score=similarity,
                matched_terms=tuple(matched),
                reason=(
                    "The memories share a meaningful set "
                    "of terms but do not show an explicit "
                    "contradiction."
                ),
            )

        # -----------------------------------------------------
        # Weak similarity
        # -----------------------------------------------------

        if similarity >= 0.15:
            return ConflictResult(
                relationship=MemoryRelationship.RELATED,
                score=similarity,
                matched_terms=tuple(matched),
                reason=(
                    "The memories share some meaningful "
                    "terms but appear to describe different "
                    "information."
                ),
            )

        # -----------------------------------------------------
        # No meaningful relationship
        # -----------------------------------------------------

        return ConflictResult(
            relationship=MemoryRelationship.UNRELATED,
            score=similarity,
            matched_terms=tuple(matched),
            reason=(
                "The memories do not share enough meaningful "
                "information to establish a relationship."
            ),
        )

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    def _normalize(
        self,
        text: str,
    ) -> str:
        """
        Normalize text for deterministic comparison.
        """

        text = text.lower()

        text = re.sub(
            r"[^\w\s-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _terms(
        self,
        text: str,
    ) -> set[str]:
        """
        Convert text into meaningful normalized terms.
        """

        normalized = self._normalize(
            text
        )

        terms = set(
            normalized.split()
        )

        return {
            term
            for term in terms
            if term
            and term not in self.STOP_WORDS
        }