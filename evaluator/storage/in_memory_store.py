"""In-memory history store for counterfactual simulation.

This module provides :class:`InMemoryHistoryStore`, a pure-Python history
store that holds :class:`EvaluationRecord` objects in a list — no disk
I/O.  It implements the same protocol surface (``load_all``, ``append``,
``clone``) as :class:`JSONHistoryStore` so it can be used as a drop-in
replacement inside the counterfactual simulation engine.
"""

from __future__ import annotations

import copy

from evaluator.storage.models import EvaluationRecord


class InMemoryHistoryStore:
    """A history store backed entirely by an in-memory list.

    Implements the minimal protocol used by the counterfactual simulator:

    - :meth:`load_all` -> ``list[EvaluationRecord]``
    - :meth:`append` -> ``None``
    - :meth:`clone`  -> new :class:`InMemoryHistoryStore` with deep-copied records

    No filesystem interaction occurs, making this ideal for
    high-throughput scenario simulation where many interventions
    are tested in rapid succession.
    """

    def __init__(self, records: list[EvaluationRecord] | None = None):
        self._records: list[EvaluationRecord] = (
            copy.deepcopy(records) if records else []
        )

    def load_all(self) -> list[EvaluationRecord]:
        """Return all records currently held in memory."""
        return self._records

    def append(self, record: EvaluationRecord) -> None:
        """Append a new record to the in-memory store."""
        self._records.append(copy.deepcopy(record))

    def clone(self) -> InMemoryHistoryStore:
        """Create a deep copy of this store.

        Returns a new :class:`InMemoryHistoryStore` with deep-copied
        records so that mutations to the clone do not affect the
        original.
        """
        return InMemoryHistoryStore(records=copy.deepcopy(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0
