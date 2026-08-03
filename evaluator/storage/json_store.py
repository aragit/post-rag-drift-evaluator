from __future__ import annotations

import json
import os
import tempfile

from evaluator.storage.models import EvaluationRecord


class JSONHistoryStore:
    """File-backed history store for :class:`EvaluationRecord` objects.

    The store maintains a JSON Lines file (one JSON object per line)
    so that concurrent appends are safe at the OS level and the file
    can be replayed line-by-line for efficient loading.

    The store is **not** coupled to any database, Streamlit app, or CLI.
    """

    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def save(self, record: EvaluationRecord) -> None:
        """Append a new record to the history file.

        - Creates the file if it does not exist.
        - Preserves all previous records.
        - Uses atomic write (temp file + rename) to avoid corruption.
        """
        data = record.to_dict()
        line = json.dumps(data)

        tmp_path = self._path + ".tmp"
        write_mode = "a" if os.path.exists(self._path) else "w"
        with open(tmp_path, write_mode) as f:
            if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
                # Append mode: we need to ensure newline separation
                with open(self._path) as existing:
                    f.write(existing.read())
            f.write(line + "\n")
        os.replace(tmp_path, self._path)

    def load_all(self) -> list[EvaluationRecord]:
        """Load all records from the history file.

        Returns an empty list if the file does not exist or is empty.
        Malformed lines are skipped with a warning.
        """
        if not os.path.exists(self._path):
            return []

        records: list[EvaluationRecord] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(EvaluationRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    from evaluator.logging_config import get_logger

                    logger = get_logger("JSONHistoryStore")
                    logger.warning("Skipping malformed history line: %s", line[:80])
                    continue
        return records

    def query_by_run(self, run_id: str) -> list[EvaluationRecord]:
        """Return all records associated with a given ``run_id``."""
        return [r for r in self.load_all() if r.run_id == run_id]

    def query_by_metric(self, metric_name: str) -> list[EvaluationRecord]:
        """Return all records whose metrics include the named metric.

        Example::

            store.query_by_metric("js_divergence")
        """
        return [
            r
            for r in self.load_all()
            if any(m.metric_name == metric_name for m in r.metrics)
        ]

    def clone(self, path: str | None = None) -> JSONHistoryStore:
        """Create a copy of this store at a new path.

        The new store contains all records from the original,
        written in a single batch for efficiency.

        Args:
            path: Optional explicit path for the clone.  When
                ``None`` (default) a temporary file is created.

        Returns:
            A new :class:`JSONHistoryStore` pointing to the copied data.
        """
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".jsonl", dir="/tmp/opencode")
            os.close(fd)

        clone_store = JSONHistoryStore(path)
        records = self.load_all()
        lines = [json.dumps(r.to_dict()) for r in records]
        with open(path, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
        return clone_store
