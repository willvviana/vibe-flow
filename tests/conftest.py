import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def isolate_memory_preflight(request, monkeypatch):
    """Unit tests with a mocked model must not depend on how much RAM the machine has right now.
    Real-model (smoke) tests keep the true preflight."""
    if "smoke" not in request.node.name:
        from engine import memory
        monkeypatch.setattr(memory, "preflight", lambda *a, **k: [])
