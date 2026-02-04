import os
import importlib
import pytest
from fastapi.testclient import TestClient


APP_IMPORT = os.getenv("APP_IMPORT", "src.api.app")


@pytest.fixture(scope="session")
def app():
    """
    Charge l'app FastAPI depuis le module configuré.
    """
    os.environ.setdefault("ENV", "test")
    mod = importlib.import_module(APP_IMPORT)

    if not hasattr(mod, "app"):
        raise RuntimeError(
            f"Module '{APP_IMPORT}' ne contient pas d'attribut 'app'. "
            "Ajuste APP_IMPORT (env var ou constant dans conftest.py)."
        )
    return mod.app


@pytest.fixture()
def client(app):
    """
    Client HTTP de test.
    """
    return TestClient(app)
