import os
import importlib
import sqlite3
import tempfile
from pathlib import Path
import inspect

STORE_IMPORT = os.getenv("STORE_IMPORT", "src.api.feedback_store")


def _load_store():
    return importlib.import_module(STORE_IMPORT)


def _find_insert_func(store):
    """
    Trouve une fonction "d'insertion feedback" dans le module.
    On accepte différents noms possibles (robuste si tu as renommé).
    """
    # Priorité aux noms explicites
    preferred = [
        "insert_feedback",
        "save_feedback",
        "store_feedback",
        "log_feedback",
        "add_feedback",
        "write_feedback",
        "record_feedback",
    ]
    for name in preferred:
        fn = getattr(store, name, None)
        if callable(fn):
            return fn

    # Fallback: n'importe quelle fonction qui contient 'feedback' et n'est pas init/ensure
    candidates = []
    for name in dir(store):
        if "feedback" not in name.lower():
            continue
        if any(x in name.lower() for x in ("init", "ensure", "create", "setup")):
            continue
        fn = getattr(store, name, None)
        if callable(fn):
            candidates.append((name, fn))

    if candidates:
        # prend la première (ordre alpha stable)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return None


def test_feedback_store_insert_and_readback():
    store = _load_store()

    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "feedback_test.db"
    os.environ["FEEDBACK_DB_PATH"] = str(db_path)

    # 1) init db
    if hasattr(store, "init_db") and callable(store.init_db):
        store.init_db()
    elif hasattr(store, "ensure_db") and callable(store.ensure_db):
        store.ensure_db()
    else:
        raise RuntimeError(f"Le module {STORE_IMPORT} doit exposer init_db() ou ensure_db().")

    # 2) insert feedback (fonction à trouver)
    insert_fn = _find_insert_func(store)
    assert insert_fn is not None, (
        f"Aucune fonction d'insertion feedback trouvée dans {STORE_IMPORT}. "
        "Attendu: insert_feedback/save_feedback/store_feedback/..."
    )

    feedback = {
        "text": "i hate delays",
        "predicted_label": "negative",
        "proba_negative": 0.91,
        "is_correct": False,
        "true_label": "not_negative",
    }

    # Appel tolérant: si la fonction n'accepte pas tous les kwargs, on filtre
    sig = inspect.signature(insert_fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in feedback.items() if k in accepted or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())}
    insert_fn(**filtered)

    # 3) read back (direct sqlite)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cur.fetchall()]
    assert tables, "Aucune table trouvée dans la DB de feedback."

    table = tables[0]
    cur.execute(f"SELECT * FROM {table} LIMIT 1;")
    row = cur.fetchone()
    con.close()

    assert row is not None, "Aucune ligne insérée en base."
