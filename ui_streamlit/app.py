import os
import requests
import streamlit as st

st.set_page_config(page_title="Tweet Sentiment", page_icon="💬", layout="centered")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"
FEEDBACK_URL = f"{API_BASE_URL}/feedback"

st.title("💬 Tweet Sentiment")
st.caption("Prédiction + feedback (correct / incorrect).")

with st.sidebar:
    st.header("⚙️ Config")
    st.text_input("API base URL", value=API_BASE_URL, key="api_url")
    st.code(f"{st.session_state.api_url.rstrip('/')}/predict", language="text")
    st.code(f"{st.session_state.api_url.rstrip('/')}/feedback", language="text")

text = st.text_area("Texte", placeholder="Ex: i hate this airline", height=120)

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None  # {"text":..., "label":..., "proba_negative":...}

if st.button("Prédire", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Mets un texte avant de prédire.")
        st.stop()

    try:
        with st.spinner("Appel API..."):
            r = requests.post(
                f"{st.session_state.api_url.rstrip('/')}/predict",
                json={"text": text.strip()},
                timeout=30,
            )
        if r.status_code != 200:
            st.error(f"Erreur API ({r.status_code})")
            st.code(r.text)
            st.stop()

        data = r.json()
        st.session_state.last_pred = {
            "text": text.strip(),
            "label": data.get("label"),
            "proba_negative": data.get("proba_negative"),
        }

    except Exception as e:
        st.error("Erreur lors de l'appel à /predict")
        st.exception(e)
        st.stop()

# Affichage prédiction + feedback
pred = st.session_state.last_pred
if pred:
    st.success("✅ Prédiction reçue")
    c1, c2 = st.columns(2)
    c1.metric("Label", pred["label"])
    if pred["proba_negative"] is not None:
        c2.metric("Proba négatif", f"{float(pred['proba_negative']):.3f}")
        st.progress(min(max(float(pred["proba_negative"]), 0.0), 1.0))

    st.markdown("### Feedback")
    st.write("La prédiction est-elle correcte ?")

    fb1, fb2 = st.columns(2)

    def send_feedback(is_correct: bool):
        payload = {
            "text": pred["text"],
            "predicted_label": pred["label"],
            "proba_negative": pred["proba_negative"],
            "is_correct": is_correct,
            # optionnel: si tu veux capturer le vrai label quand c'est faux
            # "true_label": "negative" ou "not_negative"
        }
        r = requests.post(
            f"{st.session_state.api_url.rstrip('/')}/feedback",
            json=payload,
            timeout=30,
        )
        return r

    with fb1:
        if st.button("✅ Correct", use_container_width=True):
            try:
                r = send_feedback(True)
                if r.status_code == 200:
                    st.toast("Feedback envoyé (correct).", icon="✅")
                else:
                    st.error(f"Erreur /feedback ({r.status_code})")
                    st.code(r.text)
            except Exception as e:
                st.error("Erreur en envoyant le feedback.")
                st.exception(e)

    with fb2:
        if st.button("❌ Incorrect", use_container_width=True):
            try:
                r = send_feedback(False)
                if r.status_code == 200:
                    st.toast("Feedback envoyé (incorrect).", icon="⚠️")
                else:
                    st.error(f"Erreur /feedback ({r.status_code})")
                    st.code(r.text)
            except Exception as e:
                st.error("Erreur en envoyant le feedback.")
                st.exception(e)
