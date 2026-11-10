import streamlit as st
import requests

st.set_page_config(page_title="Assistant universitaire", page_icon="🎓")
st.title("🎓 University Assistant RAG")
st.caption("Prototype académique — réponses fondées sur les documents institutionnels indexés.")
api_url = st.sidebar.text_input("URL API", "http://127.0.0.1:8000")
question = st.text_area("Votre question", placeholder="Quelles sont les consignes d'examen ?")
if st.button("Poser la question", type="primary", disabled=not question.strip()):
    try:
        response = requests.post(f"{api_url}/ask", json={"question": question}, timeout=180)
        response.raise_for_status()
        result = response.json()
        st.write(result["answer"])
        if result.get("refused"):
            st.warning("Réponse refusée : les sources sont insuffisantes.")
        if result.get("sources"):
            st.subheader("Sources récupérées")
            for src in result["sources"]:
                st.write(f"- {src['source']} — chunk {src['chunk_id']} — score {src['score']:.3f}")
    except Exception as exc:
        st.error(f"Erreur API : {exc}")
