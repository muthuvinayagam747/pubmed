import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="PubMed NLP Question Answering",
    page_icon="🧬"
)

st.title("🧬 PubMed Analysis Using NLP")
st.subheader("Question Answering System")


# -----------------------------------------
# PUBMED SAMPLE DATASET
# -----------------------------------------

data = {
    "Title": [
        "Artificial Intelligence in Cancer Detection",
        "Machine Learning for Diabetes Prediction",
        "Deep Learning in Medical Imaging",
        "NLP Applications in Healthcare"
    ],

    "Abstract": [
        "Artificial intelligence is increasingly used for cancer detection. "
        "Machine learning models can analyze medical images and identify "
        "abnormal tissue. Deep learning methods can improve cancer detection "
        "accuracy.",

        "Machine learning techniques are widely used for diabetes prediction. "
        "Patient age, glucose level, body mass index and family history are "
        "important factors. Predictive models can help identify patients at risk.",

        "Deep learning has become an important technology in medical imaging. "
        "Convolutional neural networks can analyze X-ray, CT and MRI images. "
        "These models can support doctors in detecting abnormalities.",

        "Natural Language Processing is used to analyze biomedical text. "
        "NLP can extract useful information from clinical documents and "
        "research articles. Question answering systems can help researchers "
        "retrieve relevant information quickly."
    ]
}

df = pd.DataFrame(data)


# -----------------------------------------
# DISPLAY DATASET
# -----------------------------------------

st.write("### PubMed Research Articles")

st.dataframe(df)


# -----------------------------------------
# QUESTION INPUT
# -----------------------------------------

question = st.text_input(
    "Enter your biomedical question:",
    placeholder="Example: What is NLP used for?"
)


# -----------------------------------------
# QUESTION ANSWERING
# -----------------------------------------

if st.button("Find Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        best_answer = ""
        best_title = ""
        best_score = 0

        for index, row in df.iterrows():

            # Split abstract into sentences
            sentences = re.split(r'(?<=[.!?])\s+', row["Abstract"])

            documents = [question] + sentences

            # TF-IDF conversion
            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            tfidf_matrix = vectorizer.fit_transform(documents)

            # Similarity between question and sentences
            similarity = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:]
            )[0]

            max_index = similarity.argmax()

            if similarity[max_index] > best_score:

                best_score = similarity[max_index]
                best_answer = sentences[max_index]
                best_title = row["Title"]


        # ---------------------------------
        # DISPLAY RESULT
        # ---------------------------------

        st.success("Answer Found")

        st.write("### 📄 Research Article")
        st.write(best_title)

        st.write("### 💡 Answer")
        st.write(best_answer)

        st.write(
            "Similarity Score:",
            round(best_score, 3)
        )


# -----------------------------------------
# INFORMATION
# -----------------------------------------

st.write("---")
st.info(
    "This is an educational NLP demonstration "
    "using a small PubMed-style dataset."
)
