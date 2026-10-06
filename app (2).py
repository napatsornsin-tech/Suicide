
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Suicide Risk Text Classification",
    page_icon="🔍"
)

st.title("Suicide Risk Text Classification")
st.write("ระบบจำแนกข้อความด้วย Transformer Embedding และ SVM")

# โหลดโมเดล
@st.cache_resource
def load_models():

    # ดาวน์โหลด SVM จาก Hugging Face
    model_path = hf_hub_download(
        repo_id="Napatsornn/suicide",
        filename="svm_model.joblib"
    )

    svm_model = joblib.load(model_path)

    # โหลด Embedding Model
    embedding_model = SentenceTransformer(
        "intfloat/multilingual-e5-large-instruct"
    )

    return svm_model, embedding_model


svm_model, embedding_model = load_models()


# ช่องกรอกข้อความ
text = st.text_area(
    "ข้อความ",
    placeholder="พิมพ์ข้อความที่ต้องการวิเคราะห์...",
    height=180
)


# ปุ่มวิเคราะห์
if st.button("วิเคราะห์ข้อความ"):

    if not text.strip():

        st.warning("กรุณากรอกข้อความ")

    else:

        embedding = embedding_model.encode(
            [text],
            normalize_embeddings=True
        )

        prediction = svm_model.predict(embedding)[0]

        st.subheader("ผลการจำแนก")

        st.success(str(prediction))
