import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hf_datasets_utils import load_legal_dataset  # เปลี่ยนมาใช้ฟังก์ชันนี้

# โหลดเอกสารจากไฟล์
with open("backend/documents/thai_laws.json", "r", encoding="utf-8") as f:
    law_docs = json.load(f)

# ใช้ "question" เป็น text สำหรับ vectorizer
texts = [doc["question"] for doc in law_docs]
vectorizer = TfidfVectorizer().fit(texts)

def retrieve_context(query, threshold=0.2):
    # แปลง query เป็นเวกเตอร์
    query_vec = vectorizer.transform([query])
    # คำนวณความคล้ายคลึงกัน
    sims = cosine_similarity(query_vec, vectorizer.transform(texts)).flatten()
    max_sim = sims.max()
    if max_sim >= threshold:
        # return เฉพาะ answer (string)
        best_match_idx = sims.argmax()
        best_doc = law_docs[best_match_idx]
        # คืนทั้ง question+answer เพื่อให้ context สมบูรณ์
        context = f"Q: {best_doc['answer']}"
        print("Selected context from thai_laws.json:", context)  # debug
        return context
    else:
        # ใช้ dataset จาก HuggingFace ผ่านฟังก์ชันใน hf_datasets_utils.py
        dataset = load_legal_dataset()
        sample = dataset[0]
        # Debug: print keys for inspection
        print("Sample keys from dataset:", list(sample.keys()))
        # เลือก context จาก positive_contexts ถ้ามี
        if "positive_contexts" in sample and sample["positive_contexts"]:
            dataset_texts = []
            for item in dataset:
                pcs = item["positive_contexts"]
                if pcs:
                    # ถ้าเป็น dict ให้ join, ถ้าเป็น string ใช้ตรงๆ
                    first_ctx = pcs[0]
                    if isinstance(first_ctx, dict):
                        ctx_str = " ".join(str(v) for v in first_ctx.values())
                    else:
                        ctx_str = str(first_ctx)
                    dataset_texts.append(ctx_str)
        elif "context" in sample:
            dataset_texts = [str(item["context"]) for item in dataset]
        elif "text" in sample:
            dataset_texts = [str(item["text"]) for item in dataset]
        elif "law_text" in sample:
            dataset_texts = [str(item["law_text"]) for item in dataset]
        else:
            # fallback: รวมค่าทุก field เป็น string
            dataset_texts = [" ".join(str(v) for v in item.values()) for item in dataset]
        # ป้องกัน None
        dataset_texts = [t if t is not None else "" for t in dataset_texts]
        ds_vectorizer = TfidfVectorizer().fit(dataset_texts)
        ds_query_vec = ds_vectorizer.transform([query])
        ds_sims = cosine_similarity(ds_query_vec, ds_vectorizer.transform(dataset_texts)).flatten()
        best_idx = ds_sims.argmax()
        best_match = dataset_texts[best_idx]
        print("Selected context from HuggingFace dataset:", best_match)  # debug
        return best_match