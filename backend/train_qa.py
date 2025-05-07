import json
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
import os
import torch

# 1. โหลด thai_laws.json และแปลงเป็นรูปแบบ SQuAD
def load_thai_laws(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    squad_data = []
    for i, item in enumerate(data):
        squad_data.append({
            "id": f"thai_laws_{i}",
            "title": "thai_laws",
            "context": item["answer"],
            "question": item["question"],
            "answers": {
                "text": [item["answer"]],
                "answer_start": [0]
            }
        })
    return squad_data

# 2. โหลด HuggingFace dataset และแปลงเป็นรูปแบบ SQuAD
def load_hf_dataset():
    ds = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG", split="train")
    squad_data = []
    for i, item in enumerate(ds):
        # ใช้ positive_context ตัวแรกถ้ามี
        if "positive_contexts" in item and item["positive_contexts"]:
            ctx = item["positive_contexts"][0]
            if isinstance(ctx, dict):
                context = " ".join(str(v) for v in ctx.values())
            else:
                context = str(ctx)
        elif "context" in item:
            context = str(item["context"])
        elif "text" in item:
            context = str(item["text"])
        elif "law_text" in item:
            context = str(item["law_text"])
        else:
            context = ""
        squad_data.append({
            "id": f"hf_{i}",
            "title": "hf_dataset",
            "context": context,
            "question": item.get("question", ""),
            "answers": {
                "text": [item.get("answer", "")],
                "answer_start": [context.find(item.get("answer", "")) if item.get("answer", "") in context else 0]
            }
        })
    return squad_data

# 3. รวมชุดข้อมูล
thai_laws = load_thai_laws("backend/documents/thai_laws.json")
hf_laws = load_hf_dataset()
all_data = thai_laws + hf_laws
dataset = Dataset.from_list(all_data)

# 4. Tokenize
model_name = "airesearch/WangchanX-Legal-ThaiCCL-Retriever"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    # Tokenize คำถามและบริบท
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # สร้าง start_positions และ end_positions สำหรับแต่ละตัวอย่าง
    start_positions = []
    end_positions = []
    sample_mapping = tokenized["overflow_to_sample_mapping"]
    offset_mapping = tokenized["offset_mapping"]

    # ตัวอย่าง batch: examples["answers"] เป็น list ของ dict
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        offsets = offset_mapping[i]
        # ดึง answer/text และ answer_start จาก dict
        answer = examples["answers"][sample_idx]["text"][0]
        answer_start = examples["answers"][sample_idx]["answer_start"][0]
        answer_end = answer_start + len(answer)

        # หา token index ที่ครอบคลุม answer
        sequence_ids = tokenized.sequence_ids(i)
        # หา context token indices
        context_start = None
        context_end = None
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id == 1 and context_start is None:
                context_start = idx
            if seq_id == 1:
                context_end = idx
        if context_start is None or context_end is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # หา token ที่ครอบคลุม answer
        start_pos = context_start
        end_pos = context_start
        found = False
        for idx in range(context_start, context_end + 1):
            start_char, end_char = offsets[idx]
            if start_char <= answer_start < end_char:
                start_pos = idx
            if start_char < answer_end <= end_char:
                end_pos = idx
                found = True
                break
        if not found:
            start_pos = context_start
            end_pos = context_start

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    # ลบ offset_mapping และ overflow_to_sample_mapping ออกจากผลลัพธ์
    tokenized.pop("offset_mapping", None)
    tokenized.pop("overflow_to_sample_mapping", None)
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# 5. โหลดโมเดล
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# แสดงข้อมูล GPU เพื่อ debug
if torch.cuda.is_available():
    print("พบ CUDA ใช้งาน GPU:", torch.cuda.get_device_name(0))
else:
    print("ไม่พบ CUDA ใช้งาน CPU.")

# 6. กำหนด arguments สำหรับการ train
training_args = TrainingArguments(
    output_dir="./trained_qa_model",
    per_device_train_batch_size=1,  # ลดจาก 4 เหลือ 1
    gradient_accumulation_steps=4,  # สะสม gradient เพื่อจำลอง batch size 4
    num_train_epochs=2,
    learning_rate=3e-5,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    remove_unused_columns=False,
    fp16=False
    # ลบ device ออก
)

# 7. สร้าง Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer
)

# 8. Train
trainer.train()
trainer.save_model("./trained_qa_model")
