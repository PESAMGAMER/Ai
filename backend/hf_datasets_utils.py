from datasets import load_dataset

def load_legal_dataset():
    # โหลด dataset จาก HuggingFace
    dataset = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG", split="train")
    return dataset

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    ds = load_legal_dataset()
    print(ds[0])
