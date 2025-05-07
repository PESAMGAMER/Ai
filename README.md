# LegalBot
แชทบอทสำหรับตอบคำถามกฎหมายภาษาไทย โดยใช้โมเดลจาก Hugging Face (WangchanBERTa)

## วิธีติดตั้ง
1. สร้าง Python virtual environment
2. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```
3. รัน backend:
```bash
python backend/app.py 
```
4. เปิด `frontend/index.html` ในเบราว์เซอร์

. รัน train datasets:
```bash
python backend/train_qa.py 
```
