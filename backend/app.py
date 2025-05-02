from flask import Flask, request, jsonify
from flask_cors import CORS  # เพิ่มการนำเข้า CORS
from retriever import retrieve_context
from qa_model import get_answer

app = Flask(__name__)
CORS(app)  # เปิดใช้งาน CORS

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    context = retrieve_context(question)
    answer = get_answer(question, context)
    return jsonify({"answer": answer, "context": context})

if __name__ == "__main__":
    app.run(debug=True)