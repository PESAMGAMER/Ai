function updatePlaceholder() {
    const chatWindow = document.getElementById("chat-window");
    const placeholder = document.getElementById("chat-placeholder");
    // นับเฉพาะ div.message
    const hasMessage = chatWindow.querySelectorAll('.message').length > 0;
    if (hasMessage) {
        chatWindow.classList.add('has-message');
        if (placeholder) placeholder.style.display = "none";
    } else {
        chatWindow.classList.remove('has-message');
        if (placeholder) placeholder.style.display = "";
    }
}

function appendMessage(text, sender = "bot") {
    const chatWindow = document.getElementById("chat-window");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message" + (sender === "user" ? " user" : " bot");
    msgDiv.innerHTML = `
        <div class="bubble">${text}</div>
    `;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    updatePlaceholder();
}

function ask() {
    const input = document.getElementById("question");
    const q = input.value.trim();
    if (!q) return;
    appendMessage(q, "user");
    input.value = "";
    appendMessage("กำลังค้นหาคำตอบ...", "bot");
    fetch("http://127.0.0.1:5000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    })
    .then(res => res.json())
    .then(data => {
      const chatWindow = document.getElementById("chat-window");
      chatWindow.removeChild(chatWindow.lastChild);
      appendMessage(data.answer, "bot");
    })
    .catch(() => {
      const chatWindow = document.getElementById("chat-window");
      chatWindow.removeChild(chatWindow.lastChild);
      appendMessage("เกิดข้อผิดพลาดในการเชื่อมต่อ", "bot");
    });
}

// เรียกเมื่อโหลดหน้า เพื่อแสดง placeholder ถ้ายังไม่มีข้อความ
window.addEventListener('DOMContentLoaded', updatePlaceholder);