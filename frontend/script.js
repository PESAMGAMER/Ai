function ask() {
    const q = document.getElementById("question").value;
    fetch("http://127.0.0.1:5000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    })
    .then(res => res.json())
    .then(data => {
      document.getElementById("response").innerText = "คำตอบ: " + data.answer;
    });
  }