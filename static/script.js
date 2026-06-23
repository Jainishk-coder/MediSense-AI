const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const answerBox = document.getElementById("answerBox");

// Auto Resize Textarea
function autoResize() {
  questionInput.style.height = "auto";
  questionInput.style.height = questionInput.scrollHeight + "px";
}

questionInput.addEventListener("input", autoResize);

window.addEventListener("load", autoResize);

// Ask AI Function
async function askAI() {
  const question = questionInput.value.trim();

  if (!question) {
    answerBox.innerHTML = `
      <div class="default-text">
        Please enter a question.
      </div>
    `;
    return;
  }

  // Disable button while processing
  askBtn.disabled = true;
  askBtn.innerHTML = "⏳ Analyzing...";

  answerBox.innerHTML = `
    <div class="loading">
      🤖 AI is analyzing your symptoms...
    </div>
  `;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
      }),
    });

    const data = await response.json();

    const formattedAnswer = data.answer
      .replace(/^- /gm, "• ")
      .replace(/\n/g, "<br>");

    answerBox.innerHTML = `
        <div class="formatted-answer">
          <h3 style="
            margin-bottom:20px;
            color:#00e5ff;
            font-size:30px;
            font-weight:700;
          ">
            🩺 MediSense AI Analysis
          </h3>

          ${formattedAnswer}
        </div>
      `;
    // Smooth scroll to answer
    answerBox.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  } catch (error) {
    answerBox.innerHTML = `
      <div style="
        color:#ff6b6b;
        font-weight:600;
      ">
        ❌ Unable to connect to AI Assistant.
      </div>
    `;

    console.error(error);
  } finally {
    askBtn.disabled = false;
    askBtn.innerHTML = "Analyze Symptoms";
  }
}

// Button Click
askBtn.addEventListener("click", askAI);

// Enter Key Support
questionInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();

    askAI();
  }
});
