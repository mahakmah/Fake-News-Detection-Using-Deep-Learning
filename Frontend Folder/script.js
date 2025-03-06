async function checkNews() {
    let newsText = document.getElementById("newsInput").value.trim();
    let resultBox = document.getElementById("result");

    if (newsText === "") {
        alert("⚠️ Please enter news text!");
        return;
    }

    resultBox.innerHTML = "Checking... ⏳";

    try {
        let response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ news: newsText })
        });

        let data = await response.json();

        if (data.prediction === "fake") {
            resultBox.innerHTML = "🛑 Fake News!";
            resultBox.className = "fake";
        } else {
            resultBox.innerHTML = "✅ Real News!";
            resultBox.className = "real";
        }

    } catch (error) {
        console.error("Error:", error);
        resultBox.innerHTML = "❌ Error checking news!";
        resultBox.className = "";
    }
}
