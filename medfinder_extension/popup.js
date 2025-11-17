const LOCAL_API = 'http://127.0.0.1:8501';
const PROD_API = 'https://<YOUR_RAILWAY_URL>';

function apiBase() {
    return PROD_API || LOCAL_API;
}

document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("query");
    const btn = document.getElementById("searchBtn");
    const resultsDiv = document.getElementById("results");
    const statusDiv = document.getElementById("status");

    btn.addEventListener("click", () => {
        let q = input.value.trim();
        if (!q) {
            resultsDiv.innerHTML = `<p class="error">Please enter a search query.</p>`;
            return;
        }

        statusDiv.textContent = "Searching...";
        resultsDiv.innerHTML = "";

        fetch(`${apiBase()}/api/search?q=${encodeURIComponent(q)}&limit=10`)
            .then(res => res.json())
            .then(data => {
                statusDiv.textContent = "";
                if (!data.results || data.results.length === 0) {
                    resultsDiv.innerHTML = `<p class="error">No results found.</p>`;
                    return;
                }
                data.results.forEach(item => {
                    const li=document.createElement("li");
                    li.className="result-item";
                    li.innerHTML=`
                        <img src="${item.image || ''}">
                        <div>
                            <p class="result-title">${item.title}</p>
                            <p class="result-meta">💲 ${item.price} ${item.currency} | 📦 ${item.condition}</p>
                            <a href="${item.url}" target="_blank">View</a>
                        </div>`;
                    resultsDiv.appendChild(li);
                });
            })
            .catch(err => {
                statusDiv.textContent = "Backend error";
                console.error(err);
            });
    });
});
