const API_BASE = "http://localhost:8000";
let similarityChart = null;

// Tab switching logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.target).classList.add('active');
    });
});

function updateThreshold(val) {
    document.getElementById('threshold-val').textContent = val;
}

async function checkCopyright(type) {
    const threshold = document.getElementById('threshold-slider').value;
    let url = `${API_BASE}/check-${type}`;
    
    let options = {
        method: 'POST',
    };

    if (type === 'text') {
        const text = document.getElementById('text-input').value;
        if (!text) return alert("Please enter text");
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify({ text: text });
        url += `?threshold=${threshold}`; // For text we pass it as query param based on FastAPI setup
    } else {
        const fileInput = document.getElementById(`${type}-upload`);
        if (fileInput.files.length === 0) return alert("Please select a file");
        
        // For simplicity, we just handle the first file in batch. 
        // A true batch system would loop over fileInput.files and send multiple requests or have a batch endpoint.
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('threshold', threshold);
        options.body = formData;
    }

    try {
        const response = await fetch(url, options);
        const result = await response.json();
        displayResult(result);
        fetchHistory(); // Update history table
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred. Make sure the backend is running.");
    }
}

function displayResult(result) {
    const panel = document.getElementById('results-panel');
    const content = document.getElementById('results-content');
    
    panel.style.display = 'block';
    
    const riskClass = result.risk_level === 'HIGH' ? 'risk-high' : 'risk-low';
    const simPercent = (result.similarity_score * 100).toFixed(2);
    
    content.innerHTML = `
        <p><strong>Similarity:</strong> ${simPercent}%</p>
        <p><strong>Top Match:</strong> ${result.top_match}</p>
        <p><strong>Risk Level:</strong> <span class="${riskClass}">${result.risk_level}</span></p>
    `;

    updateChart(result.similarity_score);
}

function updateChart(score) {
    const ctx = document.getElementById('similarityChart').getContext('2d');
    
    if (similarityChart) {
        similarityChart.destroy();
    }

    similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Similarity Score'],
            datasets: [{
                label: 'Score',
                data: [score],
                backgroundColor: score >= document.getElementById('threshold-slider').value ? '#cf6679' : '#03dac6',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            maintainAspectRatio: false
        }
    });
}

async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE}/history?limit=10`);
        const history = await response.json();
        
        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';
        
        history.forEach(item => {
            const riskClass = item.risk_level === 'HIGH' ? 'risk-high' : 'risk-low';
            const time = new Date(item.timestamp).toLocaleString();
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${time}</td>
                <td>${item.filename}</td>
                <td>${item.file_type}</td>
                <td>${(item.similarity_score * 100).toFixed(2)}%</td>
                <td>${item.matched_with}</td>
                <td class="${riskClass}">${item.risk_level}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error("Error fetching history:", error);
    }
}

// Initial fetch
document.addEventListener("DOMContentLoaded", fetchHistory);
