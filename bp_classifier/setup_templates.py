import os

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ── index.html ────────────────────────────────────────────────────────────────
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>BP State Classifier</title>
  <link rel="stylesheet" href="{{ url_for(\'static\', filename=\'style.css\') }}"/>
</head>
<body>
<header>
  <div class="logo">&#x1F�BA</div>
  <h1>AI Blood Pressure Classifier</h1>
  <p class="subtitle">Upload a health report or enter parameters manually</p>
</header>

{% if not models_loaded %}
<div class="alert alert-warn">
  Models not trained yet. Run <code>python train_model.py</code> first.
</div>
{% endif %}

<main>
  <div class="tabs">
    <button class="tab active" onclick="showTab(\'manual\', this)">✏️ Manual Input</button>
    <button class="tab"        onclick="showTab(\'upload\', this)">📄 Upload Report</button>
  </div>

  <div id="tab-manual" class="tab-content active">
    <form action="/predict" method="POST">
      <div class="grid">
        <div class="field">
          <label>Age</label>
          <input type="number" name="age" placeholder="e.g. 55" min="1" max="120" required/>
        </div>
        <div class="field">
          <label>Sex <span class="hint">(0=Female, 1=Male)</span></label>
          <select name="sex">
            <option value="1">Male (1)</option>
            <option value="0">Female (0)</option>
          </select>
        </div>
        <div class="field">
          <label>Resting Systolic BP <span class="hint">(mmHg)</span></label>
          <input type="number" name="trestbps" placeholder="e.g. 135" required/>
        </div>
        <div class="field">
          <label>Cholesterol <span class="hint">(mg/dl)</span></label>
          <input type="number" name="chol" placeholder="e.g. 240" required/>
        </div>
        <div class="field">
          <label>Fasting Blood Sugar &gt;120 <span class="hint">(0/1)</span></label>
          <select name="fbs">
            <option value="0">No (0)</option>
            <option value="1">Yes (1)</option>
          </select>
        </div>
        <div class="field">
          <label>Max Heart Rate <span class="hint">(thalach)</span></label>
          <input type="number" name="thalach" placeholder="e.g. 150" required/>
        </div>
        <div class="field">
          <label>Chest Pain During Exercise <span class="hint">(0/1)</span></label>
          <select name="exang">
            <option value="0">No (0)</option>
            <option value="1">Yes (1)</option>
          </select>
        </div>
        <div class="field">
          <label>Heart Stress Score <span class="hint">(0–6)</span></label>
          <input type="number" name="oldpeak" step="0.1" placeholder="e.g. 1.5" required/>
        </div>
      </div>
      <button type="submit" class="btn-predict" {% if not models_loaded %}disabled{% endif %}>
        🔍 Classify Blood Pressure State
      </button>
    </form>
  </div>

  <div id="tab-upload" class="tab-content">
    <form action="/upload" method="POST" enctype="multipart/form-data">
      <div class="upload-zone" onclick="document.getElementById(\'fileInput\').click()">
        <div class="upload-icon">📂</div>
        <p>Click to browse or drag and drop your health report</p>
        <p class="hint">Supports PDF, PNG, JPG</p>
        <input id="fileInput" type="file" name="report" accept=".pdf,.png,.jpg,.jpeg" hidden
               onchange="document.getElementById(\'fname\').textContent = this.files[0].name"/>
      </div>
      <p id="fname" class="file-name"></p>
      <button type="submit" class="btn-predict" {% if not models_loaded %}disabled{% endif %}>
        📤 Upload and Classify
      </button>
    </form>
  </div>
</main>

<footer>
  <p>Trained on UCI Cleveland Heart Disease dataset · 4 ML models · For educational use only</p>
</footer>

<script>
function showTab(name, btn) {
  document.querySelectorAll(\'.tab-content\').forEach(t => t.classList.remove(\'active\'));
  document.querySelectorAll(\'.tab\').forEach(t => t.classList.remove(\'active\'));
  document.getElementById(\'tab-\' + name).classList.add(\'active\');
  btn.classList.add(\'active\');
}
</script>
</body>
</html>''')

# ── result.html ───────────────────────────────────────────────────────────────
with open("templates/result.html", "w", encoding="utf-8") as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>BP Classification Result</title>
  <link rel="stylesheet" href="{{ url_for(\'static\', filename=\'style.css\') }}"/>
</head>
<body>
<header>
  <div class="logo">🩺</div>
  <h1>Classification Result</h1>
</header>
<main>

  <div class="verdict-card" style="border-left: 6px solid {{ final_color }};">
    <div class="verdict-label" style="color: {{ final_color }};">{{ final_label }}</div>
    <p class="verdict-sub">Consensus prediction across all {{ predictions|length }} models</p>
  </div>

  <div class="bp-legend card">
    <h3>Blood Pressure Classification Reference</h3>
    <table>
      <thead><tr><th>Category</th><th>Systolic (mmHg)</th><th>Status</th></tr></thead>
      <tbody>
        {% for label, color in bp_colors.items() %}
        <tr {% if label == final_label %}class="highlight-row"{% endif %}>
          <td><span class="dot" style="background:{{color}}"></span>{{ label }}</td>
          <td>
            {% if label == "Normal" %}&lt; 120
            {% elif label == "Elevated" %}120 to 129
            {% elif label == "Hypertension Stage 1" %}130 to 139
            {% elif label == "Hypertension Stage 2" %}140 to 179
            {% else %}&ge; 180{% endif %}
          </td>
          <td>{{ "Your Result" if label == final_label else "" }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h3>Individual Model Predictions</h3>
    <div class="model-grid">
      {% for model_name, pred in predictions.items() %}
      <div class="model-card" style="border-top: 4px solid {{ bp_colors.get(pred.label, \'#888\') }}">
        <div class="model-name">{{ model_name }}</div>
        <div class="model-pred" style="color: {{ bp_colors.get(pred.label, \'#888\') }}">{{ pred.label }}</div>
        <div class="model-conf">{{ pred.confidence }}% confidence</div>
        <div class="conf-bar">
          <div class="conf-fill" style="width:{{ pred.confidence }}%; background:{{ bp_colors.get(pred.label, \'#888\') }}"></div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="card">
    <h3>Trained Model Performance (Test Set)</h3>
    <div class="metrics-table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
        <tbody>
          {% for model_name, scores in meta.results.items() %}
          <tr {% if model_name == meta.best_model %}class="best-row"{% endif %}>
            <td>{{ model_name }}{% if model_name == meta.best_model %} 🏆{% endif %}</td>
            <td>{{ scores.accuracy }}%</td>
            <td>{{ scores.precision }}%</td>
            <td>{{ scores.recall }}%</td>
            <td>{{ scores.f1_score }}%</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    <img src="/static/plots/model_comparison.png" class="plot-img" alt="Model comparison"/>
  </div>

  <div class="card">
    <h3>Parameters Used
      {% if input_type == "upload" %}<span class="badge">OCR Upload</span>
      {% else %}<span class="badge">Manual Input</span>{% endif %}
    </h3>
    <div class="params-grid">
      {% for feat, val in params.items() %}
      <div class="param-item">
        <span class="param-label">{{ feat }}</span>
        <span class="param-val">{{ val }}</span>
      </div>
      {% endfor %}
    </div>
  </div>

  <a href="/" class="btn-back">← New Prediction</a>
</main>
<footer>
  <p>This tool is for educational purposes only and does not constitute medical advice.</p>
</footer>
</body>
</html>''')

print("✅ templates/index.html created")
print("✅ templates/result.html created")
print("\nNow run: python app.py")