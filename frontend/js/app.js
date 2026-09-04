/**
 * PhishGuard-AI Frontend Application
 * Strict Vanilla JavaScript (ES6+) - Zero framework dependencies
 */

const API_BASE = window.location.origin;

// Application State
const state = {
  currentTab: 'home',
  systemStatus: null,
  modelMetrics: null
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupCharacterCounters();
  checkSystemHealth();
  // Periodically check health
  setInterval(checkSystemHealth, 30000);
});

/* ===================================================================
   Navigation & Tabs
   =================================================================== */
function setupNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;

  // Update tabs active state
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.getAttribute('data-tab') === tabId);
  });

  // Update views active state
  document.querySelectorAll('.tab-view').forEach(v => {
    v.classList.remove('active');
  });

  const targetView = document.getElementById(`view-${tabId}`);
  if (targetView) {
    targetView.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // View-specific on-demand data loads
  if (tabId === 'performance') {
    loadMetrics();
  } else if (tabId === 'history') {
    loadScanHistory();
  }
}

/* ===================================================================
   Character Counters
   =================================================================== */
function setupCharacterCounters() {
  const urlInput = document.getElementById('urlInput');
  const urlCounter = document.getElementById('urlCharCounter');
  if (urlInput && urlCounter) {
    urlInput.addEventListener('input', () => {
      urlCounter.textContent = `${urlInput.value.length} / 2048 chars`;
    });
  }

  const msgInput = document.getElementById('messageInput');
  const msgCounter = document.getElementById('messageCharCounter');
  if (msgInput && msgCounter) {
    msgInput.addEventListener('input', () => {
      msgCounter.textContent = `${msgInput.value.length} / 10000 chars`;
    });
  }
}

/* ===================================================================
   System Health Check
   =================================================================== */
async function checkSystemHealth() {
  const statusBadge = document.getElementById('systemStatusBadge');
  const statusText = document.getElementById('systemStatusText');

  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check error');
    const data = await res.json();
    state.systemStatus = data;

    const allLoaded = Object.values(data.models_status).every(Boolean);
    if (allLoaded) {
      statusText.textContent = 'Models Online';
      statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else {
      statusText.textContent = 'Heuristic Fallback';
      statusBadge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
    }
  } catch (err) {
    statusText.textContent = 'Engine Offline';
    statusBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
  }
}

/* ===================================================================
   Presets
   =================================================================== */
function applyUrlPreset(url) {
  const input = document.getElementById('urlInput');
  if (input) {
    input.value = url;
    input.dispatchEvent(new Event('input'));
    input.focus();
  }
}

function applyMessagePreset(msg) {
  const input = document.getElementById('messageInput');
  if (input) {
    input.value = msg;
    input.dispatchEvent(new Event('input'));
    input.focus();
  }
}

/* ===================================================================
   URL Scanner Handler
   =================================================================== */
async function handleUrlSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('urlInput');
  const btn = document.getElementById('urlSubmitBtn');
  const container = document.getElementById('urlResultContainer');
  const url = input.value.trim();

  if (!url) {
    showToast('Please enter a URL to analyze.');
    return;
  }

  setButtonLoading(btn, true);
  container.style.display = 'none';

  try {
    const response = await fetch(`${API_BASE}/api/v1/predict/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    const data = await response.json();

    if (!response.ok) {
      const msg = data.details ? data.details.map(d => d.message).join(', ') : (data.message || 'Scan failed');
      throw new Error(msg);
    }

    renderPredictionResult(container, data, 'URL Threat Assessment');
    showToast('Static URL threat analysis complete');
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    setButtonLoading(btn, false);
  }
}

/* ===================================================================
   Message Scanner Handler
   =================================================================== */
async function handleMessageSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('messageInput');
  const btn = document.getElementById('messageSubmitBtn');
  const container = document.getElementById('messageResultContainer');
  const message = input.value.trim();

  if (!message) {
    showToast('Please enter message text to analyze.');
    return;
  }

  setButtonLoading(btn, true);
  container.style.display = 'none';

  try {
    const response = await fetch(`${API_BASE}/api/v1/predict/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const data = await response.json();

    if (!response.ok) {
      const msg = data.details ? data.details.map(d => d.message).join(', ') : (data.message || 'Scan failed');
      throw new Error(msg);
    }

    renderPredictionResult(container, data, 'Message Threat Assessment');
    showToast('Message sequence threat analysis complete');
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    setButtonLoading(btn, false);
  }
}

/* ===================================================================
   Result Renderer (Aesthetic & Human-Centered)
   =================================================================== */
function renderPredictionResult(container, data, title) {
  const tierClass = data.risk_level.toLowerCase();
  const pct = (data.probability * 100).toFixed(1);

  let badgeLabel = 'LOW RISK';
  let badgeEmoji = '✨';
  if (tierClass === 'high') {
    badgeLabel = 'HIGH RISK SCAM';
    badgeEmoji = '🚨';
  } else if (tierClass === 'medium') {
    badgeLabel = 'MODERATE SUSPICION';
    badgeEmoji = '⚠️';
  }

  let indicatorsHtml = '';
  if (data.indicators && data.indicators.length > 0) {
    indicatorsHtml = data.indicators.map(ind => {
      const sev = ind.severity.toLowerCase();
      let icon = 'ℹ️';
      let sevLabel = 'Normal';
      if (sev === 'critical') {
        icon = '🚨';
        sevLabel = 'High Alert';
      } else if (sev === 'warning') {
        icon = '⚠️';
        sevLabel = 'Caution';
      }
      return `
        <div class="indicator-item ${sev}">
          <span class="severity-pill ${sev}">${icon} ${sevLabel}</span>
          <div class="indicator-content">
            <h5>${escapeHtml(ind.name)}</h5>
            <p>${escapeHtml(ind.description)}</p>
            ${ind.details ? `<div class="indicator-detail">${escapeHtml(ind.details)}</div>` : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  // Feature Breakdown table if URL features present
  let featuresHtml = '';
  if (data.features && data.scan_type === 'url') {
    const rows = Object.entries(data.features).map(([k, v]) => `
      <tr>
        <td><code>${escapeHtml(k.replace(/_/g, ' '))}</code></td>
        <td><code>${typeof v === 'number' ? v : escapeHtml(String(v))}</code></td>
      </tr>
    `).join('');

    featuresHtml = `
      <div class="features-accordion">
        <button type="button" class="features-toggle" onclick="toggleFeatures(this)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
          View Technical Feature Breakdown (Optional)
        </button>
        <div class="features-table-wrapper" style="display: none;">
          <table class="features-table">
            <thead>
              <tr><th>Signal Attribute</th><th>Extracted Value</th></tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  const html = `
    <div class="result-card tier-${tierClass}">
      <div class="result-header">
        <div class="result-target-meta">
          <span class="target-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline>
            </svg>
            ${escapeHtml(title)}
          </span>
          <div class="result-target-text">${escapeHtml(data.target)}</div>
        </div>
        <div class="risk-badge ${tierClass}">
          ${badgeEmoji} ${badgeLabel}
        </div>
      </div>

      <div class="result-metrics-grid">
        <div class="metric-stat">
          <span class="stat-label">Threat Likelihood</span>
          <span class="stat-value" style="color: ${tierClass === 'high' ? 'var(--danger-rose)' : (tierClass === 'medium' ? 'var(--caution-amber)' : 'var(--safe-emerald)')};">
            ${(data.probability * 100).toFixed(1)}%
          </span>
        </div>
        <div class="metric-stat">
          <span class="stat-label">AI Confidence</span>
          <span class="stat-value">${data.confidence_score.toFixed(1)}%</span>
        </div>
        <div class="metric-stat">
          <span class="stat-label">Protection Mode</span>
          <span class="stat-value" style="font-size: 1.1rem; color: #c4b5fd;">
            ${data.is_fallback ? 'Heuristic Mode' : 'Dual Neural AI'}
          </span>
        </div>
      </div>

      <div class="probability-meter-container">
        <div class="meter-labels">
          <span>Safe &amp; Benign (0%)</span>
          <span>Caution Threshold (50%)</span>
          <span>Dangerous Phish (100%)</span>
        </div>
        <div class="meter-track">
          <div class="meter-fill ${tierClass}" style="width: ${pct}%;"></div>
        </div>
      </div>

      <div class="action-box ${tierClass}">
        <h4>💡 Recommended Action</h4>
        <p>${escapeHtml(data.recommended_action)}</p>
      </div>

      <div class="indicators-section">
        <h3>🔍 What Our AI Detected</h3>
        <div class="indicators-list">
          ${indicatorsHtml}
        </div>
      </div>

      ${featuresHtml}
    </div>
  `;

  container.innerHTML = html;
  container.style.display = 'block';
  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function toggleFeatures(btn) {
  const wrapper = btn.nextElementSibling;
  const isHidden = wrapper.style.display === 'none';
  wrapper.style.display = isHidden ? 'block' : 'none';
  btn.querySelector('svg').style.transform = isHidden ? 'rotate(180deg)' : 'none';
}

/* ===================================================================
   Model Performance Loader
   =================================================================== */
async function loadMetrics() {
  const container = document.getElementById('modelsGridContainer');
  const tsElem = document.getElementById('metricsLastLoaded');

  try {
    const res = await fetch(`${API_BASE}/api/v1/models/metrics`);
    if (!res.ok) throw new Error('Could not retrieve model metrics');
    const data = await res.json();
    state.modelMetrics = data;

    tsElem.textContent = `Updated: ${new Date().toLocaleTimeString()}`;

    if (!data.url_model && !data.message_model) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1; text-align: center; padding: 3rem;">
          <p>⚠️ No model metrics found on disk.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = `
      ${data.url_model ? renderModelCard(data.url_model, 'Link Threat Classifier (ANN)') : ''}
      ${data.message_model ? renderModelCard(data.message_model, 'Message Threat Classifier (RNN)') : ''}
    `;
  } catch (err) {
    container.innerHTML = `<div class="empty-state">Failed to load metrics: ${escapeHtml(err.message)}</div>`;
  }
}

function renderModelCard(model, title) {
  const cm = model.confusion_matrix || [[0, 0], [0, 0]];
  const tn = cm[0][0] || 0;
  const fp = cm[0][1] || 0;
  const fn = cm[1][0] || 0;
  const tp = cm[1][1] || 0;

  const splits = model.dataset_splits || {};
  const trainCount = splits.train || 0;
  const valCount = splits.validation || 0;
  const testCount = splits.test || 0;

  return `
    <div class="model-card">
      <div class="model-card-header">
        <div>
          <h3>${escapeHtml(title)}</h3>
          <span style="font-size: 0.8rem; color: var(--text-muted);">${escapeHtml(model.model_type)}</span>
        </div>
        <span class="model-arch-badge">Validated</span>
      </div>

      <div class="metrics-grid-4">
        <div class="m-stat">
          <span class="m-label">Accuracy</span>
          <span class="m-val">${(model.accuracy * 100).toFixed(1)}%</span>
        </div>
        <div class="m-stat">
          <span class="m-label">Precision</span>
          <span class="m-val">${(model.precision * 100).toFixed(1)}%</span>
        </div>
        <div class="m-stat">
          <span class="m-label">Recall</span>
          <span class="m-val">${(model.recall * 100).toFixed(1)}%</span>
        </div>
        <div class="m-stat">
          <span class="m-label">F1-Score</span>
          <span class="m-val">${(model.f1_score * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div class="confusion-matrix-box">
        <h4>Model Verification Matrix (Unseen Test Data)</h4>
        <div class="cm-grid">
          <div></div>
          <div class="cm-header">AI Verdict: Safe</div>
          <div class="cm-header">AI Verdict: Threat</div>

          <div class="cm-label">Actually Safe</div>
          <div class="cm-cell tn">Correct Safe: <strong>${tn}</strong></div>
          <div class="cm-cell fp">False Alarm: <strong>${fp}</strong></div>

          <div class="cm-label">Actual Scam</div>
          <div class="cm-cell fn">Missed Threat: <strong>${fn}</strong></div>
          <div class="cm-cell tp">Blocked Scam: <strong>${tp}</strong></div>
        </div>
      </div>

      <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.6;">
        <div><strong>Architecture:</strong> <code>${escapeHtml(model.architecture)}</code></div>
        <div style="margin-top: 0.35rem;">
          <strong>Splits:</strong> 70% Train (${trainCount}) | 15% Val (${valCount}) | 15% Test (${testCount})
        </div>
        ${model.trained_at ? `<div style="margin-top: 0.25rem;"><strong>Trained:</strong> ${new Date(model.trained_at).toLocaleString()}</div>` : ''}
      </div>
    </div>
  `;
}

/* ===================================================================
   Audit History Loader
   =================================================================== */
async function loadScanHistory() {
  const tbody = document.getElementById('historyTableBody');
  if (!tbody) return;

  try {
    const res = await fetch(`${API_BASE}/api/v1/history?limit=30`);
    if (!res.ok) throw new Error('Could not load history');
    const items = await res.json();

    if (!items || items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No audit scans recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = items.map(item => `
      <tr>
        <td>#${item.id}</td>
        <td><span class="target-badge">${escapeHtml(item.scan_type)}</span></td>
        <td style="max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${escapeHtml(item.target)}
        </td>
        <td>
          <span class="tier-badge ${item.risk_level.toLowerCase()}">
            ${escapeHtml(item.risk_level)}
          </span>
        </td>
        <td>${(item.probability * 100).toFixed(1)}%</td>
        <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-secondary);">
          ${escapeHtml(item.verdict)}
        </td>
        <td style="color: var(--text-muted); font-size: 0.75rem;">
          ${new Date(item.created_at).toLocaleString()}
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">Error loading history: ${escapeHtml(err.message)}</td></tr>`;
  }
}

async function clearScanHistory() {
  if (!confirm('Are you sure you want to clear the scan history?')) return;
  try {
    const res = await fetch(`${API_BASE}/api/v1/history`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to clear history');
    showToast('Scan history cleared successfully.');
    loadScanHistory();
  } catch (err) {
    showToast(`Error: ${err.message}`);
  }
}

/* ===================================================================
   Utilities
   =================================================================== */
function setButtonLoading(btn, isLoading) {
  if (!btn) return;
  const text = btn.querySelector('.btn-text');
  const spinner = btn.querySelector('.spinner');
  btn.disabled = isLoading;
  if (text) text.style.display = isLoading ? 'none' : 'inline';
  if (spinner) spinner.style.display = isLoading ? 'inline-block' : 'none';
}

function showToast(message) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = message;
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 4000);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
