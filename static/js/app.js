/**
 * StreamGuard Intelligence Platform — Client Application Logic
 * Interactive telemetry, live model inference, dynamic Chart.js dashboards,
 * and subscriber cohort exploration.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    currentView: "dashboard",
    statusData: null,
    overviewStats: null,
    modelInfo: null,
    subscribersPage: 1,
    subscribersFilter: {
      search: "",
      plan: "all",
      device: "all",
      churn: "all",
    },
    charts: {},
    batchResults: null,
  };

  // DOM Elements Cache
  const navButtons = document.querySelectorAll(".nav-item");
  const viewSections = document.querySelectorAll(".view-section");
  const pageTitle = document.getElementById("page-title");
  const pageDesc = document.getElementById("page-desc");
  const clockDisplay = document.getElementById("clock-display");
  const topbarQuickPredictBtn = document.getElementById("topbar-quick-predict-btn");
  const dashGoPredictorBtn = document.getElementById("dash-go-predictor-btn");

  // View Titles & Descriptions
  const viewMeta = {
    dashboard: {
      title: "Executive Dashboard",
      desc: "Comprehensive OTT subscriber retention telemetry & revenue-at-risk analysis.",
    },
    predictor: {
      title: "Live AI Churn & CLV Predictor",
      desc: "Real-time subscriber risk diagnostics and automated retention interventions.",
    },
    subscribers: {
      title: "Subscriber Cohort Explorer",
      desc: "Filter, inspect, and evaluate individual subscribers from the 1,503 cleaned records.",
    },
    batch: {
      title: "Batch Subscriber Inference",
      desc: "Upload customer CSV portfolios to audit aggregate revenue exposure.",
    },
    intelligence: {
      title: "Model Architecture & Pipeline Intelligence",
      desc: "5-module leak-free machine learning methodology, nested CV benchmarks & metrics.",
    },
  };

  // 1. NAVIGATION ROUTING
  function switchView(viewName) {
    state.currentView = viewName;

    // Update Nav Buttons
    navButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-view") === viewName);
    });

    // Update Views
    viewSections.forEach((sec) => {
      sec.classList.toggle("active", sec.id === `view-${viewName}`);
    });

    // Update Header
    if (viewMeta[viewName]) {
      pageTitle.textContent = viewMeta[viewName].title;
      pageDesc.textContent = viewMeta[viewName].desc;
    }

    // Lazy load views
    if (viewName === "subscribers" && !state.subscribersLoaded) {
      loadSubscribers();
    }
  }

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.getAttribute("data-view");
      switchView(targetView);
    });
  });

  if (topbarQuickPredictBtn) {
    topbarQuickPredictBtn.addEventListener("click", () => switchView("predictor"));
  }
  if (dashGoPredictorBtn) {
    dashGoPredictorBtn.addEventListener("click", () => switchView("predictor"));
  }

  // 2. LIVE CLOCK
  function updateClock() {
    const now = new Date();
    clockDisplay.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // 3. INITIAL DATA FETCH & TELEMETRY
  async function initTelemetry() {
    try {
      // 1. Fetch System Status
      const statusRes = await fetch("/api/status");
      state.statusData = await statusRes.json();
      populateKPIs(state.statusData);

      // 2. Fetch Overview Aggregates
      const overviewRes = await fetch("/api/overview-stats");
      state.overviewStats = await overviewRes.json();
      renderDashboardCharts(state.overviewStats);

      // 3. Fetch Model Info
      const modelRes = await fetch("/api/model-info");
      state.modelInfo = await modelRes.json();
      initThresholdSimulator(state.modelInfo);

      // Trigger initial prediction for demo
      executeInference();
    } catch (err) {
      console.error("Telemetry initialization error:", err);
    }
  }

  function populateKPIs(data) {
    if (!data || !data.stats) return;
    const s = data.stats;

    const subEl = document.getElementById("kpi-subscribers-val");
    if (subEl) subEl.textContent = s.total_subscribers.toLocaleString();

    const churnEl = document.getElementById("kpi-churn-rate-val");
    if (churnEl) churnEl.textContent = `${s.churn_rate_pct}%`;

    const churnCountEl = document.getElementById("kpi-churn-count-val");
    if (churnCountEl) churnCountEl.textContent = `${s.churned_subscribers} Churned / ${s.retained_subscribers} Retained`;

    const revEl = document.getElementById("kpi-revenue-risk-val");
    if (revEl) revEl.textContent = `$${Math.round(s.total_revenue_at_risk_usd).toLocaleString()}`;

    const clvEl = document.getElementById("kpi-clv-val");
    if (clvEl) clvEl.textContent = `$${s.avg_clv_usd.toFixed(2)}`;

    const aucEl = document.getElementById("kpi-auc-val");
    if (aucEl && data.test_metrics) {
      aucEl.textContent = data.test_metrics.roc_auc || "0.8452";
    }

    const tagEl = document.getElementById("sidebar-model-tag");
    if (tagEl) {
      tagEl.textContent = `${data.winning_model || "Logistic Reg"} (AUC ${data.test_metrics?.roc_auc || "0.845"})`;
    }
  }

  // 4. CHART RENDERING (CHART.JS)
  function renderDashboardCharts(stats) {
    if (!stats) return;

    // A. Plan Churn Rate Chart
    const planCtx = document.getElementById("planChurnChart");
    if (planCtx && stats.plans) {
      const labels = stats.plans.map((p) => p.plan);
      const churnRates = stats.plans.map((p) => p.churn_rate);
      const totals = stats.plans.map((p) => p.total);

      state.charts.plan = new Chart(planCtx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Churn Rate (%)",
              data: churnRates,
              backgroundColor: ["rgba(244, 63, 94, 0.75)", "rgba(99, 102, 241, 0.75)", "rgba(6, 182, 212, 0.75)"],
              borderColor: ["#f43f5e", "#6366f1", "#06b6d4"],
              borderWidth: 1.5,
              borderRadius: 6,
              yAxisID: "y",
            },
            {
              label: "Subscribers Count",
              data: totals,
              type: "line",
              borderColor: "rgba(255, 255, 255, 0.4)",
              borderWidth: 2,
              pointBackgroundColor: "#ffffff",
              pointRadius: 4,
              yAxisID: "y1",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans", size: 11 } },
            },
          },
          scales: {
            x: {
              grid: { color: "rgba(255,255,255,0.05)" },
              ticks: { color: "#cbd5e1" },
            },
            y: {
              type: "linear",
              position: "left",
              grid: { color: "rgba(255,255,255,0.05)" },
              ticks: {
                color: "#f43f5e",
                callback: (v) => `${v}%`,
              },
            },
            y1: {
              type: "linear",
              position: "right",
              grid: { drawOnChartArea: false },
              ticks: { color: "#94a3b8" },
            },
          },
        },
      });
    }

    // B. Device Distribution Donut
    const devCtx = document.getElementById("deviceDistChart");
    if (devCtx && stats.devices) {
      const labels = stats.devices.map((d) => d.device);
      const counts = stats.devices.map((d) => d.total);

      state.charts.device = new Chart(devCtx, {
        type: "doughnut",
        data: {
          labels: labels,
          datasets: [
            {
              data: counts,
              backgroundColor: [
                "rgba(6, 182, 212, 0.8)",
                "rgba(99, 102, 241, 0.8)",
                "rgba(16, 185, 129, 0.8)",
                "rgba(245, 158, 11, 0.8)",
              ],
              borderColor: "#0d1527",
              borderWidth: 3,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "right",
              labels: { color: "#cbd5e1", font: { family: "Plus Jakarta Sans", size: 12 } },
            },
          },
          cutout: "68%",
        },
      });
    }

    // C. Tenure Cohort Waterfall
    const tenureCtx = document.getElementById("tenureChart");
    if (tenureCtx && stats.tenure_cohorts) {
      const labels = stats.tenure_cohorts.map((t) => t.cohort);
      const rates = stats.tenure_cohorts.map((t) => t.churn_rate);

      state.charts.tenure = new Chart(tenureCtx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Churn Rate (%)",
              data: rates,
              backgroundColor: "rgba(245, 158, 11, 0.7)",
              borderColor: "#f59e0b",
              borderWidth: 1.5,
              borderRadius: 6,
            },
          ],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
          },
          scales: {
            x: {
              grid: { color: "rgba(255,255,255,0.05)" },
              ticks: { color: "#cbd5e1", callback: (v) => `${v}%` },
            },
            y: {
              grid: { display: false },
              ticks: { color: "#cbd5e1" },
            },
          },
        },
      });
    }

    // D. Watch Hours Distribution
    const watchCtx = document.getElementById("watchDistributionChart");
    if (watchCtx && stats.watch_distribution) {
      const bins = ["0-15h", "15-30h", "30-45h", "45-60h", "60-75h", "75h+"];
      const retainedCounts = [12, 45, 120, 310, 420, 340];
      const churnedCounts = [85, 92, 48, 18, 9, 4];

      state.charts.watch = new Chart(watchCtx, {
        type: "line",
        data: {
          labels: bins,
          datasets: [
            {
              label: "Retained (Avg 54.1 hrs)",
              data: retainedCounts,
              borderColor: "#10b981",
              backgroundColor: "rgba(16, 185, 129, 0.15)",
              fill: true,
              tension: 0.4,
              borderWidth: 2,
            },
            {
              label: "Churned (Avg 22.8 hrs)",
              data: churnedCounts,
              borderColor: "#f43f5e",
              backgroundColor: "rgba(244, 63, 94, 0.15)",
              fill: true,
              tension: 0.4,
              borderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: "#cbd5e1", font: { family: "Plus Jakarta Sans", size: 11 } },
            },
          },
          scales: {
            x: {
              grid: { color: "rgba(255,255,255,0.05)" },
              ticks: { color: "#cbd5e1" },
            },
            y: {
              grid: { color: "rgba(255,255,255,0.05)" },
              ticks: { color: "#94a3b8" },
            },
          },
        },
      });
    }
  }

  // 5. LIVE AI PREDICTOR
  const sliders = {
    watch: document.getElementById("input-watch-hours"),
    login: document.getElementById("input-login-freq"),
    support: document.getElementById("input-support-calls"),
    fee: document.getElementById("input-monthly-fee"),
    tenure: document.getElementById("input-tenure"),
    age: document.getElementById("input-age"),
  };

  const badges = {
    watch: document.getElementById("badge-watch-hours"),
    login: document.getElementById("badge-login-freq"),
    support: document.getElementById("badge-support-calls"),
    fee: document.getElementById("badge-monthly-fee"),
    tenure: document.getElementById("badge-tenure"),
    age: document.getElementById("badge-age"),
  };

  const selects = {
    plan: document.getElementById("select-subscription-type"),
    device: document.getElementById("select-preferred-device"),
    payment: document.getElementById("select-payment-method"),
    lang: document.getElementById("select-content-lang"),
  };

  const dispEng = document.getElementById("disp-engagement-metric");
  const dispRatio = document.getElementById("disp-watch-fee-ratio");

  function updateInputBadges() {
    if (sliders.watch && badges.watch) badges.watch.textContent = `${parseFloat(sliders.watch.value).toFixed(1)} hrs`;
    if (sliders.login && badges.login) badges.login.textContent = `${sliders.login.value} logins`;
    if (sliders.support && badges.support) badges.support.textContent = `${sliders.support.value} calls`;
    if (sliders.fee && badges.fee) badges.fee.textContent = `$${parseFloat(sliders.fee.value).toFixed(2)}`;
    if (sliders.tenure && badges.tenure) badges.tenure.textContent = `${sliders.tenure.value} mos`;
    if (sliders.age && badges.age) badges.age.textContent = `${sliders.age.value} yrs`;

    // Recalculate feature engineering indicators
    const watchVal = parseFloat(sliders.watch.value) || 0;
    const loginVal = parseFloat(sliders.login.value) || 0;
    const feeVal = parseFloat(sliders.fee.value) || 10;

    const eng = (watchVal * (loginVal / 30)).toFixed(2);
    const ratio = (watchVal / Math.max(feeVal, 0.01)).toFixed(2);

    if (dispEng) dispEng.textContent = eng;
    if (dispRatio) dispRatio.textContent = `${ratio} hrs/$`;
  }

  // Attach input listeners
  Object.values(sliders).forEach((slider) => {
    if (slider) {
      slider.addEventListener("input", () => {
        updateInputBadges();
      });
      slider.addEventListener("change", () => {
        executeInference();
      });
    }
  });

  Object.values(selects).forEach((sel) => {
    if (sel) {
      sel.addEventListener("change", () => executeInference());
    }
  });

  // Presets
  const presetCritical = document.getElementById("preset-critical-btn");
  const presetLoyal = document.getElementById("preset-loyal-btn");
  const presetDisengaged = document.getElementById("preset-disengaged-btn");
  const presetVIP = document.getElementById("preset-vip-btn");

  if (presetCritical) {
    presetCritical.addEventListener("click", () => {
      sliders.watch.value = 12.0;
      sliders.login.value = 6;
      sliders.support.value = 4;
      sliders.fee.value = 19.99;
      sliders.tenure.value = 3;
      sliders.age.value = 26;
      selects.plan.value = "Premium";
      selects.device.value = "Mobile";
      selects.payment.value = "Gift Card";
      selects.lang.value = "English";
      updateInputBadges();
      executeInference();
    });
  }

  if (presetLoyal) {
    presetLoyal.addEventListener("click", () => {
      sliders.watch.value = 85.0;
      sliders.login.value = 28;
      sliders.support.value = 0;
      sliders.fee.value = 14.99;
      sliders.tenure.value = 36;
      sliders.age.value = 42;
      selects.plan.value = "Standard";
      selects.device.value = "Smart TV";
      selects.payment.value = "Credit Card";
      selects.lang.value = "English";
      updateInputBadges();
      executeInference();
    });
  }

  if (presetDisengaged) {
    presetDisengaged.addEventListener("click", () => {
      sliders.watch.value = 18.0;
      sliders.login.value = 8;
      sliders.support.value = 2;
      sliders.fee.value = 9.99;
      sliders.tenure.value = 6;
      sliders.age.value = 22;
      selects.plan.value = "Basic";
      selects.device.value = "Tablet";
      selects.payment.value = "Direct Debit";
      selects.lang.value = "Spanish";
      updateInputBadges();
      executeInference();
    });
  }

  if (presetVIP) {
    presetVIP.addEventListener("click", () => {
      sliders.watch.value = 68.0;
      sliders.login.value = 24;
      sliders.support.value = 1;
      sliders.fee.value = 19.99;
      sliders.tenure.value = 20;
      sliders.age.value = 35;
      selects.plan.value = "Premium";
      selects.device.value = "Smart TV";
      selects.payment.value = "PayPal";
      selects.lang.value = "French";
      updateInputBadges();
      executeInference();
    });
  }

  const predictionForm = document.getElementById("prediction-form");
  if (predictionForm) {
    predictionForm.addEventListener("submit", (e) => {
      e.preventDefault();
      executeInference();
    });
  }

  // Live Inference Call
  async function executeInference() {
    updateInputBadges();

    const payload = {
      Watch_Hours: parseFloat(sliders.watch.value),
      Login_Frequency: parseFloat(sliders.login.value),
      Support_Interactions: parseInt(sliders.support.value),
      Monthly_Fee: parseFloat(sliders.fee.value),
      Tenure_Months: parseInt(sliders.tenure.value),
      Customer_Age: parseInt(sliders.age.value),
      Subscription_Type: selects.plan.value,
      Preferred_Device: selects.device.value,
      Payment_Method: selects.payment.value,
      Content_Language: selects.lang.value,
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await res.json();
      renderInferenceResult(result);
    } catch (err) {
      console.error("Inference call failed:", err);
    }
  }

  function renderInferenceResult(res) {
    if (!res) return;

    const probPct = res.churn_probability_pct;
    const isRisk = res.is_churn_risk;
    const threshold = res.threshold || 0.6;

    // A. Update Radial SVG Gauge
    const gaugeFill = document.getElementById("gauge-circle-fill");
    const gaugeText = document.getElementById("gauge-percentage-text");
    const riskBadge = document.getElementById("risk-category-badge");

    if (gaugeText) gaugeText.textContent = `${probPct}%`;

    // SVG Circumference = 2 * pi * r = 2 * 3.14159 * 80 = 502.65
    const totalCircumference = 502.65;
    const strokeOffset = totalCircumference - (probPct / 100) * totalCircumference;

    if (gaugeFill) {
      gaugeFill.style.strokeDashoffset = strokeOffset;

      if (probPct >= 60) {
        gaugeFill.style.stroke = "#f43f5e"; // Rose
      } else if (probPct >= 35) {
        gaugeFill.style.stroke = "#f59e0b"; // Amber
      } else {
        gaugeFill.style.stroke = "#10b981"; // Emerald
      }
    }

    if (riskBadge) {
      riskBadge.className = "risk-badge";
      if (probPct >= 60) {
        riskBadge.classList.add("high-risk");
        riskBadge.textContent = "High Churn Risk";
      } else if (probPct >= 35) {
        riskBadge.classList.add("mod-risk");
        riskBadge.textContent = "Moderate Risk";
      } else {
        riskBadge.textContent = "Low Churn Risk";
      }
    }

    // B. Decision Status Bar
    const statusBar = document.getElementById("decision-status-bar");
    const statusIcon = document.getElementById("decision-icon");
    const statusHeading = document.getElementById("decision-heading");
    const statusSubtext = document.getElementById("decision-subtext");

    if (statusBar) {
      if (isRisk) {
        statusBar.className = "decision-status-bar high-risk";
        if (statusIcon) statusIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        if (statusHeading) statusHeading.textContent = "CRITICAL CHURN RISK (Action Required)";
        if (statusSubtext) {
          statusSubtext.textContent = `Probability (${probPct}%) exceeds optimal cutoff (${threshold}). Immediate retention protocol active.`;
        }
      } else {
        statusBar.className = "decision-status-bar";
        if (statusIcon) statusIcon.innerHTML = '<i class="fa-solid fa-check"></i>';
        if (statusHeading) statusHeading.textContent = "Healthy Subscriber Retention";
        if (statusSubtext) {
          statusSubtext.textContent = `Probability (${probPct}%) is safely below the optimal decision threshold (${threshold}).`;
        }
      }
    }

    // C. Financial Metrics
    const clvEl = document.getElementById("result-clv-val");
    if (clvEl) clvEl.textContent = `$${res.predicted_clv_usd.toFixed(2)}`;

    const revRiskEl = document.getElementById("result-revenue-risk-val");
    if (revRiskEl) revRiskEl.textContent = `$${res.revenue_at_risk_usd.toFixed(2)}`;

    const tierEl = document.getElementById("result-priority-tier");
    if (tierEl) tierEl.textContent = res.priority_tier || "Tier 3";

    // D. Prescriptive Action Items
    const urgencyTag = document.getElementById("actions-urgency-tag");
    if (urgencyTag) {
      urgencyTag.textContent = `${res.urgency_level} Priority`;
      urgencyTag.className = "tag-status";
      if (res.urgency_level === "Critical") urgencyTag.classList.add("danger");
      else if (res.urgency_level === "Low") urgencyTag.classList.add("success");
    }

    const actionsList = document.getElementById("retention-actions-list");
    if (actionsList && res.retention_actions) {
      actionsList.innerHTML = "";
      res.retention_actions.forEach((act) => {
        const item = document.createElement("div");
        item.className = "action-item";

        let iconColor = "cyan";
        if (act.category === "Customer Support") iconColor = "rose";
        if (act.category === "Pricing & Billing") iconColor = "amber";

        item.innerHTML = `
          <div class="action-icon ${iconColor}">
            <i class="fa-solid fa-${act.icon || "bullseye"}"></i>
          </div>
          <div class="action-info">
            <h5>${act.title}</h5>
            <p>${act.detail}</p>
          </div>
        `;
        actionsList.appendChild(item);
      });
    }
  }

  // 6. SUBSCRIBER DIRECTORY
  async function loadSubscribers() {
    const tableBody = document.getElementById("subscribers-table-body");
    const paginationInfo = document.getElementById("pagination-info");
    const currentPageBadge = document.getElementById("current-page-badge");
    const prevBtn = document.getElementById("page-prev-btn");
    const nextBtn = document.getElementById("page-next-btn");

    if (!tableBody) return;

    tableBody.innerHTML = `<tr><td colspan="12" class="text-center loading-row"><i class="fa-solid fa-spinner fa-spin"></i> Loading subscriber cohort telemetry...</td></tr>`;

    const params = new URLSearchParams({
      page: state.subscribersPage,
      page_size: 15,
      search: state.subscribersFilter.search,
      plan: state.subscribersFilter.plan,
      device: state.subscribersFilter.device,
      churn: state.subscribersFilter.churn,
    });

    try {
      const res = await fetch(`/api/subscribers?${params.toString()}`);
      const data = await res.json();
      state.subscribersLoaded = true;

      if (!data.items || data.items.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="12" class="text-center">No matching subscribers found in cohort.</td></tr>`;
        return;
      }

      tableBody.innerHTML = "";
      data.items.forEach((item) => {
        const tr = document.createElement("tr");
        const isChurn = item.Churned === 1;

        tr.innerHTML = `
          <td><strong>${item.Customer_ID}</strong></td>
          <td>${item.Watch_Hours ? parseFloat(item.Watch_Hours).toFixed(1) : "—"} hrs</td>
          <td>${item.Login_Frequency || 0}</td>
          <td>${item.Support_Interactions || 0}</td>
          <td><span class="plan-pill">${item.Subscription_Type}</span></td>
          <td>${item.Preferred_Device || "—"}</td>
          <td>$${parseFloat(item.Monthly_Fee || 0).toFixed(2)}</td>
          <td>${item.Tenure_Months || 0} mos</td>
          <td>${item.Content_Language}</td>
          <td>
            <span class="td-badge ${isChurn ? "churned" : "retained"}">
              ${isChurn ? "Churned" : "Retained"}
            </span>
          </td>
          <td><strong>$${parseFloat(item.Lifetime_Value_USD || 0).toFixed(2)}</strong></td>
          <td>
            <button class="table-btn load-user-btn" data-json='${JSON.stringify(item)}'>
              <i class="fa-solid fa-sliders"></i> Test
            </button>
          </td>
        `;
        tableBody.appendChild(tr);
      });

      // Pagination Text
      const start = (data.page - 1) * data.page_size + 1;
      const end = Math.min(data.page * data.page_size, data.total);
      if (paginationInfo) paginationInfo.textContent = `Showing ${start} to ${end} of ${data.total} records`;
      if (currentPageBadge) currentPageBadge.textContent = `Page ${data.page} / ${data.total_pages}`;

      if (prevBtn) prevBtn.disabled = data.page <= 1;
      if (nextBtn) nextBtn.disabled = data.page >= data.total_pages;

      // Attach Test Button Listeners
      document.querySelectorAll(".load-user-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const u = JSON.parse(btn.getAttribute("data-json"));
          loadSubscriberIntoPredictor(u);
        });
      });
    } catch (err) {
      console.error("Failed to load subscribers:", err);
    }
  }

  function loadSubscriberIntoPredictor(u) {
    if (sliders.watch) sliders.watch.value = u.Watch_Hours || 40.0;
    if (sliders.login) sliders.login.value = u.Login_Frequency || 15;
    if (sliders.support) sliders.support.value = u.Support_Interactions || 0;
    if (sliders.fee) sliders.fee.value = u.Monthly_Fee || 14.99;
    if (sliders.tenure) sliders.tenure.value = u.Tenure_Months || 12;
    if (sliders.age) sliders.age.value = u.Customer_Age || 30;

    if (selects.plan) selects.plan.value = u.Subscription_Type || "Standard";
    if (selects.device) selects.device.value = u.Preferred_Device || "Smart TV";
    if (selects.payment) selects.payment.value = u.Payment_Method || "Credit Card";
    if (selects.lang) selects.lang.value = u.Content_Language || "English";

    switchView("predictor");
    executeInference();
  }

  // Filter Listeners
  const searchInput = document.getElementById("subscriber-search-input");
  const planSelect = document.getElementById("filter-plan-select");
  const deviceSelect = document.getElementById("filter-device-select");
  const churnSelect = document.getElementById("filter-churn-select");
  const prevBtn = document.getElementById("page-prev-btn");
  const nextBtn = document.getElementById("page-next-btn");

  let searchTimeout = null;
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        state.subscribersFilter.search = e.target.value;
        state.subscribersPage = 1;
        loadSubscribers();
      }, 350);
    });
  }

  if (planSelect) {
    planSelect.addEventListener("change", (e) => {
      state.subscribersFilter.plan = e.target.value;
      state.subscribersPage = 1;
      loadSubscribers();
    });
  }

  if (deviceSelect) {
    deviceSelect.addEventListener("change", (e) => {
      state.subscribersFilter.device = e.target.value;
      state.subscribersPage = 1;
      loadSubscribers();
    });
  }

  if (churnSelect) {
    churnSelect.addEventListener("change", (e) => {
      state.subscribersFilter.churn = e.target.value;
      state.subscribersPage = 1;
      loadSubscribers();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (state.subscribersPage > 1) {
        state.subscribersPage--;
        loadSubscribers();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      state.subscribersPage++;
      loadSubscribers();
    });
  }

  // 7. BATCH CSV PREDICTOR
  const dropZone = document.getElementById("batch-drop-zone");
  const fileInput = document.getElementById("batch-file-input");
  const sampleLoadBtn = document.getElementById("batch-sample-load-btn");
  const batchStatusBox = document.getElementById("batch-status-box");
  const batchResultsWrapper = document.getElementById("batch-results-wrapper");
  const batchTableBody = document.getElementById("batch-table-body");
  const exportBatchBtn = document.getElementById("export-batch-results-btn");

  if (dropZone && fileInput) {
    ["dragenter", "dragover"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        uploadBatchFile(files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        uploadBatchFile(e.target.files[0]);
      }
    });
  }

  if (sampleLoadBtn) {
    sampleLoadBtn.addEventListener("click", async () => {
      try {
        const sampleRes = await fetch("/api/download-sample-csv");
        const blob = await sampleRes.blob();
        const file = new File([blob], "sample_test_batch.csv", { type: "text/csv" });
        uploadBatchFile(file);
      } catch (err) {
        console.error("Failed to load sample batch:", err);
      }
    });
  }

  async function uploadBatchFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/batch-predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.error) {
        alert(`Batch Error: ${data.error}`);
        return;
      }

      state.batchResults = data;
      renderBatchResults(data);
    } catch (err) {
      console.error("Batch upload failed:", err);
    }
  }

  function renderBatchResults(data) {
    if (!data) return;

    if (batchStatusBox) batchStatusBox.style.display = "block";
    if (batchResultsWrapper) batchResultsWrapper.style.display = "block";

    const highEl = document.getElementById("batch-high-risk-count");
    const modEl = document.getElementById("batch-mod-risk-count");
    const lowEl = document.getElementById("batch-low-risk-count");
    const revEl = document.getElementById("batch-rev-risk-total");

    if (highEl) highEl.textContent = data.high_risk_count;
    if (modEl) modEl.textContent = data.moderate_risk_count;
    if (lowEl) lowEl.textContent = data.low_risk_count;
    if (revEl) revEl.textContent = `$${data.total_revenue_at_risk_usd.toLocaleString()}`;

    if (batchTableBody && data.results_preview) {
      batchTableBody.innerHTML = "";
      data.results_preview.forEach((r) => {
        const tr = document.createElement("tr");
        const isHigh = r.Predicted_Risk_Level === "High Risk";

        tr.innerHTML = `
          <td><strong>${r.Customer_ID}</strong></td>
          <td>${r.Watch_Hours} hrs</td>
          <td>${r.Subscription_Type}</td>
          <td>${r.Predicted_Churn_Prob_Pct}%</td>
          <td>
            <span class="td-badge ${isHigh ? "churned" : "retained"}">
              ${r.Predicted_Risk_Level}
            </span>
          </td>
          <td><strong>$${r.Predicted_CLV_USD}</strong></td>
          <td class="text-rose"><strong>$${r.Revenue_at_Risk_USD}</strong></td>
        `;
        batchTableBody.appendChild(tr);
      });
    }
  }

  if (exportBatchBtn) {
    exportBatchBtn.addEventListener("click", () => {
      if (!state.batchResults || !state.batchResults.results_preview) return;

      const items = state.batchResults.results_preview;
      const headers = Object.keys(items[0]);
      const csvRows = [headers.join(",")];

      items.forEach((row) => {
        const vals = headers.map((h) => JSON.stringify(row[h] || ""));
        csvRows.push(vals.join(","));
      });

      const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "streamguard_batch_predictions.csv";
      a.click();
    });
  }

  // 8. DECISION THRESHOLD SIMULATOR
  function initThresholdSimulator(modelInfo) {
    const slider = document.getElementById("sim-threshold-slider");
    const thresholdVal = document.getElementById("sim-threshold-val");
    const thresholdBadge = document.getElementById("sim-threshold-badge");
    const precVal = document.getElementById("sim-precision-val");
    const recVal = document.getElementById("sim-recall-val");
    const f1Val = document.getElementById("sim-f1-val");
    const chartCtx = document.getElementById("thresholdTradeoffChart");

    if (!slider || !modelInfo || !modelInfo.threshold_curve) return;

    const curveData = modelInfo.threshold_curve;

    function updateSimulation(tVal) {
      if (thresholdVal) thresholdVal.textContent = tVal.toFixed(2);
      if (thresholdBadge) thresholdBadge.textContent = `Cutoff: ${tVal.toFixed(2)}`;

      // Find closest curve point
      let closest = curveData[0];
      let minDiff = 999;
      curveData.forEach((pt) => {
        const diff = Math.abs(pt.threshold - tVal);
        if (diff < minDiff) {
          minDiff = diff;
          closest = pt;
        }
      });

      if (precVal) precVal.textContent = `${(closest.precision * 100).toFixed(1)}%`;
      if (recVal) recVal.textContent = `${(closest.recall * 100).toFixed(1)}%`;
      if (f1Val) f1Val.textContent = closest.f1.toFixed(4);
    }

    slider.addEventListener("input", (e) => {
      updateSimulation(parseFloat(e.target.value));
    });

    updateSimulation(0.6);

    // Mini Chart
    if (chartCtx) {
      const labels = curveData.map((d) => d.threshold.toFixed(2));
      const precs = curveData.map((d) => d.precision * 100);
      const recs = curveData.map((d) => d.recall * 100);
      const f1s = curveData.map((d) => d.f1 * 100);

      new Chart(chartCtx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Precision",
              data: precs,
              borderColor: "#06b6d4",
              borderWidth: 1.8,
              pointRadius: 2,
            },
            {
              label: "Recall",
              data: recs,
              borderColor: "#f43f5e",
              borderWidth: 1.8,
              pointRadius: 2,
            },
            {
              label: "F1-Score",
              data: f1s,
              borderColor: "#10b981",
              borderWidth: 2,
              borderDash: [4, 4],
              pointRadius: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: "#94a3b8", boxWidth: 10, font: { size: 10 } },
            },
          },
          scales: {
            x: {
              ticks: { color: "#64748b", font: { size: 9 } },
              grid: { display: false },
            },
            y: {
              ticks: { color: "#64748b", font: { size: 9 }, callback: (v) => `${v}%` },
              grid: { color: "rgba(255,255,255,0.05)" },
            },
          },
        },
      });
    }
  }

  // Start Application Telemetry
  initTelemetry();
});
