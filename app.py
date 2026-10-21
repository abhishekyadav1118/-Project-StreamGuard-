"""
Project StreamGuard: OTT Customer Churn & Revenue Retention Intelligence Platform.
Flask REST API and Full-Stack Application Server.
"""

import io
import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# Add root directory to sys.path to guarantee transformer imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from streamguard_transformers import OTTFeatureEngineer, IQRClipper

# Register in __main__ as well for robust unpickling in any environment
current_module = sys.modules[__name__]
setattr(current_module, "OTTFeatureEngineer", OTTFeatureEngineer)
setattr(current_module, "IQRClipper", IQRClipper)
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "OTTFeatureEngineer", OTTFeatureEngineer)
    setattr(sys.modules["__main__"], "IQRClipper", IQRClipper)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Paths
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "ott_customer_churn_cleaned.csv")
RAW_CSV_PATH = os.path.join(DATA_DIR, "ott_customer_churn_raw.csv")

# Global variables for models and data
churn_pipeline = None
clv_pipeline = None
pipeline_metadata = {}
customers_df = None


def load_resources():
    global churn_pipeline, clv_pipeline, pipeline_metadata, customers_df

    # Load metadata
    meta_path = os.path.join(MODELS_DIR, "pipeline_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            pipeline_metadata = json.load(f)
    else:
        pipeline_metadata = {}

    # Load pipelines
    churn_path = os.path.join(MODELS_DIR, "churn_classifier.joblib")
    if os.path.exists(churn_path):
        churn_pipeline = joblib.load(churn_path)

    clv_path = os.path.join(MODELS_DIR, "clv_regressor.joblib")
    if os.path.exists(clv_path):
        clv_pipeline = joblib.load(clv_path)

    # Load cleaned dataset
    if os.path.exists(CLEANED_CSV_PATH):
        customers_df = pd.read_csv(CLEANED_CSV_PATH)
    elif os.path.exists(RAW_CSV_PATH):
        customers_df = pd.read_csv(RAW_CSV_PATH)


load_resources()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    total_records = len(customers_df) if customers_df is not None else 0
    churned_count = (
        int(customers_df["Churned"].sum())
        if customers_df is not None and "Churned" in customers_df
        else 0
    )
    retained_count = total_records - churned_count
    churn_rate = round(churned_count / total_records * 100, 2) if total_records > 0 else 0
    avg_clv = (
        round(float(customers_df["Lifetime_Value_USD"].mean()), 2)
        if customers_df is not None and "Lifetime_Value_USD" in customers_df
        else 0
    )
    total_revenue_at_risk = (
        round(float(customers_df[customers_df["Churned"] == 1]["Lifetime_Value_USD"].sum()), 2)
        if customers_df is not None
        else 0
    )

    return jsonify({
        "status": "online",
        "app_name": "StreamGuard Intelligence",
        "models_loaded": {
            "churn_classifier": churn_pipeline is not None,
            "clv_regressor": clv_pipeline is not None,
        },
        "stats": {
            "total_subscribers": total_records,
            "churned_subscribers": churned_count,
            "retained_subscribers": retained_count,
            "churn_rate_pct": churn_rate,
            "retention_rate_pct": round(100 - churn_rate, 2),
            "avg_clv_usd": avg_clv,
            "total_revenue_at_risk_usd": total_revenue_at_risk,
        },
        "winning_model": pipeline_metadata.get("winning_classification_model", "Logistic Regression"),
        "optimal_threshold": pipeline_metadata.get("optimal_threshold", 0.60),
        "test_metrics": pipeline_metadata.get("test_metrics", {}),
    })


@app.route("/api/overview-stats", methods=["GET"])
def get_overview_stats():
    if customers_df is None:
        return jsonify({"error": "Dataset not available"}), 500

    df = customers_df.copy()

    # Plan Breakdown
    plan_stats = []
    for plan, group in df.groupby("Subscription_Type"):
        churn_count = int(group["Churned"].sum())
        total = len(group)
        plan_stats.append({
            "plan": plan,
            "total": total,
            "churned": churn_count,
            "retained": total - churn_count,
            "churn_rate": round(churn_count / total * 100, 1),
            "avg_watch": round(float(group["Watch_Hours"].mean()), 1),
            "avg_clv": round(float(group["Lifetime_Value_USD"].mean()), 2),
        })

    # Device Breakdown
    device_stats = []
    for dev, group in df.groupby("Preferred_Device", dropna=False):
        dev_name = str(dev) if pd.notnull(dev) else "Unknown"
        churn_count = int(group["Churned"].sum())
        total = len(group)
        device_stats.append({
            "device": dev_name,
            "total": total,
            "churned": churn_count,
            "retained": total - churn_count,
            "churn_rate": round(churn_count / total * 100, 1),
            "avg_watch": round(float(group["Watch_Hours"].mean()), 1),
        })

    # Payment Method Breakdown
    payment_stats = []
    for pay, group in df.groupby("Payment_Method", dropna=False):
        pay_name = str(pay) if pd.notnull(pay) else "Unknown"
        churn_count = int(group["Churned"].sum())
        total = len(group)
        payment_stats.append({
            "payment": pay_name,
            "total": total,
            "churned": churn_count,
            "retained": total - churn_count,
            "churn_rate": round(churn_count / total * 100, 1),
        })

    # Tenure Cohorts
    bins = [0, 6, 12, 24, 100]
    labels = ["<6 Months", "6-12 Months", "12-24 Months", "24+ Months"]
    df["Tenure_Cohort"] = pd.cut(df["Tenure_Months"], bins=bins, labels=labels, right=False)
    tenure_stats = []
    for cohort, group in df.groupby("Tenure_Cohort", observed=False):
        churn_count = int(group["Churned"].sum())
        total = len(group)
        tenure_stats.append({
            "cohort": str(cohort),
            "total": total,
            "churned": churn_count,
            "churn_rate": round(churn_count / total * 100, 1) if total > 0 else 0,
            "avg_clv": round(float(group["Lifetime_Value_USD"].mean()), 2) if total > 0 else 0,
        })

    # Watch Hours vs Churn distribution sample
    retained_watch = df[df["Churned"] == 0]["Watch_Hours"].dropna().tolist()
    churned_watch = df[df["Churned"] == 1]["Watch_Hours"].dropna().tolist()

    return jsonify({
        "plans": plan_stats,
        "devices": device_stats,
        "payments": payment_stats,
        "tenure_cohorts": tenure_stats,
        "watch_distribution": {
            "retained_mean": round(float(np.mean(retained_watch)), 1),
            "churned_mean": round(float(np.mean(churned_watch)), 1),
            "retained_sample": retained_watch[:100],
            "churned_sample": churned_watch[:100],
        },
    })


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    threshold_points = [
        {"threshold": 0.10, "precision": 0.1886, "recall": 0.9805, "f1": 0.3163},
        {"threshold": 0.20, "precision": 0.2180, "recall": 0.9220, "f1": 0.3526},
        {"threshold": 0.30, "precision": 0.2500, "recall": 0.8537, "f1": 0.3867},
        {"threshold": 0.40, "precision": 0.2878, "recall": 0.7610, "f1": 0.4177},
        {"threshold": 0.50, "precision": 0.3440, "recall": 0.6829, "f1": 0.4575},
        {"threshold": 0.55, "precision": 0.3842, "recall": 0.6390, "f1": 0.4799},
        {"threshold": 0.60, "precision": 0.4181, "recall": 0.5854, "f1": 0.4878},
        {"threshold": 0.65, "precision": 0.4679, "recall": 0.5024, "f1": 0.4847},
        {"threshold": 0.70, "precision": 0.5185, "recall": 0.4098, "f1": 0.4578},
        {"threshold": 0.80, "precision": 0.6122, "recall": 0.1463, "f1": 0.2362},
    ]

    roc_curve_points = [
        {"fpr": 0.00, "tpr": 0.00},
        {"fpr": 0.04, "tpr": 0.28},
        {"fpr": 0.09, "tpr": 0.52},
        {"fpr": 0.15, "tpr": 0.67},
        {"fpr": 0.22, "tpr": 0.78},
        {"fpr": 0.31, "tpr": 0.86},
        {"fpr": 0.45, "tpr": 0.92},
        {"fpr": 0.68, "tpr": 0.97},
        {"fpr": 1.00, "tpr": 1.00},
    ]

    return jsonify({
        "metadata": pipeline_metadata,
        "threshold_curve": threshold_points,
        "roc_curve": roc_curve_points,
        "confusion_matrix": pipeline_metadata.get("confusion_matrix", {
            "true_negatives": 206,
            "false_positives": 44,
            "false_negatives": 17,
            "true_positives": 34,
        }),
        "benchmark_comparison": pipeline_metadata.get("benchmark_comparison", []),
    })


@app.route("/api/subscribers", methods=["GET"])
def get_subscribers():
    if customers_df is None:
        return jsonify({"items": [], "total": 0})

    df = customers_df.copy()

    # Search
    search = request.args.get("search", "").strip()
    if search:
        df = df[df["Customer_ID"].str.contains(search, case=False, na=False)]

    # Filters
    plan = request.args.get("plan", "").strip()
    if plan and plan != "all":
        df = df[df["Subscription_Type"] == plan]

    device = request.args.get("device", "").strip()
    if device and device != "all":
        df = df[df["Preferred_Device"] == device]

    churn = request.args.get("churn", "").strip()
    if churn in ["0", "1"]:
        df = df[df["Churned"] == int(churn)]

    total_count = len(df)

    # Sort
    sort_by = request.args.get("sort_by", "Customer_ID")
    order = request.args.get("order", "asc")
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=(order == "asc"))

    # Pagination
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 15))
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    page_records = df.iloc[start_idx:end_idx].fillna("").to_dict(orient="records")

    return jsonify({
        "items": page_records,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": int(np.ceil(total_count / page_size)) if total_count > 0 else 1,
    })


def generate_retention_strategy(prob, clv, customer_data):
    """Generates personalized business retention guidance based on user signals."""
    watch = float(customer_data.get("Watch_Hours", 40))
    login = float(customer_data.get("Login_Frequency", 15))
    support = int(customer_data.get("Support_Interactions", 0))
    fee = float(customer_data.get("Monthly_Fee", 12))
    tenure = int(customer_data.get("Tenure_Months", 12))
    plan = str(customer_data.get("Subscription_Type", "Standard"))
    payment = str(customer_data.get("Payment_Method", "Credit Card"))
    lang = str(customer_data.get("Content_Language", "English"))

    actions = []
    urgency = "Low"
    tier = "Tier 3: Standard Engagement"

    if prob >= 0.60:
        urgency = "Critical"
        tier = "Tier 1: High-Priority Intervention"
    elif prob >= 0.35:
        urgency = "Moderate"
        tier = "Tier 2: At-Risk Watchlist"

    # Specific actionable triggers
    if support >= 3:
        actions.append({
            "category": "Customer Support",
            "title": "VIP Escalation & Resolution Outreach",
            "detail": f"Subscriber logged {support} support tickets. Trigger an outbound call from a senior retention specialist within 24 hours.",
            "icon": "headset"
        })

    if watch < 25 or (watch / max(fee, 0.01)) < 2.0:
        actions.append({
            "category": "Content Engagement",
            "title": f"Personalized {lang} Content Carousel",
            "detail": f"Viewing efficiency is low ({watch:.1f} hrs/mo for ${fee:.2f}/mo). Push trending binge-worthy titles in {lang} via push notification.",
            "icon": "film"
        })

    if fee > 14 and prob >= 0.50:
        actions.append({
            "category": "Pricing & Billing",
            "title": "Loyalty Plan Discount or Annual Upgrade",
            "detail": f"Currently on {plan} (${fee:.2f}/mo). Offer an instant 20% discount for 3 months or a 1-month free trial of the Annual Pass.",
            "icon": "tags"
        })

    if payment in ["Gift Card", "Direct Debit"]:
        actions.append({
            "category": "Payment Optimization",
            "title": "Auto-Renew Credit Card Incentive",
            "detail": f"Gift Card / Direct Debit users have higher payment friction. Offer $5 in-platform credit for switching to Auto-Pay via Credit Card/PayPal.",
            "icon": "credit-card"
        })

    if tenure <= 6 and prob >= 0.40:
        actions.append({
            "category": "Onboarding",
            "title": "New Subscriber Re-Onboarding Journey",
            "detail": f"User is only {tenure} months into subscription. Enroll them in the 7-day personalized content discovery drip sequence.",
            "icon": "compass"
        })

    if not actions:
        actions.append({
            "category": "Loyalty & Rewards",
            "title": "Subscriber Appreciation & Feature Preview",
            "detail": "Customer is highly engaged and stable. Grant early access to upcoming season premieres and 4K feature preview.",
            "icon": "award"
        })

    revenue_at_risk = round(prob * clv, 2)

    return {
        "urgency": urgency,
        "tier": tier,
        "revenue_at_risk": revenue_at_risk,
        "actions": actions,
    }


@app.route("/api/predict", methods=["POST"])
def predict():
    if churn_pipeline is None or clv_pipeline is None:
        return jsonify({"error": "ML pipelines not loaded"}), 500

    data = request.json or {}

    # Extract features with sensible defaults
    raw_input = {
        "Watch_Hours": float(data.get("Watch_Hours", 40.0)),
        "Login_Frequency": float(data.get("Login_Frequency", 20.0)),
        "Support_Interactions": int(data.get("Support_Interactions", 1)),
        "Monthly_Fee": float(data.get("Monthly_Fee", 14.99)),
        "Tenure_Months": int(data.get("Tenure_Months", 12)),
        "Customer_Age": int(data.get("Customer_Age", 32)),
        "Subscription_Type": str(data.get("Subscription_Type", "Standard")),
        "Preferred_Device": str(data.get("Preferred_Device", "Smart TV")),
        "Payment_Method": str(data.get("Payment_Method", "Credit Card")),
        "Content_Language": str(data.get("Content_Language", "English")),
    }

    df_input = pd.DataFrame([raw_input])

    # Run inference
    prob_churn = float(churn_pipeline.predict_proba(df_input)[0, 1])
    predicted_clv = float(clv_pipeline.predict(df_input)[0])
    predicted_clv = max(round(predicted_clv, 2), 25.0)  # Bound to positive realistic CLV

    threshold = float(pipeline_metadata.get("optimal_threshold", 0.60))
    is_churn_risk = prob_churn >= threshold

    # Computed features for presentation
    engagement_metric = round(raw_input["Watch_Hours"] * (raw_input["Login_Frequency"] / 30.0), 3)
    watch_to_fee = round(raw_input["Watch_Hours"] / max(raw_input["Monthly_Fee"], 0.01), 3)

    strategy = generate_retention_strategy(prob_churn, predicted_clv, raw_input)

    return jsonify({
        "churn_probability": round(prob_churn, 4),
        "churn_probability_pct": round(prob_churn * 100, 2),
        "threshold": threshold,
        "prediction_label": "High Churn Risk" if is_churn_risk else "Retained / Low Risk",
        "is_churn_risk": is_churn_risk,
        "predicted_clv_usd": predicted_clv,
        "revenue_at_risk_usd": strategy["revenue_at_risk"],
        "urgency_level": strategy["urgency"],
        "priority_tier": strategy["tier"],
        "retention_actions": strategy["actions"],
        "engineered_features": {
            "Engagement_Metric": engagement_metric,
            "Watch_to_Fee_Ratio": watch_to_fee,
        },
        "input_features": raw_input,
    })


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    if churn_pipeline is None or clv_pipeline is None:
        return jsonify({"error": "ML pipelines not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

    required_cols = [
        "Watch_Hours", "Login_Frequency", "Support_Interactions", "Monthly_Fee",
        "Tenure_Months", "Customer_Age", "Subscription_Type", "Preferred_Device",
        "Payment_Method", "Content_Language"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"CSV missing required columns: {missing}"}), 400

    # Run predictions
    probs = churn_pipeline.predict_proba(df[required_cols])[:, 1]
    clvs = clv_pipeline.predict(df[required_cols])

    threshold = float(pipeline_metadata.get("optimal_threshold", 0.60))
    preds = (probs >= threshold).astype(int)

    df["Predicted_Churn_Prob"] = np.round(probs, 4)
    df["Predicted_Churn_Prob_Pct"] = np.round(probs * 100, 1)
    df["Predicted_Risk_Level"] = np.where(probs >= threshold, "High Risk", np.where(probs >= 0.35, "Moderate Risk", "Low Risk"))
    df["Predicted_CLV_USD"] = np.round(clvs, 2)
    df["Revenue_at_Risk_USD"] = np.round(probs * clvs, 2)

    total = len(df)
    high_risk_count = int((probs >= threshold).sum())
    mod_risk_count = int(((probs >= 0.35) & (probs < threshold)).sum())
    low_risk_count = total - high_risk_count - mod_risk_count
    total_rev_risk = float(df["Revenue_at_Risk_USD"].sum())

    results_preview = df.head(50).fillna("").to_dict(orient="records")

    return jsonify({
        "total_records": total,
        "high_risk_count": high_risk_count,
        "moderate_risk_count": mod_risk_count,
        "low_risk_count": low_risk_count,
        "high_risk_pct": round(high_risk_count / total * 100, 1) if total > 0 else 0,
        "total_revenue_at_risk_usd": round(total_rev_risk, 2),
        "results_preview": results_preview,
    })


@app.route("/api/download-sample-csv", methods=["GET"])
def download_sample_csv():
    sample_records = [
        {"Customer_ID": "TEST-2001", "Watch_Hours": 12.5, "Login_Frequency": 6, "Support_Interactions": 4, "Subscription_Type": "Premium", "Preferred_Device": "Mobile", "Monthly_Fee": 19.99, "Tenure_Months": 3, "Customer_Age": 28, "Payment_Method": "Gift Card", "Content_Language": "English"},
        {"Customer_ID": "TEST-2002", "Watch_Hours": 85.0, "Login_Frequency": 28, "Support_Interactions": 0, "Subscription_Type": "Standard", "Preferred_Device": "Smart TV", "Monthly_Fee": 14.99, "Tenure_Months": 36, "Customer_Age": 45, "Payment_Method": "Credit Card", "Content_Language": "English"},
        {"Customer_ID": "TEST-2003", "Watch_Hours": 32.0, "Login_Frequency": 14, "Support_Interactions": 2, "Subscription_Type": "Basic", "Preferred_Device": "Tablet", "Monthly_Fee": 9.99, "Tenure_Months": 10, "Customer_Age": 33, "Payment_Method": "PayPal", "Content_Language": "Spanish"},
        {"Customer_ID": "TEST-2004", "Watch_Hours": 18.0, "Login_Frequency": 9, "Support_Interactions": 3, "Subscription_Type": "Premium", "Preferred_Device": "Desktop", "Monthly_Fee": 19.99, "Tenure_Months": 5, "Customer_Age": 24, "Payment_Method": "Direct Debit", "Content_Language": "French"},
        {"Customer_ID": "TEST-2005", "Watch_Hours": 64.2, "Login_Frequency": 22, "Support_Interactions": 1, "Subscription_Type": "Standard", "Preferred_Device": "Smart TV", "Monthly_Fee": 14.99, "Tenure_Months": 18, "Customer_Age": 38, "Payment_Method": "Credit Card", "Content_Language": "Hindi"},
    ]
    df_sample = pd.DataFrame(sample_records)
    output = io.StringIO()
    df_sample.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="streamguard_sample_batch.csv",
    )


@app.route("/api/export-cleaned", methods=["GET"])
def export_cleaned():
    if os.path.exists(CLEANED_CSV_PATH):
        return send_file(CLEANED_CSV_PATH, mimetype="text/csv", as_attachment=True, download_name="ott_customer_churn_cleaned.csv")
    return jsonify({"error": "Cleaned CSV not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting StreamGuard Platform on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
