"""
FAQ Chatbot Backend — Flask + NLTK (TF cosine similarity)
Run: python chatbot_app.py
Port: 5001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk, string, math
from collections import Counter, defaultdict

for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
#  FAQ DATASET
# ─────────────────────────────────────────────────────────────────────────────
FAQ_DATA = [
    {"id":1,"category":"Account","question":"How do I reset my password?","answer":"Go to the login page and click 'Forgot Password'. Enter your registered email and we'll send a secure reset link within 2 minutes. Check your spam folder if it doesn't arrive. The link expires after 30 minutes."},
    {"id":2,"category":"Account","question":"How do I create a new account?","answer":"Click 'Sign Up' on the homepage. Fill in your full name, email, and a strong password (min 8 chars, one uppercase, one number). Verify your email via the confirmation link we send, and your account is ready immediately."},
    {"id":3,"category":"Account","question":"How do I update my profile information?","answer":"Navigate to Settings → Profile. You can edit your display name, email, phone number, bio, and profile picture. Click 'Save Changes' to confirm. Email changes require re-verification."},
    {"id":4,"category":"Account","question":"How do I delete my account permanently?","answer":"Go to Settings → Account → Delete Account. You will be asked to confirm your password. Deletion is irreversible — all your data is erased within 30 days. Export your data first via Settings → Privacy → Download My Data."},
    {"id":5,"category":"Account","question":"Can I use the app on multiple devices?","answer":"Yes. Standard plan allows up to 5 simultaneous sessions; Pro plan is unlimited. View and revoke active sessions anytime from Settings → Security → Active Sessions."},
    {"id":6,"category":"Billing","question":"What payment methods do you accept?","answer":"We accept Visa, Mastercard, American Express, PayPal, Google Pay, Apple Pay, and UPI. All transactions are encrypted with 256-bit SSL. We do not store full card numbers on our servers."},
    {"id":7,"category":"Billing","question":"How do I cancel my subscription?","answer":"Go to Settings → Billing → Cancel Plan and follow the steps. Your subscription remains active until the end of the current billing cycle. We do not offer pro-rated refunds for partial months."},
    {"id":8,"category":"Billing","question":"How do I request a refund?","answer":"Refunds are available within 14 days of purchase. Submit a request via Settings → Billing → Request Refund or email billing@example.com. Our team reviews all requests within 3–5 business days."},
    {"id":9,"category":"Billing","question":"How do I upgrade my plan?","answer":"Visit Settings → Billing → Upgrade Plan. Choose your new plan and complete payment. The upgrade takes effect immediately and you are only charged the prorated difference for the remaining billing period."},
    {"id":10,"category":"Billing","question":"Is there a free trial available?","answer":"Yes — 14-day full Pro plan trial, no credit card required. At trial end you can choose any paid plan or continue on the free tier with limited features."},
    {"id":11,"category":"Security","question":"How do I enable two-factor authentication?","answer":"Settings → Security → Two-Factor Authentication. Choose between an authenticator app (Google Authenticator or Authy recommended) or SMS OTP. Save your backup codes in a safe place."},
    {"id":12,"category":"Security","question":"Is my data secure and private?","answer":"Yes. We use AES-256 encryption at rest and TLS 1.3 in transit. We are GDPR, SOC 2 Type II, and ISO 27001 compliant. Annual third-party penetration tests are conducted and results are published on our Trust Center."},
    {"id":13,"category":"Technical","question":"What browsers and devices are supported?","answer":"We support Chrome 90+, Firefox 88+, Safari 14+, and Edge 90+ on desktop. Mobile: iOS 14+ and Android 10+. JavaScript must be enabled. No browser extensions or plugins are required."},
    {"id":14,"category":"Technical","question":"Is there a mobile app?","answer":"Yes! Download from the App Store (iOS) or Google Play (Android). The mobile app includes offline access, push notifications, and biometric login. It syncs in real time with your web account."},
    {"id":15,"category":"Technical","question":"Does the app work offline?","answer":"Core features require internet. However, recently accessed content is cached locally. Changes made offline are queued and synced automatically when connectivity is restored."},
    {"id":16,"category":"Support","question":"How do I contact customer support?","answer":"Support is available 24/7 via: (1) Live chat on the website — avg. response under 2 min. (2) Email: support@example.com — response within 4 hrs. (3) Phone: +1-800-000-0000, Mon–Fri 9 AM–6 PM EST."},
    {"id":17,"category":"Support","question":"How do I report a bug or technical issue?","answer":"Use Help → Report a Bug in the app, or email bugs@example.com. Please include a description of the problem, steps to reproduce it, your browser and OS version, and a screenshot if possible. We triage all reports within 24 hours."},
    {"id":18,"category":"Privacy","question":"How do I download a copy of my data?","answer":"Settings → Privacy → Download My Data. We compile a full archive (account info, content, activity logs) and email you a secure download link within 24 hours. The link is valid for 7 days."},
    {"id":19,"category":"Privacy","question":"Do you sell my personal data to third parties?","answer":"No. We never sell, rent, or trade your personal data. We share data only with infrastructure partners (e.g. cloud hosting) strictly to operate the service, under binding data processing agreements."},
    {"id":20,"category":"Features","question":"How do I share content or collaborate with teammates?","answer":"Open any item, click Share, and invite people by email or generate a shareable link. You can set permissions: View, Comment, or Edit. Team plan members can also create shared workspaces for ongoing collaboration."},
]

# ─────────────────────────────────────────────────────────────────────────────
#  NLP PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

def build_tf(tokens):
    total = len(tokens) or 1
    return {w: c / total for w, c in Counter(tokens).items()}

def cosine_similarity(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot  = sum(a[w] * b[w] for w in common)
    magA = math.sqrt(sum(v*v for v in a.values()))
    magB = math.sqrt(sum(v*v for v in b.values()))
    return dot / (magA * magB) if magA and magB else 0.0

# Pre-index all FAQs at startup
_indexed = []
for faq in FAQ_DATA:
    tokens = preprocess(faq["question"] + " " + faq["answer"])
    _indexed.append({"faq": faq, "tf": build_tf(tokens)})

def find_matches(query, top_n=4):
    tokens = preprocess(query)
    if not tokens:
        return []
    qvec = build_tf(tokens)
    scored = sorted(
        [{"score": cosine_similarity(qvec, item["tf"]), "faq": item["faq"]} for item in _indexed],
        key=lambda x: x["score"], reverse=True
    )
    return scored[:top_n]

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD = 0.06

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message field is required."}), 400
    if len(message) > 500:
        return jsonify({"error": "Message must be 500 characters or fewer."}), 400

    results = find_matches(message, top_n=4)
    if not results or results[0]["score"] < THRESHOLD:
        return jsonify({
            "matched": False,
            "reply": "I couldn't find a clear answer for that. Try rephrasing, or contact support@example.com.",
            "suggestions": [f["question"] for f in FAQ_DATA[:5]],
            "score": 0,
        })

    best = results[0]
    related = [r["faq"]["question"] for r in results[1:] if r["score"] >= THRESHOLD]
    return jsonify({
        "matched": True,
        "reply": best["faq"]["answer"],
        "matched_question": best["faq"]["question"],
        "category": best["faq"]["category"],
        "score": round(best["score"], 4),
        "score_pct": min(99, round(best["score"] * 340)),
        "related": related[:2],
    })

@app.route("/faqs", methods=["GET"])
def get_faqs():
    cat = request.args.get("category", "").strip().lower()
    data = [f for f in FAQ_DATA if not cat or f["category"].lower() == cat]
    return jsonify(data)

@app.route("/categories", methods=["GET"])
def get_categories():
    counts = defaultdict(int)
    for f in FAQ_DATA:
        counts[f["category"]] += 1
    return jsonify([{"name": k, "count": v} for k, v in sorted(counts.items())])

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "faq_count": len(FAQ_DATA)})

if __name__ == "__main__":
    print("AskBot backend running on http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)