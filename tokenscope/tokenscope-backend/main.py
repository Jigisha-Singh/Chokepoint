from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- INIT ----------------
load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- MODELS ----------------
class PromptRequest(BaseModel):
    prompt: str

class CompareRequest(BaseModel):
    prompt1: str
    prompt2: str


# ---------------- TOKEN COUNT ----------------
def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------- STOPWORDS ----------------
STOPWORDS = {
    "the", "is", "in", "and", "with", "a", "an", "to", "of", "for", "on", "at", "by"
}


# ---------------- ML: TOKEN IMPORTANCE ----------------
def get_token_importance(text):
    words = text.split()

    if len(words) < 2:
        return {word: 1.0 for word in words}

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words)

    base_scores = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))

    scores = {}

    for word in words:
        w = word.lower()
        score = base_scores.get(w, 0)

        # Boost meaningful words
        score += len(word) * 0.02

        # Penalize stopwords
        if w in STOPWORDS:
            score *= 0.3

        # Boost technical words
        if len(word) > 6:
            score += 0.1

        scores[word] = round(score, 3)

    return scores


# ---------------- VISUALIZATION ----------------
def prepare_visualization_data(text, scores):
    words = text.split()
    visualization = []

    for word in words:
        score = scores.get(word, 0)

        if score > 0.25:
            level = "high"
        elif score > 0.15:
            level = "medium"
        else:
            level = "low"

        visualization.append({
            "word": word,
            "score": score,
            "level": level
        })

    return visualization


# ---------------- PROMPT TRIMMING ----------------
def trim_prompt(text, scores, threshold=0.15):
    words = text.split()

    filtered = [w for w in words if scores.get(w, 0) >= threshold]

    if not filtered:
        filtered = words[:max(2, len(words)//2)]

    return " ".join(filtered)


# ---------------- CORE ANALYZER ----------------
def analyze_prompt(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content

    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(reply)
    total_tokens = prompt_tokens + response_tokens
    cost = round(total_tokens * 0.000001, 6)

    scores = get_token_importance(prompt)

    trimmed_prompt = trim_prompt(prompt, scores)
    trimmed_tokens = count_tokens(trimmed_prompt)
    tokens_saved = prompt_tokens - trimmed_tokens

    return {
        "original_prompt": prompt,
        "response": reply,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "scores": scores,
        "trimmed_prompt": trimmed_prompt,
        "trimmed_tokens": trimmed_tokens,
        "tokens_saved": tokens_saved
    }


# ---------------- PDF ----------------
def generate_pdf(data, filename="tokenscope_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("TokenScope Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Original Prompt: {data['original_prompt']}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Metrics:", styles["Heading2"]))
    for key, value in data["metrics"].items():
        content.append(Paragraph(f"{key}: {value}", styles["Normal"]))

    content.append(Spacer(1, 10))

    content.append(Paragraph("Optimized Prompt:", styles["Heading2"]))
    content.append(Paragraph(data["analysis"]["trimmed_prompt"], styles["Normal"]))

    doc.build(content)


# ---------------- ROOT ----------------
@app.get("/")
def home():
    return {"message": "🚀 TokenScope Backend Running (Full Features Enabled)"}


# ---------------- ANALYZE ----------------
@app.post("/analyze")
async def analyze(req: PromptRequest):
    result = analyze_prompt(req.prompt)

    visualization = prepare_visualization_data(req.prompt, result["scores"])

    return {
        "metrics": {
            "prompt_tokens": result["prompt_tokens"],
            "response_tokens": result["response_tokens"],
            "total_tokens": result["total_tokens"],
            "cost_estimate": result["cost"],
            "tokens_saved": result["tokens_saved"]
        },
        "analysis": {
            "importance_scores": result["scores"],
            "visualization": visualization,
            "trimmed_prompt": result["trimmed_prompt"],
            "trimmed_tokens": result["trimmed_tokens"]
        },
        "response": result["response"]
    }


# ---------------- DOWNLOAD PDF ----------------
@app.post("/download-report")
async def download_report(req: PromptRequest):
    result = analyze_prompt(req.prompt)

    data = {
        "original_prompt": req.prompt,
        "metrics": {
            "prompt_tokens": result["prompt_tokens"],
            "response_tokens": result["response_tokens"],
            "total_tokens": result["total_tokens"],
            "cost_estimate": result["cost"],
            "tokens_saved": result["tokens_saved"]
        },
        "analysis": {
            "trimmed_prompt": result["trimmed_prompt"],
            "trimmed_tokens": result["trimmed_tokens"]
        }
    }

    filename = "tokenscope_report.pdf"
    generate_pdf(data, filename)

    return FileResponse(filename, media_type="application/pdf", filename=filename)


# ---------------- COMPARE ----------------
@app.post("/compare")
async def compare(req: CompareRequest):
    result1 = analyze_prompt(req.prompt1)
    result2 = analyze_prompt(req.prompt2)

    viz1 = prepare_visualization_data(req.prompt1, result1["scores"])
    viz2 = prepare_visualization_data(req.prompt2, result2["scores"])

    if result1["total_tokens"] < result2["total_tokens"]:
        winner = "Prompt 1"
    elif result2["total_tokens"] < result1["total_tokens"]:
        winner = "Prompt 2"
    else:
        winner = "Equal"

    return {
        "prompt1": {
            **result1,
            "visualization": viz1
        },
        "prompt2": {
            **result2,
            "visualization": viz2
        },
        "result": {
            "winner": winner,
            "message": f"{winner} is more cost-efficient"
        }
    }