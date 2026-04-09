#  TokenScope  
### Smarter Prompts. Lower Costs. Better AI.

TokenScope is a lightweight tool that helps developers **analyze, optimize, and reduce the cost of AI prompts** by identifying which tokens actually matter.

---

##  Problem

LLM APIs charge based on **token usage**, but:

- Prompts often contain **30–40% unnecessary words**
- Developers have **no visibility into token importance**
- This leads to:
  -  Higher API costs  
  -  Lower response quality  

---

##  Solution

TokenScope provides a **token-level analysis system** that:

- Tracks token usage and estimated cost  
- Identifies **high-impact vs low-impact words**  
- Suggests a **trimmed, optimized prompt**  

---

##  How It Works

1. **Input Prompt**
2. **Token Analysis**
   - Token count via LLM API
   - Cost estimation
3. **Importance Scoring**
   - TF-IDF for relevance
   - Stopword removal for noise reduction
   - POS tagging (nouns/verbs weighted higher)
4. **Prompt Optimization**
   - Removes low-score tokens
   - Generates a shorter, meaningful prompt

---

##  Features

-  **Token Tracking**
  - Prompt & response token count
  - Real-time cost estimation  

-  **Token Importance Heatmap**
  - Highlights impactful words  
  - Visual distinction between useful vs noise  

-  **Prompt Optimizer**
  - Generates trimmed prompts  
  - Reduces token usage without losing intent  

-  **(Bonus) Usage Tracking**
  - Store and analyze prompt history  

---

##  Example

**Input Prompt:**
```
Please explain artificial intelligence in detail with some good examples
```

**Optimized Output:**
```
artificial intelligence explain examples
```

**Result:**
-  ~30–40% token reduction  
-  Lower cost  
-  Cleaner input for better responses  

---

##  Tech Stack

**Frontend**
- React / HTML / CSS  

**Backend**
- Python (Flask / FastAPI)

**NLP**
- TF-IDF (scikit-learn)  
- POS Tagging (spaCy)

**API**
- OpenAI / Gemini (for token usage)

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/tokenscope.git
cd tokenscope
```

### 2. Backend setup
```bash
pip install -r requirements.txt
python app.py
```

### 3. Frontend setup
```bash
cd frontend
npm install
npm start
```

---

##  Future Improvements

- Multi-turn conversation tracking  
- Prompt comparison tool  
- IDE integration (VS Code extension)  
- Advanced prompt rewriting using LLMs  

---

##  Contributing

Contributions are welcome!  
Feel free to fork the repo and submit a PR.

---

##  License

MIT License

---

##  Closing Thought

> Optimize before you prompt.
