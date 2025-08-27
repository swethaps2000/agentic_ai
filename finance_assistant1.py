from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import os, time
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from collections import OrderedDict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Tuple
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyDOal5Wb5J49UxtZqNA8rrmkm-B0N25IPU")
client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
db = client['finance_app']
income = db["income"]
expense = db["expense"]
feedback_collection = db["feedback"]

INCOME_SOURCES = ["Salary", "Freelance", "Bonus", "Investment", "Dividend"]
EXPENSE_SOURCES = ["Groceries", "Eating Out", "Transport", "Bills", "Shopping", "Entertainment", "Health", "Rent", "Others"]
all_categories = INCOME_SOURCES + EXPENSE_SOURCES

# reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL)

# ----------------------------
# Embeddings + Vectorstore
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# helper normalize
def normalize(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0 or np.isnan(norm):
        return v
    return v / norm

# precompute category embeddings
category_embeddings = {cat: normalize(embeddings.embed_query(cat)) for cat in all_categories}

# ----------------------------
# Simple in-memory cache (TTL)
# ----------------------------
class SimpleCache:
    def __init__(self, ttl=30):
        self.ttl = ttl
        self.store = {}
    def get(self, key):
        v = self.store.get(key)
        if not v: return None
        ts, val = v
        if time.time() - ts > self.ttl:
            del self.store[key]
            return None
        return val
    def set(self, key, val):
        self.store[key] = (time.time(), val)
    def clear(self):
        self.store.clear()

cache = SimpleCache(ttl=60)  # cache rag results for 60s by default

# ----------------------------
# Intent classifier (embeddings-based)
# ----------------------------
INTENT_TEMPLATES = {
    "pattern": "User wants trends/plots/patterns over time (e.g., 'show spending pattern Jan-Jun 2024')",
    "math": "User asks for totals, sums, averages, net savings, balances (e.g., 'total spent last month')",
    "general": "General question about transactions or merchant names (fallback)",
    "feedback": "User feedback/correction to a previous answer"
}
intent_embeddings = {k: normalize(embeddings.embed_query(v)) for k, v in INTENT_TEMPLATES.items()}

def classify_intent(query: str, threshold: float = 0.55) -> str:
    q_emb = normalize(embeddings.embed_query(query))
    best_intent, best_sim = None, -1
    for name, emb in intent_embeddings.items():
        sim = float(np.dot(q_emb, emb))
        if sim > best_sim:
            best_intent, best_sim = name, sim
    # fallback to general if low confidence
    if best_sim < threshold:
        return "general"
    return best_intent

# ----------------------------
# Helper: dedupe documents preserving order
# ----------------------------

def dedupe_docs(docs):
    seen = set()
    out = []
    for d in docs:
        key = (d.page_content, tuple(sorted(d.metadata.items())) if d.metadata else None)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

# ----------------------------
# Patterns and aggregation
# ----------------------------

def income_pattern(user_id: str, start=None, end=None):
    match = {"user_id": user_id}
    if start or end:
        match["date"] = {}
        if start: match["date"]["$gte"] = start
        if end:   match["date"]["$lte"] = end

    pipeline = [
        {"$match": match},
        {"$group": {"_id": {"month": {"$month": "$date"}, "year": {"$year": "$date"}, "source": "$source"}, "total": {"$sum": "$amount"}}},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]

    res = list(income.aggregate(pipeline))
    pattern = []
    for r in res:
        pattern.append({
            "year": r["_id"]["year"],
            "month": r["_id"]["month"],
            "source": r["_id"]["source"],
            "total": r["total"]
        })
    return pattern


def spending_pattern(user_id: str, start=None, end=None, category: Optional[str] = None):
    match = {"user_id": user_id}
    if category:
        match["category"] = {"$regex": f"^{category}$", "$options": "i"}
    if start or end:
        match["date"] = {}
        if start: match["date"]["$gte"] = start
        if end:   match["date"]["$lte"] = end

    pipeline = [
        {"$match": match},
        {"$group": {"_id": {"month": {"$month": "$date"}, "year": {"$year": "$date"}, "category": "$category"}, "total": {"$sum": "$amount"}}},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]

    res = list(expense.aggregate(pipeline))
    pattern = []
    for r in res:
        pattern.append({
            "year": r["_id"]["year"],
            "month": r["_id"]["month"],
            "category": r["_id"]["category"],
            "total": r["total"]
        })
    return pattern

# ----------------------------
# RAG retrieval + rerank (cached wrapper)
# ----------------------------

def retrieve_and_rerank(user_id: str, query: str, k_initial=20, k_final=5):
    # NOTE: we keep this function pure (no external cache) and add caching at caller level
    user_docs = load_user_docs(user_id)
    bm25 = BM25Retriever.from_documents(user_docs)
    bm25.k = k_initial
    bm25_results = bm25.get_relevant_documents(query)

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k_initial, "filter": {"user_id": user_id}})
    dense_results = dense_retriever.get_relevant_documents(query)

    combined = bm25_results + dense_results
    combined = dedupe_docs(combined)
    candidates = combined[:k_initial]

    if not candidates:
        return []

    rerank_inputs = [[query, d.page_content] for d in candidates]
    scores = reranker.predict(rerank_inputs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, s in scored[:k_final]]
    return top_docs

# ----------------------------
# Format docs with citation markers
# ----------------------------

def format_docs_with_citations(docs):
    lines = []
    citations = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        date = meta.get("date") or meta.get("datetime") or ""
        category = meta.get("category") or meta.get("type") or ""
        short_date = str(date)[:10] if date else ""
        citation_label = f"[#{idx}:{short_date}/{category}]" if (short_date or category) else f"[#{idx}]"
        citations.append({"label": citation_label, "meta": meta})
        lines.append(f"{doc.page_content} {citation_label}")
    context = "\n".join(lines)
    return context, citations

# ----------------------------
# Safe answer generator for numbers (deterministic)
# ----------------------------

def safe_explain_numbers(user_id: str, question: str, details: dict, use_llm: bool = False):
    """
    Deterministic, template-based explanation. When use_llm=True we still call LLM but validate.
    This avoids numeric hallucinations.
    """
    # Extract fields safely
    start = details.get("period_start")
    end = details.get("period_end")
    cat = details.get("category", "ALL")
    total_income = float(details.get("total_income", 0) or 0)
    income_count = int(details.get("income_count", 0) or 0)
    total_expense = float(details.get("total_expense", 0) or 0)
    expense_count = int(details.get("expense_count", 0) or 0)
    net = total_income - total_expense

    # human friendly period
    if start and end:
        try:
            start_s = start.strftime("%Y-%m-%d")
            end_s = end.strftime("%Y-%m-%d")
            period = f"from {start_s} to {end_s}"
        except Exception:
            period = "for the requested period"
    else:
        period = "for the requested period"

    template = (f"{period}, category: {cat}. Total income: {total_income:.2f} across {income_count} items. "
                f"Total expense: {total_expense:.2f} across {expense_count} items. Net savings: {net:.2f}.")

    if not use_llm:
        return template

    # optional: call LLM for a richer explanation but validate numbers
    msg = f"Explain the following numbers clearly. Do NOT change or invent numbers.\n\nNumbers:\n{template}\n\nQuestion: {question}\n"
    try:
        resp = llm.invoke(msg)
        text = getattr(resp, "content", getattr(resp, "text", str(resp)))
    except Exception:
        text = template

    # simple validation: check that the computed net exists (as substring) or else fallback
    if f"{net:.2f}" not in text:
        # LLM likely altered numbers — fallback to the safe template
        return template + "\n\n(Note: verified numbers provided above.)"
    return text

# ----------------------------
# Helpers to insert data (unchanged)
# ----------------------------

def ai_map_category(source: str, threshold: float = 0.45) -> str:
    src_emb = normalize(embeddings.embed_query(source))
    best_cat, best_sim = None, -1
    for cat, emb in category_embeddings.items():
        sim = float(np.dot(src_emb, emb))
        if sim > best_sim:
            best_cat, best_sim = cat, sim
    # debug print
    print(f"DEBUG mapping '{source}' → {best_cat} (sim={best_sim:.2f})")
    return best_cat if best_sim >= threshold else "Others"


def add_data(user_id, source, amount, account_type, date=None, note=""):
    date = date or datetime.now(timezone.utc)
    category = ai_map_category(source)
    if category in INCOME_SOURCES:
        income.insert_one({"user_id": user_id, "source": category, "amount": float(amount), "account_type": account_type, "date": date, "note": note})
        print(f"✅ Income added: {source} → mapped to {category} ({amount})")
    elif category in EXPENSE_SOURCES:
        expense.insert_one({"user_id": user_id, "category": category, "amount": float(amount), "account_type": account_type, "date": date, "note": note})
        print(f"✅ Expense added: {source} → mapped to {category} ({amount})")
    else:
        expense.insert_one({"user_id": user_id, "category": "Others", "amount": float(amount), "account_type": account_type, "date": date, "note": note})
        print(f"⚠️ Unknown category: {source} → saved as 'Others'")
uid1 = "user_123"      
# add_data(uid1, "Salary", 50000, "Bank", date=datetime(2024,1,1,tzinfo=timezone.utc), note="January salary")
# add_data(uid1, "Groceries", 1800, "Card", date=datetime(2024,1,5,tzinfo=timezone.utc), note="Weekly veggies")
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024,1,2,tzinfo=timezone.utc))
# add_data(uid1, "Transport", 600, "Cash", date=datetime(2024,1,10,tzinfo=timezone.utc), note="Metro & cabs")

# # February
# add_data(uid1, "Salary", 51000, "Bank", date=datetime(2024,2,1,tzinfo=timezone.utc), note="February salary")
# add_data(uid1, "Groceries", 2000, "Card", date=datetime(2024,2,6,tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024,2,2,tzinfo=timezone.utc))
# add_data(uid1, "Entertainment", 1200, "Card", date=datetime(2024,2,15,tzinfo=timezone.utc), note="Movie and dinner")

# # March
# add_data(uid1, "Salary", 52000, "Bank", date=datetime(2024,3,1,tzinfo=timezone.utc), note="March salary")
# add_data(uid1, "Groceries", 2100, "Card", date=datetime(2024,3,7,tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024,3,2,tzinfo=timezone.utc))
# add_data(uid1, "Transport", 700, "Cash", date=datetime(2024,3,12,tzinfo=timezone.utc))

# # April
# add_data(uid1, "Salary", 50000, "Bank", date=datetime(2024,4,1,tzinfo=timezone.utc), note="April salary")
# add_data(uid1, "Groceries", 2200, "Card", date=datetime(2024,4,5,tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024,4,2,tzinfo=timezone.utc))
# add_data(uid1, "Health", 3000, "Bank", date=datetime(2024,4,14,tzinfo=timezone.utc), note="Doctor visit + medicines")

# # May
# add_data(uid1, "Salary", 53000, "Bank", date=datetime(2024,5,1,tzinfo=timezone.utc), note="May salary")
# add_data(uid1, "Groceries", 1900, "Card", date=datetime(2024,5,9,tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024,5,2,tzinfo=timezone.utc))
# add_data(uid1, "Shopping", 4000, "Card", date=datetime(2024,5,18,tzinfo=timezone.utc), note="Clothes + accessories")

# # June
# add_data(uid1, "Salary", 52000, "Bank", date=datetime(2024,6,1,tzinfo=timezone.utc), note="June salary")
# add_data(uid1, "Groceries", 2100, "Card", date=datetime(2024,6,7,tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024,6,2,tzinfo=timezone.utc))
add_data(uid1, "Biriyani", 800, "Cash", date=datetime(2024,6,11,tzinfo=timezone.utc), note="Biriyani")

# ----------------------------
# Document building
# ----------------------------

def inc_text(d):
    date = d.get("date")
    d_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
    return f"On {d_str}, income of {d['amount']} from {d['source']} via {d['account_type']}. Note: {d.get('note','')}"


def exp_text(d):
    date = d.get("date")
    d_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
    return f"On {d_str}, expense of {d['amount']} for {d['category']} via {d['account_type']}. Note: {d.get('note','')}"


def load_user_docs(user_id: str):
    inc = list(income.find({"user_id": user_id}))
    exp = list(expense.find({"user_id": user_id}))
    docs = []

    def safe_date(d):
        if isinstance(d, datetime):
            return d.isoformat()
        return str(d)

    for r in inc:
        docs.append(Document(page_content=inc_text(r), metadata={"user_id": r["user_id"], "type": "income", "amount": float(r["amount"]), "date": safe_date(r.get("date"))}))
    for r in exp:
        docs.append(Document(page_content=exp_text(r), metadata={"user_id": r["user_id"], "type": "expense", "amount": float(r["amount"]), "date": safe_date(r.get("date")), "category": r.get("category")}))
    return docs

# ----------------------------
# Vectorstore sync / handle
# ----------------------------

PERSIST_DIR = "./chroma_finance"
COLLECTION = "finance_txns"

def sync_user_to_vectorstore(user_id: str, batch_size=5, sleep_sec=2):
    user_docs = load_user_docs(user_id)
    if not user_docs:
        return None
    for i in range(0, len(user_docs), batch_size):
        batch = user_docs[i:i+batch_size]
        vs = Chroma.from_documents(documents=batch, embedding=embeddings, collection_name=COLLECTION, persist_directory=PERSIST_DIR)
        vs.persist()
        time.sleep(sleep_sec)
    return vs

vectorstore = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=PERSIST_DIR)

# ----------------------------
# Adaptive retriever weights
# ----------------------------

def get_user_weights(user_id: str, base_dense=0.6, base_bm25=0.4):
    fb = list(feedback_collection.find({"user_id": user_id}))
    if not fb:
        return base_dense, base_bm25
    good_dense = bad_dense = good_bm25 = bad_bm25 = 0
    for f in fb:
        if f.get("docs"):
            for d in f["docs"]:
                if f["feedback"] == "good":
                    if d.get("type") == "dense": good_dense += 1
                    if d.get("type") == "bm25": good_bm25 += 1
                elif f["feedback"] == "bad":
                    if d.get("type") == "dense": bad_dense += 1
                    if d.get("type") == "bm25": bad_bm25 += 1
    dense_score = max(1, good_dense - bad_dense)
    bm25_score = max(1, good_bm25 - bad_bm25)
    total = dense_score + bm25_score
    return dense_score/total, bm25_score/total

# ----------------------------
# make_hybrid_retriever
# ----------------------------

def make_hybrid_retriever(user_id: str, k_dense=6, k_bm25=6):
    w_dense, w_bm25 = get_user_weights(user_id)
    user_docs = load_user_docs(user_id)
    bm25 = BM25Retriever.from_documents(user_docs)
    bm25.k = k_bm25
    dense = vectorstore.as_retriever(search_kwargs={"k": k_dense, "filter": {"user_id": user_id}})
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[w_dense, w_bm25])

# ----------------------------
# LLM and prompt (unchanged)
# ----------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
    You are a finance assistant for a single user. Use only the provided context (their transactions).
    If asked for totals/aggregations, be cautious and prefer precise numbers.
    If unsure, say don't have enough context.

    Question : {question}
    Context : {context}

    Answer clearly and concisely.
    """
)


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


def build_rag_chain(user_id: str):
    hybrid = make_hybrid_retriever(user_id)
    return ({"context": hybrid | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# ----------------------------
# Date helpers + totals (unchanged)
# ----------------------------

def _date_range_for_query(q: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    ql = q.lower()
    now = datetime.now(timezone.utc)
    if "today" in ql:
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
    elif "yesterday" in ql:
        end = datetime(now.year, now.month, now.day)
        start = end - timedelta(days=1)
    elif "last week" in ql or "past week" in ql:
        end = now
        start = now - timedelta(days=7)
    elif "last month" in ql:
        first_this = datetime(now.year, now.month, 1)
        end = first_this
        start = (first_this - relativedelta(months=1))
    elif "this month" in ql or "current month" in ql:
        start = datetime(now.year, now.month, 1)
        end = (start + relativedelta(months=1))
    else:
        start = None
        end = None
    return start, end


def _sum_pipeline(match):
    return [{"$match": match}, {"$group": {"_id": None, "total": {"$sum": "$amount"}, "count": {"$sum": 1}}}]


def total_income(user_id: str, start=None, end=None):
    match = {"user_id": user_id}
    if start or end:
        match["date"] = {}
        if start: match["date"]["$gte"] = start
        if end:   match["date"]["$lte"] = end
    res = list(income.aggregate(_sum_pipeline(match)))
    return res[0] if res else {"total": 0.0, "count": 0}


def total_expense(user_id: str, category: Optional[str] = None, start=None, end=None):
    match = {"user_id": user_id}
    if category:
        match["category"] = {"$regex": f"^{category}$", "$options": "i"}
    if start or end:
        match["date"] = {}
        if start: match["date"]["$gte"] = start
        if end:   match["date"]["$lte"] = end
    res = list(expense.aggregate(_sum_pipeline(match)))
    return res[0] if res else {"total": 0.0, "count": 0}

# ----------------------------
# Improved category detection using embeddings
# ----------------------------

def detect_category_from_text(text: str, threshold: float = 0.45) -> Optional[str]:
    q_emb = normalize(embeddings.embed_query(text))
    best_cat, best_sim = None, -1
    for cat, emb in category_embeddings.items():
        sim = float(np.dot(q_emb, emb))
        if sim > best_sim:
            best_cat, best_sim = cat, sim
    if best_sim >= threshold:
        return best_cat
    return None

# ----------------------------
# RAG wrapper with cache + citations
# ----------------------------

def rag_answer_with_citations_cached(user_id: str, question: str, k_initial=20, k_final=5):
    cache_key = f"rag:{user_id}:{question}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    docs = retrieve_and_rerank(user_id, question, k_initial=k_initial, k_final=k_final)
    if not docs:
        out = "I couldn't find matching transaction records for that query."
        cache.set(cache_key, out)
        return out

    context, citations = format_docs_with_citations(docs)

    prompt_text = f"""
You are a finance assistant. Use ONLY the provided transaction records and their citation markers to answer precisely.
If you reference a specific transaction or total, append the citation marker (e.g. [#1:2024-06-09/Groceries]).
If the question asks for totals, compute them exactly from the records below.

Question: {question}

Records:
{context}

Answer concisely. Include citation markers for any numbers or transactions you mention.
"""

    text = None
    try:
        resp = llm.invoke(prompt_text)

        # Gemini can respond in several structures, so check safely
        if hasattr(resp, "content") and resp.content:
            text = resp.content
        elif hasattr(resp, "text") and resp.text:
            text = resp.text
        elif hasattr(resp, "candidates"):
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                pass
        if not text:
            text = "I could not extract an answer from the model."

    except Exception as e:
        text = f"Model error: {e}"

    # Clean up prompt-echo if Gemini parrots it
    if "Records:" in text and "Answer concisely" in text:
        text = text.split("Answer concisely.")[-1].strip()

    citation_summary = "Sources: " + ", ".join([f"{c['label']}" for c in citations])
    out = f"{text}\n\n{citation_summary}"
    cache.set(cache_key, out)
    return out


# ----------------------------
# Router agent (uses classifier + safe explain)
# ----------------------------

MATH_TRIGGERS = ["total", "sum", "average", "avg", "net", "savings", "left", "balance", "spend", "spent", "income", "expense"]

def is_math_query(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in MATH_TRIGGERS)


def router_agent(user_id: str, query: str):
    start, end = _date_range_for_query(query)
    # classify intent using embeddings
    intent = classify_intent(query)
    # detect category using embeddings
    cat = detect_category_from_text(query)

    docs_used = []
    pattern_data = None

    if intent == "pattern":
        expense_pat = spending_pattern(user_id, start=start, end=end, category=cat)
        income_pat = income_pattern(user_id, start=start, end=end)
        pattern_data = {"income": income_pat, "expense": expense_pat}
        details = {"months": len(set([f"{p['year']}-{p['month']}" for p in expense_pat + income_pat])), "total_income": sum(p["total"] for p in income_pat), "total_expense": sum(p["total"] for p in expense_pat), "net_savings": sum(p["total"] for p in income_pat) - sum(p["total"] for p in expense_pat)}
        answer = safe_explain_numbers(user_id, query, details, use_llm=False)
    elif intent == "math" or is_math_query(query) or cat:
        inc = total_income(user_id, start, end)
        exp = total_expense(user_id, category=cat, start=start, end=end)
        details = {"period_start": start, "period_end": end, "category": cat or "ALL", "total_income": inc["total"], "income_count": inc["count"], "total_expense": exp["total"], "expense_count": exp["count"], "net_savings": inc["total"] - exp["total"]}
        answer = safe_explain_numbers(user_id, query, details, use_llm=False)
    else:
        docs_used = retrieve_and_rerank(user_id, query, k_initial=20, k_final=5)
        answer = rag_answer_with_citations_cached(user_id, query, k_initial=20, k_final=5)

    return {"answer": answer, "pattern": pattern_data, "docs": docs_used, "feedback_fn": lambda fb, notes="": save_feedback(user_id, query, answer, fb, notes, docs=[d.metadata for d in docs_used])}



# End of file
