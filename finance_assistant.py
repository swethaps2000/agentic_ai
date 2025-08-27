from pymongo import MongoClient
from datetime import datetime, timedelta,timezone
from dateutil.relativedelta import relativedelta
import os
from langchain.schema import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Tuple


# Correct: must be GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyBhLq6uZ-Q3HChLQ7HFuFaqFdCIkqGOAt4")

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBhLq6uZ-Q3HChLQ7HFuFaqFdCIkqGOAt4" 
# MONGO_URI ="mongodb://localhost:27017/"
client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
db = client['finance_app']
income = db["income"]
expense = db["expense"]

"""Define categories"""

INCOME_SOURCES = ["Salary", "Freelance", " Bonus"," Investment","Divident"]
EXPENSE_SOURCES = ["Groceries","Eating Out","Transport","Bills","Shopping","Entertainment","Health","Rent","Others"]

"""Helpers to insert data"""

def add_data(user_id,source,amount,account_type,date=None,note=""):
  date = date or datetime.now(timezone.utc)

  if source.lower() in [s.lower() for s in INCOME_SOURCES]:
    income.insert_one({"user_id":user_id,"source":source,"amount":amount,"account_type":account_type,"date":date,"note":note})
    print(f"Data added successfully : {source} -> {amount}")

  elif source in EXPENSE_SOURCES:

     expense.insert_one({
            "user_id": user_id,
            "date": date,
            "category": source,
            "amount": float(amount),
            "account_type": account_type,
            "note": note
        })

     print(f"Data added successfully : {source} -> {amount}")

  else:
    print(f"Invalid source: {source}")

uid1, uid2 = "user_123", "user_456"
# add_data(uid1, "Salary", 50000, "Bank", date=datetime(2024, 2, 1, tzinfo=timezone.utc), note="February salary")
# add_data(uid1, "Groceries", 1800, "Card", date=datetime(2024, 2, 5, tzinfo=timezone.utc), note="Weekly veggies")
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024, 2, 2, tzinfo=timezone.utc))
# add_data(uid1, "Transport", 600, "Cash", date=datetime(2024, 2, 10, tzinfo=timezone.utc), note="Metro & cabs")

# add_data(uid1, "Salary", 51000, "Bank", date=datetime(2024, 3, 1, tzinfo=timezone.utc), note="March salary")
# add_data(uid1, "Groceries", 2000, "Card", date=datetime(2024, 3, 7, tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024, 3, 2, tzinfo=timezone.utc))
# add_data(uid1, "Transport", 700, "Cash", date=datetime(2024, 3, 12, tzinfo=timezone.utc))

# add_data(uid1, "Salary", 50000, "Bank", date=datetime(2024, 4, 1, tzinfo=timezone.utc), note="April salary")
# add_data(uid1, "Groceries", 2200, "Card", date=datetime(2024, 4, 6, tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024, 4, 2, tzinfo=timezone.utc))
# add_data(uid1, "Transport", 650, "Cash", date=datetime(2024, 4, 15, tzinfo=timezone.utc))

# add_data(uid1, "Salary", 52000, "Bank", date=datetime(2024, 5, 1, tzinfo=timezone.utc), note="May salary")
# add_data(uid1, "Groceries", 1900, "Card", date=datetime(2024, 5, 8, tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024, 5, 2, tzinfo=timezone.utc))
# add_data(uid1, "Transport", 800, "Cash", date=datetime(2024, 5, 14, tzinfo=timezone.utc))

# add_data(uid1, "Salary", 50000, "Bank", date=datetime(2024, 6, 1, tzinfo=timezone.utc), note="June salary")
# add_data(uid1, "Groceries", 2100, "Card", date=datetime(2024, 6, 9, tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024, 6, 2, tzinfo=timezone.utc))
# add_data(uid1, "Transport", 700, "Cash", date=datetime(2024, 6, 16, tzinfo=timezone.utc))

# add_data(uid1, "Salary", 53000, "Bank", date=datetime(2024, 7, 1, tzinfo=timezone.utc), note="July salary")
# add_data(uid1, "Groceries", 2000, "Card", date=datetime(2024, 7, 5, tzinfo=timezone.utc))
# add_data(uid1, "Rent", 15000, "Bank", date=datetime(2024, 7, 2, tzinfo=timezone.utc))
# add_data(uid1, "Transport", 750, "Cash", date=datetime(2024, 7, 12, tzinfo=timezone.utc))

# add_data(uid2, "salary",   42000, "Bank", note="August salary")
# add_data(uid2, "Groceries", 1200, "Card")
# add_data(uid2, "Entertainment", 900, "Card")

# 4) Build LangChain Documents per user
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
            return d.isoformat()   # convert datetime → string
        return str(d)

    for r in inc:
        docs.append(Document(
            page_content=inc_text(r),
            metadata={
                "user_id": r["user_id"],
                "type": "income",
                "amount": float(r["amount"]),
                "date": safe_date(r.get("date"))  # ✅ fix here
            }
        ))

    for r in exp:
        docs.append(Document(
            page_content=exp_text(r),
            metadata={
                "user_id": r["user_id"],
                "type": "expense",
                "amount": float(r["amount"]),
                "date": safe_date(r.get("date")),  # ✅ fix here
                "category": r.get("category")
            }
        ))
    return docs


# 5) Vector store (Chroma) + Embeddings (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# We'll persist a shared collection and filter by user_id during retrieval
PERSIST_DIR = "./chroma_finance"
COLLECTION = "finance_txns"

# Build or extend the vector indx for a user

def sync_user_to_vectorstore(user_id:str):
     user_docs = load_user_docs(user_id)
     if not user_docs:
         return None
     
     vs = Chroma.from_documents(
         documents = user_docs,
         embedding=embeddings,
         collection_name=COLLECTION,
         persist_directory=PERSIST_DIR
     )
     vs.persist()
     return vs
 
#   Initial sync for sample users

sync_user_to_vectorstore(uid1)
sync_user_to_vectorstore(uid2)  

# Create a handle to the collection (no-op if already exists)

vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)


# Hybrid Retriever (BM25 + Dense) , user scoped

def make_hybrid_retriever(user_id:str,k_dense=6,k_bm25 = 6,w_bm25 =0.4,w_dense=0.6):
    
     # Load only this user's docs for BM25
    user_docs = load_user_docs(user_id)
    bm25 = BM25Retriever.from_documents(user_docs)
    bm25.k = k_bm25
    
    dense = vectorstore.as_retriever(search_kwargs={"k": k_dense, "filter": {"user_id": user_id}})

    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[w_dense, w_bm25]
    )
    
    
    
# 7) LLM + Prompt + RAG Chain
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0)

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


def build_rag_chain(user_id:str):
    hybrid = make_hybrid_retriever(user_id)
    return(
        {"context": hybrid | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
# 8) Data Agent (Mongo aggregations)

def _date_range_for_query(q:str) -> Tuple[Optional[datetime],Optional[datetime]]:
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
    return [
        {"$match": match},
        {"$group": {"_id": None, "total": {"$sum": "$amount"}, "count": {"$sum": 1}}},
    ]

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

# 9) Advisor Agent (LLM explains numbers)

def explain_numbers(user_id: str, question: str, details: dict):
    context = "\n".join([f"{k}: {v}" for k,v in details.items()])
    msg = f"""User: {user_id}
Question: {question}
Numbers:
{context}

Explain these results succinctly for the user. If a period was implied, mention it."""
    return llm.invoke(msg).content

# 10) Router Agent

MATH_TRIGGERS = ["total", "sum", "average", "avg", "net", "savings", "left", "balance", "spend", "spent", "income", "expense"]

def is_math_query(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in MATH_TRIGGERS)

def detect_category(q: str) -> Optional[str]:
    cats = ["groceries", "rent", "transport", "entertainment", "bills", "shopping", "health"]
    ql = q.lower()
    for c in cats:
        if c in ql:
            return c
    return None

def router_agent(user_id: str, query: str):
    if is_math_query(query):
        start, end = _date_range_for_query(query)
        cat = detect_category(query)

        inc = total_income(user_id, start, end)
        if cat:
            exp = total_expense(user_id, category=cat, start=start, end=end)
        else:
            exp = total_expense(user_id, start=start, end=end)

        details = {
            "period_start": start,
            "period_end": end,
            "category": cat or "ALL",
            "total_income": inc["total"],
            "income_count": inc["count"],
            "total_expense": exp["total"],
            "expense_count": exp["count"],
            "net_savings": inc["total"] - exp["total"],
        }
        return explain_numbers(user_id, query, details)
    else:
        # RAG (hybrid retrieval confined to user_id)
        rag = build_rag_chain(user_id)
        return rag.invoke(query)
    
print("— User:", uid1)
print(router_agent(uid1, "Are there months where my income was higher than usual?"))