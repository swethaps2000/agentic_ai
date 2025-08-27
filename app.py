from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone

from finance_assistant1 import add_data, sync_user_to_vectorstore, router_agent

app = FastAPI(title="Finance Assistant API", version="1.0")


class TransactionIn(BaseModel):
    user_id: str
    source: str
    amount: float
    account_type: str
    date: Optional[datetime] = None
    note: Optional[str] = ""

class BulkTransactionsIn(BaseModel):
    user_id: str
    transactions: List[TransactionIn]

class QueryIn(BaseModel):
    user_id: str
    question: str


# ----------------------------
# Endpoints
# ----------------------------

@app.post("/add_transaction/")
def add_transaction(txn: TransactionIn):
    try:
        add_data(
            user_id=txn.user_id,
            source=txn.source,
            amount=txn.amount,
            account_type=txn.account_type,
            date=txn.date or datetime.now(timezone.utc),
            note=txn.note
        )
        sync_user_to_vectorstore(txn.user_id)
        return {"message": "Transaction added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bulk_add_transactions/")
def bulk_add_transactions(payload: BulkTransactionsIn):
    try:
        for txn in payload.transactions:
            add_data(
                user_id=payload.user_id,
                source=txn.source,
                amount=txn.amount,
                account_type=txn.account_type,
                date=txn.date or datetime.now(timezone.utc),
                note=txn.note
            )
        sync_user_to_vectorstore(payload.user_id)
        return {"message": f"{len(payload.transactions)} transactions added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/")
def ask_question(q: QueryIn):
    try:
        res = router_agent(q.user_id, q.question)
        return {
            "answer": res["answer"],
            "pattern": res.get("pattern", None),
            "docs_used": [d.page_content for d in res["docs"]] if res["docs"] else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
