import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db/chroma")
collection = client.get_or_create_collection(name="aml_typologies")

CONFIDENCE_THRESHOLD = 0.25


def build_transaction_text(row):
    amount = float(row["amount_paid"])
    same_account = row["from_account"] == row["to_account"]
    same_bank = row["from_bank"] == row["to_bank"]
    cross_currency = row["payment_currency"] != row["receiving_currency"]

    text = (
        "Transaction from bank "
        + str(row["from_bank"])
        + " to bank "
        + str(row["to_bank"])
        + ". "
    )
    text += "Payment format: " + str(row["payment_format"]) + ". "
    text += "Amount paid: " + str(amount) + " " + str(row["payment_currency"]) + ". "

    if cross_currency:
        text += (
            "Cross-currency transaction from "
            + str(row["payment_currency"])
            + " to "
            + str(row["receiving_currency"])
            + ". "
        )
    if same_account:
        text += "Source and destination account are identical, possible self-transfer or reinvestment. "
    if same_bank:
        text += "Transaction within the same bank. "
    else:
        text += "Transaction across different banks. "
    if 9000 <= amount <= 9999:
        text += (
            "Amount is just below reporting threshold, possible structuring behavior. "
        )
    if amount >= 10000:
        text += "Large amount transaction above reporting threshold. "

    return text


def apply_rules(row):
    amount = float(row["amount_paid"])
    same_account = str(row["from_account"]) == str(row["to_account"])
    same_bank = str(row["from_bank"]) == str(row["to_bank"])
    cross_currency = str(row["payment_currency"]) != str(row["receiving_currency"])

    if same_account and same_bank:
        return {
            "status": "CLEAN",
            "typology": "Self-Transfer",
            "risk_level": "Low",
            "similarity": 1.0,
            "escalate_to_llm": False,
            "rule_triggered": "Self-transfer: same account and bank",
        }

    if 9000 <= amount <= 9999 and not same_bank:
        return {
            "status": "RAG_MATCH",
            "typology": "Fan-Out Micro",
            "risk_level": "High",
            "similarity": 1.0,
            "escalate_to_llm": False,
            "rule_triggered": "Structuring: amount between 9000-9999 cross-bank",
        }

    if cross_currency and amount >= 10000:
        return {
            "status": "RAG_MATCH",
            "typology": "Stack",
            "risk_level": "High",
            "similarity": 1.0,
            "escalate_to_llm": False,
            "rule_triggered": "Cross-currency large transfer above 10000",
        }

    if cross_currency:
        return {
            "status": "RAG_MATCH",
            "typology": "Cycle",
            "risk_level": "Medium",
            "similarity": 1.0,
            "escalate_to_llm": False,
            "rule_triggered": "Cross-currency transaction",
        }

    return None


def query_rag(row):
    rule_result = apply_rules(row)
    if rule_result is not None:
        rule_result["transaction_text"] = build_transaction_text(row)
        rule_result["matched_doc"] = ""
        return rule_result

    transaction_text = build_transaction_text(row)
    embedding = model.encode([transaction_text]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    distance = results["distances"][0][0]
    similarity = 1 - (distance / 2)

    matched_typology = results["metadatas"][0][0]["typology"]
    risk_level = results["metadatas"][0][0]["risk_level"]
    matched_doc = results["documents"][0][0]

    if similarity >= CONFIDENCE_THRESHOLD:
        return {
            "status": "RAG_MATCH",
            "typology": matched_typology,
            "risk_level": risk_level,
            "similarity": round(similarity, 4),
            "matched_doc": matched_doc,
            "transaction_text": transaction_text,
            "escalate_to_llm": False,
            "rule_triggered": None,
        }
    else:
        return {
            "status": "LLM_ESCALATION",
            "typology": None,
            "risk_level": None,
            "similarity": round(similarity, 4),
            "matched_doc": matched_doc,
            "transaction_text": transaction_text,
            "escalate_to_llm": True,
            "rule_triggered": None,
        }


if __name__ == "__main__":
    test_row = {
        "from_bank": 10,
        "from_account": "ACC001",
        "to_bank": 10,
        "to_account": "ACC001",
        "amount_paid": 3000,
        "payment_currency": "US Dollar",
        "amount_received": 3000,
        "receiving_currency": "US Dollar",
        "payment_format": "Reinvestment",
    }
    result = query_rag(test_row)
    print("Status: " + result["status"])
    print("Typology: " + str(result["typology"]))
    print("Rule: " + str(result.get("rule_triggered")))
