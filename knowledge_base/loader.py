import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db/chroma")


AML_TYPOLOGIES = [
    {
        "id": "fan_out_001",
        "typology": "Fan-Out",
        "description": "One sender account transfers funds to many different receiver accounts across multiple banks in a short time. Each transfer is a separate wire or cheque payment of similar amounts.",
        "indicators": "Single source bank, multiple destination banks, multiple receivers, similar transfer amounts, rapid sequence of payments",
        "risk_level": "High",
    },
    {
        "id": "fan_in_001",
        "typology": "Fan-In",
        "description": "Many different sender accounts from different banks transfer funds into one single receiver account. Funds arrive from multiple banks and accumulate quickly.",
        "indicators": "Multiple source banks, single destination bank, single receiver account, aggregating amounts, multiple senders",
        "risk_level": "High",
    },
    {
        "id": "cycle_001",
        "typology": "Cycle",
        "description": "Funds are transferred across a chain of different banks and accounts and eventually return to the original sender account. The same amount moves from bank to bank in a loop.",
        "indicators": "Cross-bank transfers, amount returns to origin, circular chain of accounts, layering across banks",
        "risk_level": "Critical",
    },
    {
        "id": "scatter_gather_001",
        "typology": "Scatter-Gather",
        "description": "A sender disperses funds to many receiver accounts across different banks, then those receivers send funds back into one single account. Two stage movement: scatter then gather.",
        "indicators": "First stage multiple receivers different banks, second stage single receiver, same network reused, funds consolidated",
        "risk_level": "Critical",
    },
    {
        "id": "stack_001",
        "typology": "Stack",
        "description": "Funds move in a linear chain from one bank account to another through several intermediate accounts. Each account receives and then immediately sends to the next. Amount decreases slightly at each step due to fees.",
        "indicators": "Sequential cross-bank transfers, pass-through accounts, amount slightly decreasing each hop, wire or SWIFT format",
        "risk_level": "High",
    },
    {
        "id": "fan_out_002",
        "typology": "Fan-Out Micro",
        "description": "A single sender makes many small payments just below the reporting threshold to many different receiver accounts across different banks. Payments are structured to avoid detection.",
        "indicators": "Single sender, amounts between 9000 and 9999, multiple receivers, different destination banks, cheque or wire format",
        "risk_level": "High",
    },
    {
        "id": "fan_in_002",
        "typology": "Fan-In Aggregation",
        "description": "Many senders make small low value transfers to a single receiver account over a long period. Individual amounts are small but total aggregated amount is large.",
        "indicators": "Many source banks, one destination account, low individual amounts, high total volume, slow accumulation over time",
        "risk_level": "Medium",
    },
    {
        "id": "cycle_002",
        "typology": "Rapid Cycle",
        "description": "Funds complete a full circular journey across multiple bank accounts and return to origin within hours. Transfers happen in rapid succession suggesting automated movement.",
        "indicators": "Cross-bank wire transfers, very short time between transfers, identical amounts, automated pattern, returns to origin account",
        "risk_level": "Critical",
    },
]


def load_typologies():
    try:
        client.delete_collection(name="aml_typologies")
        print("Existing collection cleared.")
    except Exception as e:
        print("No existing collection to clear: " + str(e))

    collection = client.get_or_create_collection(name="aml_typologies")

    documents = []
    ids = []
    metadatas = []

    for item in AML_TYPOLOGIES:
        text = (
            "Typology: "
            + item["typology"]
            + ". "
            + item["description"]
            + " Indicators: "
            + item["indicators"]
        )
        documents.append(text)
        ids.append(item["id"])
        metadatas.append(
            {"typology": item["typology"], "risk_level": item["risk_level"]}
        )

    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas
    )

    print("Knowledge base loaded with " + str(len(AML_TYPOLOGIES)) + " AML typologies.")


if __name__ == "__main__":
    load_typologies()
