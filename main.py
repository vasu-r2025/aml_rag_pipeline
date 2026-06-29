import time
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from pipeline.ingestor import load_transactions
from pipeline.rag_engine import query_rag
from pipeline.llm_handler import analyze_with_llm
from db.storage import init_db, save_result, get_summary

BATCH_LIMIT = 5
LLM_ESCALATION_LIMIT = 3


def run_pipeline():
    print("Initializing database...")
    init_db()

    print("Starting AML pipeline...")
    print("")

    llm_call_count = 0
    batch_count = 0

    for batch_num, chunk in load_transactions(chunksize=500):
        if batch_count >= BATCH_LIMIT:
            print("Batch limit reached. Stopping.")
            break

        print(
            "Processing batch "
            + str(batch_num)
            + " with "
            + str(len(chunk))
            + " transactions..."
        )

        for _, row in chunk.iterrows():
            result = query_rag(row)

            if result["escalate_to_llm"] and llm_call_count < LLM_ESCALATION_LIMIT:
                print(
                    "  Escalating to LLM for transaction from "
                    + str(row["from_account"])
                    + "..."
                )
                llm_result = analyze_with_llm(
                    result["transaction_text"], result["similarity"]
                )
                result.update(llm_result)
                llm_call_count += 1

            save_result(row, result)

        batch_count += 1
        print("Batch " + str(batch_num) + " complete.")
        print("")

    print("Pipeline complete.")
    print("")

    summary = get_summary()
    print("Total processed: " + str(summary["total"]))
    print("RAG matches: " + str(summary["rag_matches"]))
    print("LLM analyzed: " + str(summary["llm_analyzed"]))
    print("Clean: " + str(summary["clean"]))
    print("High risk: " + str(summary["high_risk"]))


if __name__ == "__main__":
    run_pipeline()
