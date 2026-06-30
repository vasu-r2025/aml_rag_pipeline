from dotenv import load_dotenv
import time

load_dotenv(dotenv_path=".env")

from pipeline.ingestor import load_transactions
from pipeline.rag_engine import query_rag, add_to_knowledge_base, collection
from pipeline.llm_handler import analyze_with_llm
from db.storage import init_db, save_result, save_batch_run, get_summary

BATCH_LIMIT = 3
LLM_ESCALATION_LIMIT = 29


def run_pipeline():
    print("Initializing database...")
    init_db()

    print("Starting AML pipeline...")
    print("")

    batch_count = 0

    for batch_num, chunk in load_transactions(chunksize=200):
        if batch_count >= BATCH_LIMIT:
            print("Batch limit reached. Stopping.")
            break

        llm_call_count = 0
        batch_total = 0
        batch_rag_resolved = 0
        batch_llm_escalated = 0
        batch_llm_unknown = 0

        print(
            "Processing batch "
            + str(batch_num)
            + " with "
            + str(len(chunk))
            + " transactions..."
        )

        for _, row in chunk.iterrows():
            result = query_rag(row)
            batch_total += 1

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
                time.sleep(1)
                batch_llm_escalated += 1

                if llm_result["status"] == "LLM_ERROR":
                    print("  -> LLM call failed, not counted as unknown")
                elif llm_result["typology"].upper() == "UNKNOWN":
                    batch_llm_unknown += 1

                learned = add_to_knowledge_base(llm_result, result["transaction_text"])
                if learned:
                    print(
                        "  -> New pattern written to knowledge base: "
                        + llm_result["typology"]
                    )
            else:
                batch_rag_resolved += 1

            save_result(row, result)

        kb_size = collection.count()
        save_batch_run(
            batch_number=batch_num,
            total_transactions=batch_total,
            rag_resolved=batch_rag_resolved,
            llm_escalated=batch_llm_escalated,
            llm_unknown=batch_llm_unknown,
            knowledge_base_size=kb_size,
        )

        batch_count += 1
        print("Batch " + str(batch_num) + " complete.")
        print("  RAG resolved: " + str(batch_rag_resolved))
        print("  LLM escalated: " + str(batch_llm_escalated))
        print("  LLM unknown: " + str(batch_llm_unknown))
        print("  Knowledge base size: " + str(kb_size))
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
