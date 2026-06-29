import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


def analyze_with_llm(transaction_text, similarity_score):
    prompt = (
        "You are an AML (Anti-Money Laundering) analyst. "
        "A transaction has been flagged for review because it did not match any known fraud typology with sufficient confidence. "
        "The RAG similarity score was "
        + str(similarity_score)
        + " which is below the confidence threshold.\n\n"
        "Transaction details:\n" + transaction_text + "\n\n"
        "Based on AML knowledge, analyze this transaction and respond in this exact format:\n"
        "TYPOLOGY: <name of closest AML typology or UNKNOWN>\n"
        "RISK_LEVEL: <Low, Medium, High, or Critical>\n"
        "REASONING: <one sentence explanation>\n"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        return parse_llm_response(raw, transaction_text)

    except Exception as e:
        print("LLM call failed: " + str(e))
        return {
            "status": "LLM_ERROR",
            "typology": "UNKNOWN",
            "risk_level": "Medium",
            "reasoning": "LLM call failed: " + str(e),
            "raw_response": "",
        }


def parse_llm_response(raw, transaction_text):
    typology = "UNKNOWN"
    risk_level = "Medium"
    reasoning = ""

    for line in raw.split("\n"):
        if line.startswith("TYPOLOGY:"):
            typology = line.replace("TYPOLOGY:", "").strip()
        elif line.startswith("RISK_LEVEL:"):
            risk_level = line.replace("RISK_LEVEL:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {
        "status": "LLM_ANALYZED",
        "typology": typology,
        "risk_level": risk_level,
        "reasoning": reasoning,
        "raw_response": raw,
    }


if __name__ == "__main__":
    test_text = (
        "Transaction from bank 10 to bank 99. "
        "Payment format: Wire Transfer. "
        "Amount paid: 50000 US Dollar. "
        "Large amount transaction above reporting threshold. "
        "Transaction across different banks."
    )

    result = analyze_with_llm(test_text, similarity_score=0.21)
    print("Status: " + result["status"])
    print("Typology: " + result["typology"])
    print("Risk Level: " + result["risk_level"])
    print("Reasoning: " + result["reasoning"])
