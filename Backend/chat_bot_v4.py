#!/usr/bin/env python3
import os
import sys
import json
import datetime
import pathlib
from dotenv import load_dotenv
from google import genai
from google.genai import types
from elasticsearch import Elasticsearch


load_dotenv()

# ── Gemini LLM setup ─────────────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: set GOOGLE_API_KEY in your .env", file=sys.stderr)
    sys.exit(1)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25")
client = genai.Client(api_key=API_KEY)

ELASTIC_HOST_URL    = os.getenv("ELASTIC_HOST_URL")
ELASTIC_USERNAME  = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD  = os.getenv("ELASTIC_PASSWORD")

try:
    es = Elasticsearch(
        [ELASTIC_HOST_URL], 
        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD)
    )
    if not es.ping():
        raise Exception("Failed to connect to Elasticsearch")
    print("Successfully connected to Elasticsearch")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}", file=sys.stderr)
    client = None

# ── PDF Upload & Caching Setup ───────────────────────────────────────────────
UPLOAD_TTL = int(os.getenv("PDF_UPLOAD_TTL_SECONDS", 14 * 24 * 3600))
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "upload_cache.json")
DATA_DIR = "data"

# ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def load_upload_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            # Reconstruct File objects
            return {
                cat: types.File(
                    uri=info["uri"],
                    mime_type=info.get("mime_type"),
                    name=info.get("name"),
                    display_name=info.get("display_name")
                )
                for cat, info in data.items()
            }
        except Exception:
            return {}
    return {}


def save_upload_cache(cache: dict):
    serializable = {
        cat: {
            "uri": f.uri,
            "name": f.name,
            "mime_type": f.mime_type,
            "display_name": getattr(f, 'display_name', None)
        }
        for cat, f in cache.items()
    }
    with open(CACHE_FILE, "w") as out:
        json.dump(serializable, out)


def upload_all_pdfs() -> dict:
    cache = load_upload_cache()
    # list all .pdf files under data/
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        category = os.path.splitext(fname)[0]
        if category in cache:
            continue
        pdf_path = os.path.join(DATA_DIR, fname)
        try:
            with open(pdf_path, 'rb') as f:
                uploaded = client.files.upload(
                    file=f,
                    config={"mime_type": "application/pdf"}
                )
            cache[category] = uploaded
        except Exception as e:
            print(f"Warning: failed to upload {fname}: {e}", file=sys.stderr)
    save_upload_cache(cache)
    return cache


def upload_single_pdf(category_id: str) -> types.File:
    import sys
    # pdf_path = os.path.join(sys.path[0], f"data/{category_id}.pdf")
    pdf_path = f"./Backend/data/{category_id}.pdf"
    print(pdf_path, "---------------------")


    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found for category '{category_id}'")
    with open(pdf_path, 'rb') as f:
        uploaded = client.files.upload(
            file=f,
            config={"mime_type": "application/pdf"}
        )
    cache = load_upload_cache()
    cache[category_id] = uploaded
    save_upload_cache(cache)
    return uploaded

# Preload cache and uploads
UPLOAD_CACHE = upload_all_pdfs()


def fetch_user_chat(user_id: str) -> list:
    """Grab the most recent chat_history for this user."""
    try:
        resp = es.search(
            index="user_chat",
            body={"size": 1, "query": {"term": {"user_id": user_id}}}
        )
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"].get("chat_history", [])
    except Exception:
        pass
    return []


def ask_chatbot_v3(user_id: str, page_context: str, question: str) -> dict:
    """
    RAG-chatbot with correct File URI & caching:
      • Cache and reload File objects including URI.
      • Ensure File object has uri & mime_type for Gemini.
    """
    # Parse context
    if "|" in page_context:
        category_id, part_name = page_context.split("|", 1)
        page_ctx_str = f"{category_id} (part: {part_name})"
    else:
        category_id = page_context
        page_ctx_str = category_id

    chat_history = fetch_user_chat(user_id)

    # Ensure File
    pdf_file_obj = UPLOAD_CACHE.get(category_id)
    if not pdf_file_obj:
        try:
            pdf_file_obj = upload_single_pdf(category_id)
            UPLOAD_CACHE[category_id] = pdf_file_obj
        except Exception as e:
            raise RuntimeError(f"Unable to upload or find PDF for '{category_id}': {e}")

    # Build prompt
    system_prompt = f"""
You are a expert technical assistant for **{category_id}** spare parts for an automobile industry.
- Use the provided PDF content; do not generate any external info.
- For procedures, extract full step-by-step instructions.
- Embed answers directly, avoid page referances.
- For specifications, locate the exact table or chart, extract the data from matching row(s), and present them either as list type with bullet points.

Always label columns or list items clearly .
- If missing, reply: 'I'm sorry, but I don't have that information.'

Return the final answer in Markdown format only. Always cross check your answer before generating.
"""
    
    chat_history_str = f"""
Chat History:
{json.dumps(chat_history, indent=2)}

Question:
{question}"""

    # Call Gemini
    resp = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1),
        contents=[pdf_file_obj, chat_history_str]
    )
    raw = getattr(resp, "text", None) or resp.get("text") or resp["text"]
    try:
        answer_obj = json.loads(raw)
    except json.JSONDecodeError:
        answer_obj = {"query_answer": raw, "products_code": None}

    print(resp)

    # Persist chat
    entry = {"question": question, "answer": answer_obj.get("query_answer", ""),
             "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
    script = {"source": (
            "if (ctx._source.chat_history == null) {"
            "ctx._source.chat_history = [params.entry];}"
            " else {ctx._source.chat_history.add(params.entry);}"
        ), "lang": "painless", "params": {"entry": entry}}
    try:
        es.update(index="user_chat", id=user_id,
                  body={"script": script, "upsert": {"user_id": user_id, "chat_history": [entry]}})
    except Exception:
        print("Warning: failed to save chat history", file=sys.stderr)

    return answer_obj




# CLI entrypoint
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test chatbot_v3 with correct File URI & caching")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--page-context", required=True,
                        help="e.g. 'Lubricator' or 'Lubricator|AL24P-060AS'")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    try:
        out = ask_chatbot_v3(args.user_id, args.page_context, args.query)
        print(json.dumps(out, indent=2))
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
