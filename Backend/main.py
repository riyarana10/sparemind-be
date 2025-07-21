# main.py
import os
import sys
sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Request, Depends, status, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
# from chatbot import ask_chatbot  
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from elasticsearch import Elasticsearch
from datetime import timedelta
from auth import verify_password, create_access_token, get_password_hash, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from jose import JWTError, jwt, ExpiredSignatureError
from fastapi import Query
import re
from fastapi import Depends
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime
from typing import Optional
from fastapi import HTTPException, status
import traceback
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from aws import S3Manager
from fastapi.concurrency import run_in_threadpool
import asyncio
import traceback

from typing import List, Dict

from chat_bot_v4 import ask_chatbot_v3, fetch_user_chat

from vanna_chat_v2 import vn, format_output_as_markdown_with_gemini

class ChatPayload(BaseModel):
     page_context: str
     query:        str

app = FastAPI()

INDEX = "part-search-v23"
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ELASTIC_HOST_URL    = os.getenv("ELASTIC_HOST_URL")
ELASTIC_USERNAME  = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD  = os.getenv("ELASTIC_PASSWORD")

load_dotenv()

INDEX_NAME = "part-search-v23"
USER_INDEX          = "pg-users"
TRANSACTIONS_IDX    = "pg-transactions"
SECRET_KEY        = os.getenv("SECRET_KEY")
ALGORITHM         = "HS256"

# S3 Configuration (Consider moving sensitive keys to .env)
S3_BUCKET_NAME    = os.getenv("S3_BUCKET_NAME", "partsgenie-data")
S3_AWS_ACCESS_KEY = os.getenv("S3_AWS_ACCESS_KEY")
S3_AWS_SECRET_KEY = os.getenv("S3_AWS_SECRET_KEY")
S3_REGION         = os.getenv("S3_REGION", "us-west-2")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

try:
    client = Elasticsearch(
        [ELASTIC_HOST_URL], 
        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD)
    )
    if not client.ping():
        raise Exception("Failed to connect to Elasticsearch")
    print("Successfully connected to Elasticsearch")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}", file=sys.stderr)
    client = None

# Initialize S3 client
s3_client = None
try:
    session = boto3.Session(
        aws_access_key_id=S3_AWS_ACCESS_KEY,
        aws_secret_access_key=S3_AWS_SECRET_KEY,
        region_name=S3_REGION
    )
    s3_client = session.client('s3')
    # Test connection by listing buckets (optional, requires ListBuckets permission)
    # s3_client.list_buckets()
    print("Successfully connected to S3")
except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
    print(f"Error connecting to S3: {e}", file=sys.stderr)
    s3_client = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to decode the JWT token and return user info.
# ── Auth / User helpers ──────────────────────────────────────────────────────
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role     = payload.get("role")
        user_id  = payload.get("user_id")   # ← extract user_id
        if not username or not role:
            raise JWTError()
        return {"username": username, "role": role, "user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_username(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
def fetch_vanna_chat(user_id: str) -> list:
    """Grab the most recent chat_history for this user."""
    try:
        resp = client.search(
            index="pg-vanna_chat",
            body={"size": 1, "query": {"term": {"user_id": user_id}}}
        )
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"].get("chat_history", [])
    except Exception:
        pass
    return []



class DecisionPayload(BaseModel):
    original_part_name: str
    replacement_part_name: str
    accepted: bool
    rejected: bool
    comment: str

# ── Decision → Transaction Endpoint ───────────────────────────────────────────
@app.post("/decision")
async def record_decision(
    payload: DecisionPayload,
    current: dict = Depends(get_current_user)
):
    username = current["username"]
    user_id  = current["user_id"]

    # 1) update main parts index
    update_body = {
        "script": {
            "source": """
                ctx._source.accepted = params.accepted;
                ctx._source.rejected = params.rejected;
                ctx._source.comment  = params.comment;
                ctx._source.modified_by = params.user;
            """,
            "lang": "painless",
            "params": {
                "accepted": payload.accepted,
                "rejected": payload.rejected,
                "comment":  payload.comment,
                "user":     username
            }
        },
        "query": {
            "bool": {
                "must": [
                    {"term": {"original_part_name": payload.original_part_name}},
                    {"term": {"replacement_part_name": payload.replacement_part_name}}
                ]
            }
        }
    }
    client.update_by_query(index=INDEX_NAME, body=update_body)

    # 2) fetch that doc for its price_difference and locations/category
    part_src = {}
    savings  = 0.0
    try:
        search_body = {"size": 1, "query": update_body["query"]}
        r = client.search(index=INDEX_NAME, body=search_body)
        hits = r.get("hits", {}).get("hits", [])
        if hits:
            part_src = hits[0]["_source"]
            savings  = part_src.get("price_difference", 0.0)
    except:
        pass

    # 3) (optional) legacy decisions index logging…
    #    … your existing code here if you still want it …

    # 4) new transactions index
    txn = {
        "user_id":                   user_id,
        "original_part_name":   payload.original_part_name,
        "replacement_part_name":payload.replacement_part_name,
        "original_part_location":    part_src.get("original_part_location"),
        "replacement_part_location": part_src.get("replacement_part_location"),
        "category":                  part_src.get("category"),
        "status":                    ("accepted" if payload.accepted else
                                      "rejected" if payload.rejected else
                                      "commented"),
        "comment":                   payload.comment,
        "price_difference":          savings,
        "updated_at":                datetime.utcnow().isoformat()
    }
    try:
        client.index(index=TRANSACTIONS_IDX, document=txn)
    except Exception as e:
        print("Failed to index transaction:", e)

    return {"status": "ok"}


# ── Fetch Transactions ─────────────────────────────────────────────────────────
@app.get("/transactions")
async def get_transactions(
    user_id: Optional[str] = Query(None, description="Filter by user_id"),
    size:    int           = Query(10,   ge=1,   description="How many to return")
):
    if not client:
        return {"transactions": []}

    # build the ES query
    if user_id:
        # exact term on the keyword field
        query = {"term": {"user_id": user_id}}
    else:
        query = {"match_all": {}}

    body = {
        "size": size,
        "query": query,
        "sort": [{"updated_at": {"order": "desc"}}]
    }

    # debug print so you can verify the request in your logs
    print(f"🔍 /transactions → body: {body}")

    try:
        resp = client.search(index=TRANSACTIONS_IDX, body=body)
        txns = [hit["_source"] for hit in resp["hits"]["hits"]]
        return {"transactions": txns}
    except Exception as e:
        print("❌ Transaction search failed:", e)
        raise HTTPException(status_code=500, detail="Could not fetch transactions")
  
@app.delete("/users/{username}")
async def delete_user(username: str, current_user=Depends(get_current_user)):
    if current_user["role"]!="admin":
        raise HTTPException(403)
    client.delete(index=USER_INDEX, id=username, ignore=[404])
    return {"status":"deleted"}

@app.patch("/users/{username}/role")
async def change_role(username: str, new_role: str, current_user=Depends(get_current_user)):
    if current_user["role"]!="admin":
        raise HTTPException(403)
    client.update(
      index=USER_INDEX,
      id=username,
      body={"doc":{"role":new_role}}
    )
    return {"status":"role_updated"}


# -------------------- Existing Endpoints --------------------
@app.get("/test/parts")
async def get_all_test_parts():
    sample_parts_data = [
        {"original_part_item_code": "MA0BN03K000", "description": "Engine Oil Filter"},
        {"original_part_item_code": "XYZ12345", "description": "Brake Pads"},
        {"original_part_item_code": "ABC9876", "description": "Spark Plug"},
    ]
    return {"results": sample_parts_data}

@app.get("/test/parts/{item_code}")
async def get_test_part_by_code(item_code: str):
    sample_parts_data = [
        {"original_part_item_code": "MA0BN03K000", "description": "Engine Oil Filter"},
        {"original_part_item_code": "XYZ12345", "description": "Brake Pads"},
        {"original_part_item_code": "ABC9876", "description": "Spark Plug"},
    ]
    for part in sample_parts_data:
        if part["original_part_item_code"] == item_code:
            return {"results": [part]}
    raise HTTPException(status_code=404, detail="Part not found")

async def get_category_details(category_name: str) -> dict:
    """Fetch msil_category and image from pg-categories_v5 index"""
    if not client:
        return {}
    
    query = {
        "size": 1,
        "query": {
            "term": {"name": category_name.lower()}
        }
    }
    try:
        resp = client.search(index="pg-categories_v5", body=query)
        if resp["hits"]["total"]["value"] > 0:
            return resp["hits"]["hits"][0]["_source"]
    except Exception as e:
        print(f"Error fetching category details: {e}")
    return {"msil_category": None, "image": None}



@app.get("/search_exact")
async def search_exact(
    q: str = Query(..., description="Exact code or exact name to look up (original or replacement)"),
    max_hits: int = Query(20, ge=1, le=100, description="Max hits to fetch")
):
    if not client:
        raise HTTPException(503, "Elasticsearch unavailable")

    term = q.upper().strip()

    # 1) Try exact ORIGINAL lookup (code OR name)
    orig_body = {
        "size": max_hits,
        "query": {
            "bool": {
                "should": [
                    {"term": {"original_part_item_code": term}},
                    {"term": {"original_part_name":      term}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    # print(f"/search_exact → original lookup, body={orig_body}")
    resp = client.search(index=INDEX_NAME, body=orig_body)
    hits = resp["hits"]["hits"]

    if hits:
        src = hits[0]["_source"]
        original = src
        category = src.get("category")

        category_details = await get_category_details(category)
        msil_category = category_details.get("msil_category")
        category_image = category_details.get("image")

        replacements = []
        
        # Check if original has built-in replacement info
        if src.get("replacement_part_name") not in (None, "", "N/A"):
            # Add the original document itself as a replacement if it has replacement info
            replacements.append(src)
        
        # Also fetch any additional replacements for this original
        repl_body = {
            "size": max_hits,
            "query": {
                "term": {"original_part_item_code": src["original_part_item_code"]}
            },
            "sort": [{"price_difference": {"order": "desc"}}]
        }
        print(f"/search_exact → fetching replacements, body={repl_body}")
        rresp = client.search(index=INDEX_NAME, body=repl_body)
        additional_replacements = [
            h["_source"] for h in rresp["hits"]["hits"]
            if h["_source"].get("replacement_part_item_code") not in (None, "", "N/A")
            and h["_source"].get("unique_id") != src.get("unique_id")  # Avoid duplicating the original if it's already added
        ]
        
        replacements.extend(additional_replacements)

        return {
            "original": original,
            "replacements": replacements,
            "category": {"name": category,
                "msil_category": msil_category,
                "image": category_image}
        }

    # 2) No original → try exact REPLACEMENT lookup (code OR name)
    repl_body = {
        "size": max_hits,
        "query": {
            "bool": {
                "should": [
                    {"term": {"replacement_part_item_code": term}},
                    {"term": {"replacement_part_name": term}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    print(f"/search_exact → replacement lookup, body={repl_body}")
    rresp = client.search(index='part-search-v23', body=repl_body)
    r_hits = rresp["hits"]["hits"]

    if not r_hits:
        return {"original": None, "replacements": [], "category": None}

    # first replacement hit
    rep_src = r_hits[0]["_source"]
    category = rep_src.get("category")

    category_details = await get_category_details(category)
    msil_category = category_details.get("msil_category")
    category_image = category_details.get("image")

    # fetch its original
    orig_lookup = {
        "size": 1,
        "query": {
            "term": {"original_part_item_code": rep_src["original_part_item_code"]}
        }
    }
    print(f"/search_exact → fetching parent original, body={orig_lookup}")
    oresp = client.search(index=INDEX_NAME, body=orig_lookup)
    o_hits = oresp["hits"]["hits"]
    original = o_hits[0]["_source"] if o_hits else None

    # return all exact‐matching replacements
    replacements = [h["_source"] for h in r_hits]

    return {
        "original": original,
        "replacements": replacements,
        "category": {"name": category,
                "msil_category": msil_category,
                "image": category_image}
    }


# in main.py (just below your existing /search_exact and /search_all endpoints)

async def get_bulk_category_details(category_names: List[str]) -> Dict[str, dict]:
    """
    Fetches details for a list of category names in a single bulk Elasticsearch query.
    Returns a dictionary mapping original category_name to its details.
    """
    if not client or not category_names:
        return {}

    # Filter out None values and create case mapping dictionaries
    original_to_lower_map = {name: str(name).lower() for name in category_names if name is not None}
    lower_to_original_map = {v: k for k, v in original_to_lower_map.items()}
    query_category_names = list(original_to_lower_map.values())

    if not query_category_names:
        return {}

    # Use a terms query with lowercase category names
    query = {
        "size": len(query_category_names) + 50,  # Ensure we get all matches
        "query": {"terms": {"name.keyword": query_category_names}}  # Use .keyword for exact matching
    }
    
    # Initialize result dictionary with default values
    category_details_map = {
        orig_name: {"msil_category": None, "image": None, "name": orig_name}
        for orig_name in category_names if orig_name is not None
    }

    try:
        # Execute the query
        resp = await run_in_threadpool(client.search, index="pg-categories_v5", body=query)
        
        # Process each hit
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            name_from_doc = source.get("name")
            
            if name_from_doc:
                name_from_doc_lower = str(name_from_doc).lower()
                
                # Try to find the original case version of this category
                if name_from_doc_lower in lower_to_original_map:
                    original_casing_name_key = lower_to_original_map[name_from_doc_lower]
                    
                    # Update the details map with data from Elasticsearch
                    category_details_map[original_casing_name_key] = {
                        "name": source.get("name"),
                        "msil_category": source.get("msil_category"),
                        "image": source.get("image")
                    }
                    
                    # Debug output for successful matches
                    print(f"Matched category: '{name_from_doc}' to '{original_casing_name_key}'")
                    print(f"  msil_category: {source.get('msil_category')}")
                    print(f"  image: {source.get('image')}")
                else:
                    # Debug output for unmatched categories
                    print(f"Category '{name_from_doc}' from pg-categories_v5 not in lower_to_original_map.")
                    print(f"Available keys: {list(lower_to_original_map.keys())[:5]}...")
    except Exception as e:
        print(f"Error fetching bulk category details: {e}")
        # Add additional error details
        import traceback
        traceback.print_exc()
    
    return category_details_map


@app.get("/search_all")
async def search_all(
    q: str = Query(..., alias="query", description="Search term (code/name/category/brand)"),
    max_hits: int = Query(30, ge=1, le=1000, description="Max number of original parts to return") # Default to 30
):
    if not client:
        raise HTTPException(503, "Elasticsearch unavailable")

    processed_query_lower = q.strip().lower()
    processed_query_upper = q.strip().upper()

    # --- Elasticsearch Query Construction ---
    
    # Clauses for general matching across multiple fields
    general_match_clauses = [
        {"multi_match": {
            "query": processed_query_lower,
            "type": "best_fields", # Finds docs that match any field well
            "fields": [
                "category^3", # Boost category matches in general search
                "original_part_item_code.keyword^2", # Exact item code
                "original_part_name^2",
                "item_description^1",
                "brand.keyword^1.5"
            ],
            "fuzziness": "AUTO",
            "prefix_length": 1,
            "max_expansions": 50,
            "minimum_should_match": "70%" # Require a good portion of terms to match for multi_match
        }},
        {"multi_match": { # A cross_fields query for better phrase-like matching on name and description
            "query": processed_query_lower,
            "type": "cross_fields",
            "fields": ["original_part_name", "item_description"],
            "operator": "and", # All terms in the query must be present in at least one field
            "boost": 1.5
        }},
        # Prefix queries for "type-ahead" like behavior
        {"prefix": {"category.keyword": {"value": processed_query_lower, "case_insensitive": True, "boost": 2.0}}},
        {"prefix": {"original_part_item_code.keyword": {"value": processed_query_upper, "boost": 1.5}}},
        {"prefix": {"original_part_name": {"value": processed_query_lower, "case_insensitive": True, "boost": 1.0}}},
        {"prefix": {"brand.keyword": {"value": processed_query_lower, "case_insensitive": True, "boost": 1.0}}},
    ]

    # Function score to heavily boost exact matches on key fields
    # These filters identify documents that should be ranked much higher.
    exact_match_boost_functions = [
        { # Very high boost if the query IS an exact category
            "filter": {"term": {"category.keyword": processed_query_lower}},
            "weight": 100 
        },
        { # Very high boost if the query IS an exact original part item code
            "filter": {"term": {"original_part_item_code.keyword": processed_query_upper}},
            "weight": 90
        },
        { # High boost if the query IS an exact original part name
            "filter": {"term": {"original_part_name.keyword": processed_query_lower}}, # Assumes original_part_name.keyword exists
            "weight": 80
        },
        { # Boost if the query is a significant part of the category (using match_phrase)
          "filter": {"match_phrase": {"category": processed_query_lower}},
          "weight": 50
        }
    ]

    es_query_body = {
        "size": max_hits,
        "query": {
            "function_score": {
                "query": { # The base query that finds potential candidates
                    "bool": {
                        "should": general_match_clauses,
                        # If the query is very short, require at least one clause to match.
                        # If longer, more can contribute.
                        "minimum_should_match": 1
                    }
                },
                "functions": exact_match_boost_functions,
                "score_mode": "sum", # Add weights of matching functions to the base query score
                "boost_mode": "sum"  # How the function scores are combined with the query score
            }
        },
        # Primary sort is by score. No secondary sort needed if relevance is handled well.
        "sort": [{"_score": {"order": "desc"}}],
        "highlight": {
            "fields": {
                "category": {"number_of_fragments": 0}, # Highlight whole field
                "original_part_name": {},
                "item_description": {}
            },
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"] # Example highlight tags
        },
        "aggs": {
            "top_categories": {"terms": {"field": "category.keyword", "size": 15}},
            "top_brands": {"terms": {"field": "brand.keyword", "size": 15}}
        }
    }

    try:
        # print(f"Elasticsearch Query for /search_all: {json.dumps(es_query_body, indent=2)}")
        resp = await run_in_threadpool(client.search, index=INDEX_NAME, body=es_query_body)
    except Exception as e:
        print(f"Elasticsearch query failed for /search_all: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search service query failed: {str(e)}")

    hits = resp.get("hits", {}).get("hits", [])
    # print(f"Total hits from ES: {resp.get('hits', {}).get('total', {}).get('value', 0)}")


    # --- Data Processing ---
    originals = []
    unique_categories_from_hits = set()
    unique_brands_from_hits = set()

    # If the user searched for a specific category, try to put those items first
    # This is a post-processing sort, ES function_score should ideally handle this.
    # However, for absolute certainty if query was "air filter", we can do an extra client-side sort.
    is_exact_category_search = any(
        func["filter"].get("term", {}).get("category.keyword") == processed_query_lower
        for func in exact_match_boost_functions
    )
    
    # Python-side sorting: if the query was an exact category, push those to the front.
    # This is a fallback or supplement to Elasticsearch's scoring.
    if is_exact_category_search:
        hits.sort(key=lambda h: (
            str(h['_source'].get('category', '')).lower() != processed_query_lower, # False (0) for matching category, True (1) for non-matching
            -h.get('_score', 0) # Then by ES score descending
        ))


    for h in hits:
        src = h["_source"]
        cat = src.get("category")
        brand = src.get("brand")
        
        if cat: unique_categories_from_hits.add(cat)
        if brand: unique_brands_from_hits.add(brand)

        highlight_data = h.get("highlight", {})
        
        original_entry = {
            "original_part_item_code": src.get("original_part_item_code"),
            "original_part_name": src.get("original_part_name"),
            "original_part_location": src.get("original_part_location"),
            "original_part_stock": src.get("original_part_stock"),
            "original_part_price": src.get("original_part_price"),
            "original_part_image": src.get("original_part_image"), # Image for the original part itself
            "original_part_name_breakdown_definition": src.get("original_part_name_breakdown_definition"),
            "brand": brand,
            "category": cat, # Category of the original part
            "item_description": src.get("item_description"),
            "_score": h.get("_score", 0),
            "highlight": {
                "category": highlight_data.get("category", [cat if cat else ""])[0],
                "original_part_name": highlight_data.get("original_part_name", [src.get("original_part_name","")])[0],
                "item_description": highlight_data.get("item_description", [src.get("item_description","")])[0],
            }
        }
        originals.append(original_entry)

    # Consolidate categories from hits and aggregations for the filter panel
    panel_categories_set = set(unique_categories_from_hits)
    if "aggregations" in resp and "top_categories" in resp["aggregations"]:
        agg_buckets = resp["aggregations"]["top_categories"].get("buckets", [])
        for b in agg_buckets:
            if b.get("key"): panel_categories_set.add(b["key"])
    
    str_panel_categories_list = [str(c) for c in panel_categories_set if c is not None]
    category_details_map = await get_bulk_category_details(str_panel_categories_list)

    categories_response_list = []
    for cat_name_str in str_panel_categories_list:
        details = category_details_map.get(cat_name_str, {"name": cat_name_str, "msil_category": None, "image": None})
        categories_response_list.append({
            "name": details.get("name", cat_name_str),
            "msil_category": details.get("msil_category"),
            "image": details.get("image") # This is the image for the CATEGORY, not the part
        })
        # if not details.get("image"):
            # print(f"Debug: No image for category (in panel) '{cat_name_str}'. Mapped details: {details}")

    # Consolidate brands for the filter panel
    panel_brands_set = set(unique_brands_from_hits)
    if "aggregations" in resp and "top_brands" in resp["aggregations"]:
        brand_agg_buckets = resp["aggregations"]["top_brands"].get("buckets", [])
        for b in brand_agg_buckets:
            if b.get("key"): panel_brands_set.add(b["key"])
    
    return {
        "originals":    originals, # Only original parts, no replacements list
        # "replacements": [], # Explicitly showing it's removed or just omit
        "categories":   categories_response_list, # For the filter panel
        "brands":       [str(b) for b in panel_brands_set if b is not None] # For the filter panel
    }

from fastapi import Depends


@app.get("/popular-parts")
async def get_featured_parts_detailed():
    if not client:
        return {"featured_parts": []}

    required_fields = [
        "original_part_image",
        "original_part_item_code",
        "item_description",
        "original_part_name",
        "original_part_location",
        "original_part_stock",
        "category",
        "brand",
        "original_part_price",
        "original_part_name_breakdown_definition"
    ]

    try:
        query_body = {
            "size": 5,
            "_source": required_fields,
            "query": {
                "terms": {
                    "original_part_item_code": [
                         "M4S06012MG4",
                        "MA0UI01G000",
                        "MA00F0D0003",
                        "MA0FF035000",
                        "MA0BZ002000",
                        "MA0OL0C8000"
                    ]
                }
            },
            "collapse": {
                "field": "original_part_item_code",
            }
        }
        
        resp = client.search(index=INDEX_NAME, body=query_body)
        hits = resp["hits"]["hits"]
        
        featured_parts = [{"original": [hit["_source"]]} for hit in hits] 
        
        return {"featured_parts": featured_parts}
        
    except Exception as e:
        print(f"Failed to fetch popular parts: {e}")
        return {"featured_parts": [], "error": str(e)}

@app.get("/search")
async def search_parts(original_part_item_code: str = Query(..., description="Original part item code to search for")):
    if not original_part_item_code:
        return {"results": []}
    if client:
        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {"term": {"original_part_item_code": original_part_item_code}},
                        {"term": {"original_part_name": original_part_item_code}}
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        try:
            response = client.search(index=INDEX_NAME, body=search_query)
            hits = response["hits"]["hits"]
            results = [hit["_source"] for hit in hits]
            return {"results": results}
        except Exception as e:
            print(f"Elasticsearch query failed: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="Search service unavailable")
    else:
        return {"results": []}


@app.get("/autocomplete")
async def autocomplete(query: str = Query(...)):
    print(f"[DEBUG /autocomplete] called with query={query!r}")
    print(f"[DEBUG /autocomplete] hitting ES index={INDEX_NAME}")
    body = {
      "size": 5,
      "query": {
        "bool": {
          "should": [
            {
              "prefix": {
                "original_part_item_code": {
                  "value": query.upper(),
                  "case_insensitive": True
                }
              }
            },
            {
              "prefix": {
                "original_part_name": {
                  "value": query.upper(),
                  "case_insensitive": True
                }
              }
            },
            {
              "prefix": {
                "category": {
                  "value": query.upper(),
                  "case_insensitive": True
                }
              }
            },
            {
              "prefix": {
                "brand": {
                  "value": query.upper(),
                  "case_insensitive": True
                }
              }
            }
          ]
        }
      }
    }
    print(f"[DEBUG /autocomplete] ES body ➞ {body}")
    resp = client.search(index=INDEX_NAME, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    print(f"[DEBUG /autocomplete] ES returned {len(hits)} hits")
    print("[DEBUG /autocomplete] hits codes =", [h["_source"]["original_part_item_code"] for h in hits])
    suggestions = [
      {
        "original_part_item_code": h["_source"]["original_part_item_code"],
        "original_part_name": h["_source"].get("original_part_name", ""),
        "category": h["_source"].get("category", ""),
        "brand": h["_source"].get("brand", "")
      }
      for h in hits
    ]
    print(f"[DEBUG /autocomplete] suggestions ➞ {suggestions}")
    return {"suggestions": suggestions}

@app.get("/")
async def read_root():
    return {"message": "Part Genie API is running!"}

# ------------------ New Endpoint: Text Search ------------------
@app.get("/search_text")
async def search_text(q: str = Query(..., description="Text query for parts search")):
    if client:
        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"original_part_name": {"query": q, "fuzziness": "AUTO"}}},
                        {"match": {"category": {"query": q, "fuzziness": "AUTO"}}},
                        {"match": {"brand": {"query": q, "fuzziness": "AUTO"}}},
                        {"match": {"item_description": {"query": q, "fuzziness": "AUTO"}}}
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        try:
            response = client.search(index=INDEX_NAME, body=search_query)
            hits = response["hits"]["hits"]
            results = [hit["_source"] for hit in hits]
            return {"results": results}
        except Exception as e:
            print(f"Elasticsearch query failed: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="Search service unavailable")
    else:
        return {"results": []}
    
# ------------------ New Endpoint: Get Categories ------------------

@app.get("/all-categories")
async def all_categories():
    """
    Return a list of {"name", "msil_category"} for all categories where active=True.
    """
    if not client:
        return {"categories": []}

    try:
        body = {
            "size": 10000,                  # adjust if you expect more than 10k docs
            "_source": ["name", "msil_category"],
            "query": {
                "term": {
                    "active": True
                }
            }
        }
        resp = client.search(index="pg-categories_v5", body=body)
        hits = resp["hits"]["hits"]
        categories = [
            {
                "name":            h["_source"]["name"],
                "msil_category":   h["_source"]["msil_category"],
            }
            for h in hits
        ]
        return {"categories": categories}

    except Exception as e:
        print(f"Failed to fetch all categories: {e}")
        raise HTTPException(500, "Could not fetch all categories")


@app.get("/categories")
async def list_categories(
    size: int = Query(5, ge=1, description="How many categories to return")
):
    """
    Return the top `size` categories, sorted by stored `rankscore` descending.
    """
    if not client:
        return {"categories": []}

    try:
        body = {
            "size": size,
            "sort": [
                {
                    "category_score": {
                        "order": "desc"
                    }
                }
            ]
        }
        resp = client.search(index="pg-categories_v5", body=body)
        hits = resp["hits"]["hits"]
        categories = [
            {
                "name":     h["_source"]["name"],
                "category_score":  h["_source"].get("category_score", 0),
                "msil_category": h["_source"]["msil_category"],
                "image":    h["_source"].get("image", ""),
            }
            for h in hits
        ]
        return {"categories": categories}

    except Exception as e:
        print(f"Failed to fetch categories: {e}")
        raise HTTPException(500, "Could not fetch categories")

    
# ------------------ New Endpoint: Get Category Parts ------------------
@app.get("/search_by_category")
async def search_by_category(
    category: str = Query(..., description="Exact category name to match"),
    distinct_on_field: str = Query("original_part_item_code"),
    size: int = Query(50, ge=1, le=500, description="Number of results to return"),
):
    """
    Return up to `size` parts with exact category matches,
    sorted by price_difference descending.
    """
    if not client:
        return {"results": []}

    
    processed_category = category.strip().lower()
    
    body = {
        "size": size,
        "query": {
            "term": {
                "category": { 
                    "value": processed_category,
                    "case_insensitive": True
                }
            }
        },
        "collapse": {
            "field": distinct_on_field
        },
        "sort": [
            {"price_difference": {"order": "desc"}}
        ]
    }

    try:
        resp = client.search(index="part-search-v23", body=body)
        hits = resp["hits"]["hits"]
        results = [h["_source"] for h in hits]
        return {"results": results}
    except Exception as e:
        print(f"Exact category search failed: {e}")
        raise HTTPException(status_code=500, detail="Category search unavailable")

# ── Fetch PDF links from S3 by Category ───────────────────────────────────────
@app.get("/pdf_link")
async def get_pdf_links(
    category_id: str = Query(..., min_length=1, description="Category ID (S3 folder name)"),
    series_name: str = Query(..., min_length=1, description="Series name (subfolder name)"),
):
    """
    Fetch presigned URLs for PDFs in a specific category/series folder
    """
    try:
        s3_manager = S3Manager()
        
        # Validate S3 client initialization
        if not s3_manager.s3_boto_client:
            raise HTTPException(
                status_code=500,
                detail="S3 service unavailable - check server configuration"
            )

        # Get PDF keys
        pdf_keys = s3_manager.list_pdfs_in_category(series_name, category_id)
        
        if not pdf_keys:
            raise HTTPException(
                status_code=404,
                detail=f"No PDFs found for category '{category_id}' and series '{series_name}'"
            )

        # Generate presigned URLs
        presigned_urls = s3_manager.generate_presigned_urls(pdf_keys)
        
        if not presigned_urls:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate access URLs for found PDFs"
            )

        return {"pdf_links": presigned_urls}

    except ClientError as e:
        error_msg = e.response.get('Error', {}).get('Message', 'Unknown S3 error')
        print(f"S3 Error: {error_msg}")
        raise HTTPException(
            status_code=503,
            detail=f"S3 service error: {error_msg}"
        )
        
    except HTTPException:
        # Re-raise already handled exceptions
        raise
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error - check application logs"
        )

# ------------------ New Endpoint: Login ------------------
# ── Login / User Endpoints ───────────────────────────────────────────────────
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not client:
        raise HTTPException(status_code=500, detail="Elasticsearch not available")

    # 0) Show which index we're hitting
    print(f"Login → using USER_INDEX = '{USER_INDEX}'")

    # 1) Check index existence
    exists = client.indices.exists(index=USER_INDEX)
    print(f"Login → does index '{USER_INDEX}' exist? -> {exists}")
    if not exists:
        raise HTTPException(status_code=500, detail=f"Index '{USER_INDEX}' not found")

    # 2) Dump the mapping so you can verify the field names
    mapping = client.indices.get_mapping(index=USER_INDEX)
    print(f"Login → mapping for '{USER_INDEX}':\n{mapping}")

    # 3) Use a strict term query on the keyword
    username_lower = form_data.username.lower()
    query_body = {
      "query": {
        "term": {
          "username": {
            "value": username_lower
          }
        }
      }
    }
    print("Login → ES query body:", query_body)

    try:
        resp = client.search(index='pg-users', body=query_body)
    except Exception as e:
        print("Error querying pg-users:", e)
        raise HTTPException(status_code=500, detail="User search failed")

    hits = resp.get("hits", {}).get("hits", [])
    print("Login → ES response hits:", [h["_source"] for h in hits])

    if not hits:
        # nothing matched
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    user = hits[0]["_source"]
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # 4) Everything matched, issue token (including user_id)
    access_token = create_access_token(
      data={
        "sub":     user["username"],
        "role":    user["role"],
        "user_id": user.get("id"),
      },
      expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    print("Login → issuing token for", user["username"])
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/verify-token")
async def verify_token(authorization: str = Header(...)):
    """
    Verify if the token passed in Authorization header is expired.
    The Authorization header should be like: "Bearer <token>"
    """
    try:
        # Extract token from 'Bearer <token>'
        token = authorization.split(" ")[1]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid authorization header")

    try:
        # Decode token without verifying expiration to get payload
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # If decode successful and no ExpiredSignatureError, token not expired
        return {"expired": False}
    except ExpiredSignatureError:
        # Token has expired
        return {"expired": True}
    except JWTError:
        # Token is invalid for other reasons
        raise HTTPException(status_code=401, detail="Invalid token")




# ------------------ New Endpoint: Create New User (Admin Only) ------------------
@app.post("/users")
async def create_user(username: str, password: str, role: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create users")
    
    new_user = {
        "username": username.lower(),
        "hashed_password": get_password_hash(password),
        "role": role
    }
    try:
        response = client.index(index='pg-users', document=new_user)
        return {"message": "User created successfully", "result": response}
    except Exception as e:
        print("Error creating user:", e, file=sys.stderr)
        raise HTTPException(status_code=500, detail="User creation failed")
    
class ChatMessage(BaseModel):
    sender: str
    text:   str

class HistoryResponse(BaseModel):
    history: List[ChatMessage]

@app.get("/chat/history", response_model=HistoryResponse)
async def get_chat_history(current: dict = Depends(get_current_user)):
    """
    Returns the flattened chat history for the current user as a
    list of { sender, text } messages.
    """
    user_id = current["user_id"]
    try:
        # fetch_user_chat returns a list of {question,answer,timestamp}
        raw = fetch_user_chat(user_id)
    except Exception:
        raw = []

    # build your messages array
    msgs = []
    for entry in raw:
        msgs.append(ChatMessage(sender="user", text=entry["question"]))
        msgs.append(ChatMessage(sender="bot",  text=entry["answer"]))

    return {"history": msgs}


@app.get("/vanna_chat/history", response_model=HistoryResponse)
async def get_chat_history(current: dict = Depends(get_current_user)):
    """
    Returns the flattened chat history for the current user as a
    list of { sender, text } messages.
    """
    user_id = current["user_id"]
    try:
        # fetch_user_chat returns a list of {question,answer,timestamp}
        raw = fetch_vanna_chat(user_id)
    except Exception:
        raw = []

    # build your messages array
    msgs = []
    for entry in raw:
        msgs.append(ChatMessage(sender="user", text=entry["question"]))
        msgs.append(ChatMessage(sender="bot",  text=entry["answer"]))

    return {"history": msgs}
# New payload for resetting chat
class ResetPayload(BaseModel):
    user_id: str

@app.post("/chat/reset", status_code=200)
async def reset_chat_endpoint(
     current: dict = Depends(get_current_user)
 ):
    """
    Clears the nested chat_history array for this user in ES.
    """
    user_id = current["user_id"]
    try:
        client.update(
            index="pg-user_chat",
            id=user_id,
            body={"doc": {"chat_history": []}}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    return {"status": "ok"}


@app.post("/vanna_chat/reset", status_code=200)
async def reset_chat_endpoint(
     current: dict = Depends(get_current_user)
 ):
    """
    Clears the nested chat_history array for this user in ES.
    """
    user_id = current["user_id"]
    try:
        client.update(
            index="pg-vanna_chat",
            id=user_id,
            body={"doc": {"chat_history": []}}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    return {"status": "ok"}



# @app.post("/ask")
# async def ask_question(request: Request):
#     try:
#         # Parse JSON input
#         try:
#             data = await request.json()
#             user_query = data.get("question", "").strip()
#             if not user_query:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail="Empty question provided"
#                 )
#         except ValueError as e:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid JSON format"
#             )

#         # Process query with error handling
#         try:
#             response = ask_chatbot(user_query)
#             if not isinstance(response, dict):
#                 raise ValueError("Chatbot returned invalid response format")
            
#             return response
#         except Exception as e:
#             print(f"Chatbot error: {str(e)}\n{traceback.format_exc()}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Chatbot processing failed: {str(e)}"
#             )

#     except HTTPException as http_err:
#         # Re-raise FastAPI HTTP exceptions
#         raise http_err
#     except Exception as unexpected_err:
#         print(f"Unexpected error: {unexpected_err}\n{traceback.format_exc()}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An unexpected error occurred"
#         )
@app.post("/chat", status_code=200)
async def ask_chatbot_v3_endpoint(
     payload: ChatPayload,
     current: dict = Depends(get_current_user)
 ):
     """
     Expects JSON:
       { page_context: str, query: str }
     """
     user_id = current["user_id"]
     return ask_chatbot_v3(user_id, payload.page_context, payload.query)


#-------------Vanna chatbot---------------------
@app.post("/vanna-chat")
async def handle_vanna_question(
    payload: dict,
    current: dict = Depends(get_current_user)
):
    user_id = current["user_id"]
    question = payload.get("question", "").strip()
    user_question = question
    followup_answer = payload.get("followup_answer", "").strip()
 
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty question provided"
        )
 
    # Step 0: Fetch existing chat history
    try:
        existing_doc = client.get(index="pg-vanna_chat", id=user_id, ignore=[404])
        chat_history = existing_doc.get("_source", {}).get("chat_history", [])
    except Exception:
        chat_history = []
 
    # Build chat context for prompts
    chat_context = "\n".join(
        f"Q: {entry['question']}\nA: {entry['answer']}" for entry in chat_history
    )
 
    # Step 1: Ask judgment model
    judgment_prompt = f"""
    You are a technical SQL intent analyzer for a parts database. Use chat history to understand context.
Analyze if the question contains enough technical specifics to generate SQL directly.
You have to returen "YES" or ask a follow-up question.
 
**Previous Chat Context:**
---
{chat_context}
---
 
Return "YES" if ANY of these are true only when complete code:
 
1. **Replacements**:
   - Contains "replacement for" + [item_code/part_name]
   - Asks "alternatives to X" with code/name
 
2. **Savings Analysis**:
   - Combines [part_name/item_code] + "most savings in [category]"
   - "Price difference analysis for X in Y category"
 
3. **Category Savings**:
   - "Total savings for [category]"
   - "Which category has highest price differences?"
 
4. **Optimal Replacements**:
   - "Replacement with highest savings for X"
   - "Best alternative part for [original_part]"
 
5. **Alternative Options**:
   - "Parts with most alternatives"
   - "Which item has highest replacement options?"
 
6. **Stock Queries**:
   - "Stock count of [item_code/part_name]"
   - "Availability for X replacement part"
 
**Return FOLLOW-UP question (where FOLLOW-UP means the next clarifying question) ONLY if:**
- Pure greeting (respond: "How may I help you?")
- Generic request without technical parameters (e.g. "Show me parts" → "Which specific part category?")
- Ambiguous codes without context (e.g. "AL30" alone → "Is AL30 an original part needing replacements?")
- replacement for AL30 (e.g. "AL30" → "which part from the AL30 series?")
 
**Examples of YES:**
User: "Stock availability for Hydraulic Pump B"
User: "Which alternator has most savings in Energy category?"
 
**Examples of FOLLOW-UP question:**
User: "Show replacements" → "Which specific part needs replacement?"
User: "Stock count" → "Could you specify the part name/code?"
User: "AL30" → "There are many parts in series, please specify and what do you want to know about it, stock, price or replacements?"
 
 
    User question: "{question}"
    """
 
    judgment_response = vn.submit_prompt(prompt=judgment_prompt).strip()
 
    followup_answer = judgment_response
 
    if judgment_response.lower() == "yes":
        
        refined_question = question
 
    elif followup_answer:
        # Step 2: Rewrite question using follow-up answer
        refine_prompt = f"""
        Original question: "{question}"
 
        Based on the following follow-up answer, rewrite the question so that it can be answered with a SQL query.
 
        Follow-up: {judgment_response}
        Answer: {followup_answer}
 
        Return ONLY the rewritten question in plain text.
        """
        refined_question = vn.submit_prompt(prompt=refine_prompt).strip()
 
        # Save follow-up Q&A to chat history
        chat_history.append({
            "question": judgment_response,
            "answer": followup_answer,
            "timestamp": datetime.now().isoformat()
        })
 
        return {
        "answer": judgment_response
        }
 
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"follow_up": judgment_response}
        )
 
    # Step 3: Generate SQL using refined question
    question_sql_list = vn.get_similar_question_sql(refined_question)
    ddl_list = vn.get_related_ddl(refined_question)
    doc_list = vn.get_related_documentation(refined_question)
 
    initial_prompt = vn.config.get("initial_prompt", None) if vn.config else None
 
    original_prompt = vn.get_sql_prompt(
        initial_prompt=initial_prompt,
        question=refined_question,
        question_sql_list=question_sql_list,
        ddl_list=ddl_list,
        doc_list=doc_list,
    )
 
    custom_prompt = f"""
    You are a highly skilled SQL assistant.
 
    Below is the chat history to help understand user intent:
    ---
    {chat_context}
    ---
 
    Your task is to write the most efficient and correct SQL query possible based on the user's request.
 
    User question: {user_question}
    Refined question: {refined_question}
 
    You may refer to the schema and context below:
    ---
    {original_prompt}
 
    Note:
    1. If the user query contains alphanumeric word search it with both part/item name or part/item code because it can be any of both
    2. If the user asks for part or part name then the output query should include the part name.
    3. If searching for any string (e.g., part name) then use: WHERE string_search_field LIKE '%value%'
    """
 
    final_sql = vn.submit_prompt(prompt=custom_prompt)
    extracted_sql = vn.extract_sql(final_sql)
 
    df = vn.run_sql(extracted_sql)
    summary = vn.generate_summary(refined_question, df)
    answer = format_output_as_markdown_with_gemini(df, summary)
 
    # Step 4: Store new Q&A in chat history
    new_entry = {
        "question": user_question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }
    chat_history.append(new_entry)
 
    # Step 5: Update Elasticsearch
    client.update(
        index="pg-vanna_chat",
        id=user_id,
        body={
            "doc": {
                "user_id": user_id,
                "chat_history": chat_history
            },
            "doc_as_upsert": True
        }
    )
 
    return {
        "original_question": user_question,
        "refined_question": refined_question,
        "answer": answer
    }
 