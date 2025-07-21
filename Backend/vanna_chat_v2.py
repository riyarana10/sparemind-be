from vanna.vannadb import VannaDB_VectorStore
from vanna.google import GoogleGeminiChat
from dotenv import load_dotenv
import os
import json
import pandas as pd
import sys
from google import genai
from google.genai import types

load_dotenv()

host=os.getenv("DB_HOST")
user=os.getenv("DB_USER")
password=os.getenv("DB_PASSWORD")
dbname=os.getenv("DB_NAME")
port=3306
vanna_api_key="7760350302f44402a1663f70a3c1a54e"
gemini_api=os.getenv("GOOGLE_API_KEY")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: set GOOGLE_API_KEY in your .env", file=sys.stderr)
    sys.exit(1)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
client = genai.Client(api_key=API_KEY)

class MyVanna(VannaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, host, user, password, dbname, port, config=None):
        MY_VANNA_MODEL = 'bvr'

        # Initialize parent classes
        VannaDB_VectorStore.__init__(self, vanna_model=MY_VANNA_MODEL, vanna_api_key=vanna_api_key, config=config)
        GoogleGeminiChat.__init__(self, config={'api_key': gemini_api, 'model': 'gemini-2.0-flash'})

        # Connect to MySQL
        self.connect_to_mysql(
            host=host,
            user=user,
            password=password,
            dbname=dbname,
            port=port
        )

vn = MyVanna(
    host=host,
    user=user,
    password=password,
    dbname=dbname,
    port=port
)

#------------Converstion to markdown format-----------------------

def format_output_as_markdown_with_gemini(df: pd.DataFrame, summary_text: str) -> str:
    # Convert dataframe to CSV for easier Gemini parsing
    df_csv = df.to_csv(index=False)

    prompt = f"""
You are a Markdown formatting assistant specialized in technical data presentation.

Convert the SQL results and summary into SIMPLE MARKDOWN WITHOUT TABLES using these rules:

1. **Data Presentation**
- Convert each row to bullet points with clear labels
- Format like:
  • **Item Code:** MA0BO008000  
  • **Description:** AL30-03B-11-X2111, AIR LUBRICATOR, SMC  
  • **Stock:** **2**  

2. **Number Highlighting**
- Make ALL numerical values bold using **
- Highlight critical values in summary using bold

3. **Structure**
- Start with "### Results"
- Follow with "### Key Findings" for summary
- Use horizontal rules (---) between sections

4. **Prohibited**
- No tables
- No complex nesting
- No code blocks

Example Output:
### Results
• **Item Code:** MA0BO008000  
• **Description:** AL30-03B-11-X2111, AIR LUBRICATOR, SMC  
• **Total Stock:** **2**  

---

### Key Findings
The total stock of AL30-03B-11-X2111 (AIR LUBRICATOR, SMC) is **2**.

Now format this data:

Query Output (CSV):
{df_csv}

Summary:
{summary_text}
"""


    # Send to Gemini for formatting
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return response.text.strip()


#-------------Training------------------

# def upload_documentation(txt_path: str):
#     try:
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#             vn.add_documentation(documentation=content)
#             print(f"[✓] Uploaded documentation from {txt_path}")
#     except Exception as e:
#         print(f"[✗] Error uploading documentation: {e}")


# def upload_ddl(txt_path: str):
#     try:
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             ddls = f.read()
#             vn.add_ddl(ddl=ddls)
#         print(f"[✓] Uploaded DDL statements from {txt_path}")
#     except Exception as e:
#         print(f"[✗] Error uploading DDLs: {e}")



# def upload_question_sql_pairs(json_path: str):
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             question_sql_pairs = json.load(f)

#         for pair in question_sql_pairs:
#             question = pair.get("question")
#             sql = pair.get("sql")
#             if question and sql:
#                 status_id = vn.add_question_sql(question, sql)
#                 print(f"[✓] Added Q-SQL pair with ID: {status_id}")
#             else:
#                 print(f"[!] Skipped invalid entry: {pair}")
#     except Exception as e:
#         print(f"[✗] Error uploading Q-SQL pairs: {e}")


# def safe_extract_json(response_text):
#     try:
#         start = response_text.find("{")
#         end = response_text.rfind("}") + 1
#         json_str = response_text[start:end]
#         return json.loads(json_str)
#     except Exception as e:
#         print("⚠️ Failed to extract JSON from response:")
#         print(response_text)
#         raise e


# # Step 1: Original user question
# question = "find replacement for CY3R32-350-Z73L"



# # judgment_json = safe_extract_json(judgment_result)


# # Step 3: Handle follow-ups if required
# judgment_prompt = f"""
# You are a language model assistant that determines whether a user's question can be answered using SQL alone.
# If the user's question is clear and specific (e.g., contains necessary context like full product names or filter conditions), return "YES".

# If the question is vague, too short, alphanumeric-only (e.g., just a code like "AL20" or "AL30"), or lacks key details to be translated into SQL,
# return a follow-up question that would help clarify it.

# For example:
# - If user input is just "AL20", ask: "What is the full product name or category for 'AL20'?"
# - If the question is ambiguous or incomplete, ask a clarifying question to make it suitable for a SQL query.

# No need to ask ques regarding table or schema 
# No need to ask what type of replacement it want
# User question: "{question}"
# """


# judgment_result = vn.submit_prompt(prompt=judgment_prompt)



# # Extract the result text and clean it
# judgment_response = judgment_result.strip().lower()

# if judgment_response == "yes":
#     refined_question = question
# else:
#     # Assume the response is a follow-up question
#     followups = [judgment_response]  # you can modify this if multiple follow-ups are returned
#     answers = {}

#     # Prompt user for follow-up answers
#     for q in followups:
#         ans = input(f"Follow-up needed: {q}\nYour answer: ")
#         answers[q] = ans

#     # Reconstruct the refined question
#     followup_context = "\n".join([f"{q} Answer: {a}" for q, a in answers.items()])

#     refine_prompt = f"""
#     Original question: "{question}"

#     Based on the following follow-up answers, rewrite the question so that it can be answered with a SQL query.

#     {followup_context}

#     Return ONLY the rewritten question in plain text.
#     """

#     refined_question = vn.submit_prompt(prompt=refine_prompt).strip()


# # Step 5: Continue with SQL generation as before
# question_sql_list = vn.get_similar_question_sql(refined_question)
# ddl_list = vn.get_related_ddl(refined_question)
# doc_list = vn.get_related_documentation(refined_question)

# initial_prompt = vn.config.get("initial_prompt", None) if vn.config else None

# original_prompt = vn.get_sql_prompt(
#     initial_prompt=initial_prompt,
#     question=refined_question,
#     question_sql_list=question_sql_list,
#     ddl_list=ddl_list,
#     doc_list=doc_list,
# )

# custom_prompt = f"""
# You are a highly skilled SQL assistant.

# Your task is to write the most efficient and correct SQL query possible based on the user's request.

# User question: {refined_question}

# You may refer to the schema and context below:
# ---
# {original_prompt}

# Note:
# 1. If the user query contains alphanumeric word search it with both part/item name or part/item code because it can be any of both
# 2. If the user asks for part or part name then the output query should include the part name.
# 3. If searching for any string (e.g., part name) then use: WHERE string_search_field LIKE '%value%'
# """

# # Step 6: Generate and execute SQL
# final_sql = vn.submit_prompt(prompt=custom_prompt)
# extracted_sql = vn.extract_sql(final_sql)

# df = vn.run_sql(extracted_sql)
# summary = vn.generate_summary(refined_question, df)

# # Step 7: Output result
# print("\nGenerated SQL Query:\n", extracted_sql)
# print("\nQuery Result:\n", df)
# print("\nSummary:\n", summary)
 