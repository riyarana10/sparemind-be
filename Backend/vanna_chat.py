from vanna.vannadb import VannaDB_VectorStore
from vanna.google import GoogleGeminiChat
from dotenv import load_dotenv
import os

load_dotenv()
# host=os.getenv("DB_HOST")
# user=os.getenv("DB_USER")
# password=os.getenv("DB_PASSWORD")
# dbname=os.getenv("DB_NAME")
# port=3306
# vanna_api_key=os.getenv("VANNA_API_KEY")
# gemini_api=os.getenv("GOOGLE_API_KEY")

DB_HOST="bvr-staging.cynf49jrhf7w.us-west-2.rds.amazonaws.com"
DB_USER="root"
DB_PASSWORD="_2aGUuCz#U7_fK+K"
DB_NAME="staging_bvr"
 
VANNA_API_KEY=os.getenv("VANNA_API_KEY")

host=DB_HOST
user=DB_USER
password=DB_PASSWORD
dbname=DB_NAME
port=3306
vanna_api_key=VANNA_API_KEY
gemini_api=os.getenv("GOOGLE_API_KEY")



class MyVanna(VannaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        MY_VANNA_MODEL = 'hemant' 
        VannaDB_VectorStore.__init__(self, vanna_model=MY_VANNA_MODEL, vanna_api_key=vanna_api_key, config=config)
        GoogleGeminiChat.__init__(self, config={'api_key':gemini_api , 'model': 'gemini-2.0-flash'})

vn = MyVanna()


vn.connect_to_mysql(
    host=host,
    user=user,
    password=password,
    dbname=dbname,
    port=port,
)

#--------------training------------------

# # 1. Train with DDL (this defines your table structure)
# vn.train(ddl="""
# CREATE TABLE IF NOT EXISTS test_table (
#     ic_Item_Code VARCHAR(50),
#     ic_item_Description TEXT,
#     ic_Consume_Qty VARCHAR(20),
#     si_Item_Code VARCHAR(50),
#     si_Item_Description TEXT,
#     si_Item_Category VARCHAR(100),
#     si_Unit VARCHAR(10),
#     si_Location VARCHAR(100),
#     si_Stock VARCHAR(20),
#     si_Rate VARCHAR(20),
#     si_Amnt VARCHAR(20),
#     si_Mil VARCHAR(20),
#     si_Rol VARCHAR(20),
#     si_Remarks TEXT,
#     "si_Item Type" VARCHAR(50),
#     in_stock_inventory VARCHAR(50),
#     in_work_order VARCHAR(50)
# )
# """)

# # 2. Train with SQL Queries (your training examples)
# queries = [
#     # Q1: Top 5 items with highest consumption
#     "SELECT si_Item_Code, ic_item_Description, CAST(REPLACE(ic_Consume_Qty, ',', '') AS UNSIGNED) AS consumed_qty FROM test_table ORDER BY consumed_qty DESC LIMIT 5",

#     # Q2: Total stock value (amount)
#     "SELECT SUM(CAST(REPLACE(si_Amnt, ',', '') AS DECIMAL(15,2))) AS total_stock_value FROM test_table",

#     # Q3: All cable tie items with stock > 10000
#     "SELECT si_Item_Code, si_Item_Description, CAST(REPLACE(si_Stock, ',', '') AS UNSIGNED) AS stock FROM test_table WHERE si_Item_Category = 'CABLE TIE' AND CAST(REPLACE(si_Stock, ',', '') AS UNSIGNED) > 10000",

#     # Q4: Items with minimum reorder level
#     "SELECT si_Item_Code, si_Item_Description, CAST(REPLACE(si_Rol, ',', '') AS UNSIGNED) AS reorder_level FROM test_table ORDER BY reorder_level ASC LIMIT 5",

#     # Q5: Items in EMTS1CF002E003 location
#     "SELECT * FROM test_table WHERE si_Location = 'EMTS1CF002E003'",

#     # Q6: Count of unique categories
#     "SELECT COUNT(DISTINCT si_Item_Category) AS unique_categories FROM test_table",

#     # Q7: Stock value by category
#     "SELECT si_Item_Category, SUM(CAST(REPLACE(si_Amnt, ',', '') AS DECIMAL(15,2))) AS total_value FROM test_table GROUP BY si_Item_Category",

#     # Q8: All consumable items
#     "SELECT * FROM test_table WHERE \"si_Item Type\" = 'Consumables'",

#     # Q9: Items used in work orders but not in stock
#     "SELECT si_Item_Code, si_Item_Description FROM test_table WHERE in_work_order = 'present' AND in_stock_inventory != 'present'",

#     # Q10: All items with 'SHELL' in description
#     "SELECT * FROM test_table WHERE si_Item_Description LIKE '%SHELL%'",

#     # Q11: All items with stock less than reorder level
#     "SELECT si_Item_Code, si_Item_Description FROM test_table WHERE CAST(REPLACE(si_Stock, ',', '') AS UNSIGNED) < CAST(REPLACE(si_Rol, ',', '') AS UNSIGNED)",

#     # Q12: Average rate of all items
#     "SELECT AVG(CAST(REPLACE(si_Rate, ',', '') AS DECIMAL(10,2))) AS avg_rate FROM test_table"
# ]

# for query in queries:
#     vn.train(sql=query)