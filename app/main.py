import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import requests
import hmac
import hashlib
import os
import json
from datetime import datetime
from typing import Any, List, Optional, Type
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SimpleSearchInput(BaseModel):
    event: str = Field(..., description="Event you want a gift for")
    person: str = Field(..., description="How the person is related to you") 
    age: int = Field(..., description="How old the person is")
    hobbies: List[str] = Field(..., description="What the person likes")
    priceRange: str = Field(..., description="What the person is willing to pay")

class ScrapeTool(BaseTool):
    name: str = "paapi_search"  # Add explicit annotation for the `name` field
    description: str = "Use to find search index of products from PA-API"  # Explicitly annotate `description`
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, **kwargs) -> str:
        """Use the tool."""
        return ['All', 'AmazonVideo', 'Apparel', 'Appliances', 'ArtsAndCrafts', 'Automotive', 'Baby', 'Beauty', 'Books', 'Classical', 'Collectibles', 'Computers', 'DigitalMusic', 'DigitalEducationalResources', 'Electronics', 'EverythingElse', 'Fashion', 'FashionBaby', 'FashionBoys', 'FashionGirls', 'FashionMen', 'FashionWomen', 'GardenAndOutdoor', 'GiftCards', 'GroceryAndGourmetFood', 'Handmade', 'HealthPersonalCare', 'HomeAndKitchen', 'Industrial', 'Jewelry', 'KindleStore', 'LocalServices', 'Luggage', 'LuxuryBeauty', 'Magazines', 'MobileAndAccessories', 'MobileApps', 'MoviesAndTV', 'Music', 'MusicalInstruments', 'OfficeProducts', 'PetSupplies', 'Photo', 'Shoes', 'Software', 'SportsAndOutdoors', 'ToolsAndHomeImprovement', 'ToysAndGames', 'VHS', 'VideoGames', 'Watches']
# Helper Functions for AWS Signature Version 4
def extract_search_indexes(response):
    # Assuming the response["output"] contains the structured JSON as a string
    pattern = r"\{\s*(?:\"[^\"]+\"\s*:\s*\"[^\"]*\"\s*,?\s*)+\}"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        extracted_json = match.group()
        print("Extracted JSON:")
        print(extracted_json)
        return json.loads(extracted_json)
    else:
        print("No JSON found.")
        return None
def generate_keywords_with_langchain(request_payload):
    tools = [
    ScrapeTool(),
    ]

# Initialize a ChatOpenAI model
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

    # Pull the prompt template from the hub
    # Pull the prompt template from the hub
    messages = [ 
        ("system","""You are an expert in generating relevant keywords for product searches on Amazon. You have access to the following tools:

{tools}

Use the following format:
**Format:**
Question: The input question you must answer
Thought: What Search Index fits best, given the context
Action: Validate and ensure the selected Search Index matches the allowed values, you can use these tools [{tool_names}]
Action Input: The input to the action
Observation: Result of the action
Thought: Create the Search Index and keyword pairs
Final Answer: The key-value pairs of valid Search Indexes and keywords

Important Note: The following search indexes should not be used in your response: {excluded_indexes}
"""),

        ("human","""
Begin!

Event: {event}
Person: {person}
Age: {age}
Gift Style: {gift_style}
Hobbies: {hobbies}
Price Range: {price_range}
User Keywords: {keywords}

Your task is to create three different key-value pairs of Search Indexes and one keyword for each search index, each search index should be from the list given by this tools [{tool_names}]. Ensure that none of the search indexes match any of the excluded ones (from previous suggestion).
NOTE: If you not able to find a search index from [{tool_names}], you can use "All".

Output Format:
{{
[search_index_1]: [keyword_1],
[search_index_2]: [keyword_2],
[search_index_3]: [keyword_3]
}}
{agent_scratchpad}
""")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # Initialize the ReAct agent using the create_tool_calling_agent function
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Test the agent with sample queries
    response = agent_executor.invoke({
        "keywords": request_payload.keywords if request_payload.keywords else "",
       "excluded_indexes": (", ").join(request_payload.suggestion) if request_payload.suggestion else "",
       "event":request_payload.event,
       "person":request_payload.person,
       "age":request_payload.age,
       "gift_category":request_payload.giftCategory,
       "gift_style":request_payload.giftStyle,
       "hobbies": (", ").join(request_payload.hobbies) if request_payload.hobbies else "",
       "price_range":request_payload.priceRange,
       "gift_style": "",
       "tools": tools,
       "tool_names": [tool.name for tool in tools],
       "agent_scratchpad": ""  # Provide an empty initial value for agent_scratchpad
    })
    #   response = llm.invoke(prompt)
    return response["output"]  # Split the response into a list of keywords

# ... rest of your code ...
def sign(key, message):
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()

def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, "aws4_request")
    return k_signing

# Request payload schema
class SearchRequest(BaseModel):
    person: str
    event: str
    keywords: Optional[str] = None
    suggestion: Optional[Any] = None
    age: Optional[str] = None
    country: Optional[str] = None
    giftCategory: Optional[str] = None
    giftStyle: Optional[str] = None
    hobbies: Optional[List[str]] = []
    priceRange: Optional[str] = None
def create_payload(search_index,keyword):
    return {
        "Marketplace": "www.amazon.com",
        "PartnerType": "Associates",
        "PartnerTag": "yourgiftwish-20",
        "Keywords": keyword,  # Assuming final_response is already defined
        "SearchIndex": search_index,  # Set the correct search index for each call
        "ItemCount": 1,
        "ItemPage": 1,
        "Resources": [
            "Images.Primary.Large",
            "ItemInfo.Title",
            "ItemInfo.Features",
            "ItemInfo.ContentInfo",
        ],
        "Merchant": "All",
        "DeliveryFlags": ["FreeShipping"],
        "Condition": "New",
        "MinPrice": 500,
        "MaxPrice": 6000,
    }

async def make_amazon_api_request(search_index,keyword):
    # Define the constants as before
    access_key_id = os.getenv("ACCESS_KEY_ID")
    secret_access_key = os.getenv("SECRET_ACCESSKEY")
    region = os.getenv("REGION")
    service = os.getenv("SERVICE")
    host = os.getenv("HOST")
    endpoint = f"https://{host}/paapi5/searchitems"

    current_date = datetime.utcnow()
    amz_date = current_date.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = current_date.strftime("%Y%m%d")

    payload = create_payload(search_index,keyword)

    canonical_request = "\n".join([
        "POST",
        "/paapi5/searchitems",
        "",
        f"content-encoding:amz-1.0\nhost:{host}\nx-amz-date:{amz_date}\nx-amz-target:com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems\n",
        "content-encoding;host;x-amz-date;x-amz-target",
        hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest(),
    ])

    # Create String to Sign
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        algorithm,
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    # Calculate Signature
    signing_key = get_signature_key(secret_access_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    # Create Authorization Header
    authorization_header = (
        f"{algorithm} Credential={access_key_id}/{credential_scope}, "
        f"SignedHeaders=content-encoding;host;x-amz-date;x-amz-target, "
        f"Signature={signature}"
    )

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Host": host,
        "X-Amz-Date": amz_date,
        "X-Amz-Target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.SearchItems",
        "Content-Encoding": "amz-1.0",
        "User-Agent": "paapi-docs-curl/1.0.0",
        "Authorization": authorization_header,
    }

    # Make the API Request
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code == 400:
            return await make_amazon_api_request("All", keyword)
        if response.status_code == 429 or response.status_code == 401:
            error = response.json()
            raise HTTPException(status_code=response.status_code, detail=error["Errors"][0]["Message"])
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {search_index}: {e}")
        return await make_amazon_api_request("All", keyword)
    
  
@app.post("/call-external-api")
async def search_items(request_payload: SearchRequest):
    response = generate_keywords_with_langchain(request_payload)
    response = extract_search_indexes(response)
    print(response)
    api_response = {"Items": []}
    if response is not None:
        for search_index,keyword in response.items():
            print(search_index,keyword)
            result = await make_amazon_api_request(search_index,keyword)
            if result:
                print(result)
                api_response["Items"].append(result['SearchResult']['Items'][0])
    print(api_response)
    try:
        def create_custom_response(api_response):
            items = api_response.get("Items", [])
            custom_items = [
                {
                    "redirectURL": item.get("DetailPageURL"),
                    "imageUrl": item.get("Images", {}).get("Primary", {}).get("Large", {}).get("URL", "No image available"),
                    "title": item.get("ItemInfo", {}).get("Title", {}).get("DisplayValue", "No title available"),
                    "description": item.get("ItemInfo", {}).get("Features", {}).get("DisplayValues", [""])[0],
                    "tag": "Technology",
                    "color": "#0072C6",
                }
                for item in items
            ]

            suggestion_list = list(response.keys()) if response else []
            return {"suggestion": suggestion_list, "items": custom_items}

        return create_custom_response(api_response)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))