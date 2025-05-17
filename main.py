import os
import json
import dotenv
import re
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

# Load environment variables from .env file
dotenv.load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY_2")

# Setup Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# Define prompt for standardizing the scheme info
prompt_template = PromptTemplate(
    input_variables=["scheme"],
    template="""
Given the following government scheme details, extract a JSON in this format:

{{
  "schemeName": "...", 
  "eligibility": {{
    "residence": "...", `urban or rural only default should be rural`
    "minAge": 0,
    "maxAge": 0,
    "gender": false, `male means true , female false , null for rest`
    "casteCategory": "...", `OBC , SC , ST , etc`
    "minIncome": 0,
    "maxIncome": 0,
    "disability": false,
    "minority": false,
    "maritalStatus": "..."
  }},
  "category": "...", `means education, agriculture , sports , health and wellness , women and child , travel tourism , etc`
  "applicationProcedure": ["..."],
  "documentsRequired": ["..."],
  "state": "...",
  "mode": false, `if application offline true else false`
  "sourceLink": "..." `officical link`  
}}

Use only information provided. If something is not mentioned, assign a default like 0, false, or "Any".

Scheme Details:
{scheme}

Output only the JSON, no explanation.
"""
)

# Create chain with LLM
chain = prompt_template | llm

# Load all JSON files in the "Assam" folder
schemes = []
folder_path = "OutputJson"
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                schemes.extend(data)
            else:
                schemes.append(data)

# Output folder
output_folder = "new_structured_schemes"
os.makedirs(output_folder, exist_ok=True)

# Clean file names for saving
def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")[:100]

# Clean LLM output for JSON parsing
def clean_llm_output(raw_output):
    cleaned = re.sub(r"^```(json)?\n?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
    cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)  # Remove trailing commas
    return cleaned

# Process each scheme
for scheme in schemes:
    scheme_name = scheme.get("schemeName", "Unknown")
    try:
        # Prepare input and invoke LLM
        scheme_str = json.dumps(scheme, ensure_ascii=False, indent=2)
        result = chain.invoke({"scheme": scheme_str})

        # Clean LLM output
        raw_output = result.content.strip()
        cleaned_output = clean_llm_output(raw_output)

        # Parse to JSON
        structured = json.loads(cleaned_output)

        # Save output
        filename = clean_filename(structured["schemeName"]) + ".json"
        with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)

        print(f"[✓] Processed: {structured['schemeName']}")

    except Exception as e:
        print(f"[!] Failed to process: {scheme_name} → {e}")

    # Sleep to avoid rate limiting
    print("[⏳] Sleeping for 60 seconds...")
    time.sleep(25)
