import os
import json
import numpy as np
from typing import List
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from sentence_transformers import SentenceTransformer, util
import re

# Load your 10-15 filtered schemes
def load_schemes(filepath):
    with open(filepath, 'r') as file:
        raw = json.load(file)

    # Assuming structure is: [ { "response": [ <actual_schemes> ] } ]
    if isinstance(raw, list) and "response" in raw[0]:
        return raw[0]["response"]
    else:
        raise ValueError("Unexpected JSON structure. Expected list with a 'response' key.")


# Turn a scheme into a semantic description string
def get_scheme_semantic_text(scheme: dict) -> str:
    return f"""
    Scheme Name: {scheme['schemeName']}
    Category: {scheme.get('category', '')}
    Documents: {', '.join(scheme.get('documents', []))}
    Application Procedure: {'; '.join(scheme.get('applicationProcedure', []))}
    """

# Generate adaptive, high-discrimination questions
def generate_questions(schemes: List[dict], groq_api_key: str) -> List[str]:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    prompt = PromptTemplate(
        input_variables=["scheme_texts"],
        template="""
You are helping a user find the most suitable government scheme from a filtered list.

Given the following scheme descriptions:

{scheme_texts}

Generate 5 smart, diverse questions to ask the user.
Each question should help eliminate schemes based on required documents, user eligibility (like student, sportsperson, widow), or application context.
Respond with 5 questions only.
"""
    )

    scheme_texts = "\n\n".join([get_scheme_semantic_text(s) for s in schemes])
    chain = prompt | llm
    questions_output = chain.invoke({"scheme_texts": scheme_texts})
    # print("RAW LLM OUTPUT:\n", questions_output.content)
    question_lines = re.findall(r"\d+\.\s*(.*)", questions_output.content)
    return [q.strip() for q in question_lines if q.strip()]

# Embed schemes using sentence-transformers
def embed_schemes(schemes: List[dict]) -> List[np.ndarray]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [get_scheme_semantic_text(s) for s in schemes]
    return model.encode(texts, convert_to_tensor=True)

# Embed user answer and filter top schemes
def filter_by_similarity(user_answers: List[str], schemes: List[dict], scheme_embeddings: List[np.ndarray]) -> List[dict]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    joined = ". ".join(user_answers)
    answer_embedding = model.encode(joined, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(answer_embedding, scheme_embeddings)[0]
    top_indices = np.argsort(-similarities.cpu().numpy())[:3]  # top 4
    return [schemes[i] for i in top_indices]

# Main function
def main():
    file_path = "schemes.json"  # should have 10-15 schemes
    groq_api_key = os.getenv("GROQ_API_KEY_2") or input("Enter your GROQ API key: ")

    schemes = load_schemes(file_path)
    print(f"Loaded {len(schemes)} schemes for refinement.\n")

    # Step 1: Generate 5 smart questions
    questions = generate_questions(schemes, groq_api_key)
    print("Please answer the following questions to help us filter the schemes further:\n")
    # Display all questions first
    print("\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]))
    print("\nPlease provide your answers below. Separate each answer with a new line:\n")

    user_answers = []
    for i in range(len(questions)):
        ans = input(f"Answer {i+1}: ")
        user_answers.append(ans)

    # Step 2: Embed all schemes and filter
    scheme_embeddings = embed_schemes(schemes)
    final_schemes = filter_by_similarity(user_answers, schemes, scheme_embeddings)

    print("\n\n✅ Final Matched Schemes:")
    print(json.dumps(final_schemes, indent=2))

if __name__ == "__main__":
    main()
