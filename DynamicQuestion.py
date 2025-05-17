import os
import json
import numpy as np
from typing import List
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from sentence_transformers import SentenceTransformer, util
import re

def load_schemes(filepath):
    with open(filepath, 'r') as file:
        raw = json.load(file)

    if isinstance(raw, list) and "response" in raw[0]:
        return raw[0]["response"]
    else:
        raise ValueError("Unexpected JSON structure. Expected list with a 'response' key.")


def get_scheme_semantic_text(scheme: dict) -> str:
    return f"""
    Scheme Name: {scheme['schemeName']}
    Category: {scheme.get('category', '')}
    Eligibility: {scheme.get('eligibility', '')}
    Documents: {', '.join(scheme.get('documents', []))}
    Application Procedure: {'; '.join(scheme.get('applicationProcedure', []))}
    Mode: {scheme.get('mode', '')}
    """

def summarize_scheme(scheme: dict, groq_api_key: str) -> str:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    scheme_text = get_scheme_semantic_text(scheme)
    prompt = PromptTemplate(
        input_variables=["scheme_text"],
        template="""
You are a helpful assistant. Summarize the following government scheme into a concise, clear description highlighting key points:

{scheme_text}

Summary:
"""
    )
    chain = prompt | llm
    summary_output = chain.invoke({"scheme_text": scheme_text})
    return summary_output.content.strip()

def generate_questions(scheme_summaries: List[str], groq_api_key: str) -> List[str]:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    prompt = PromptTemplate(
        input_variables=["scheme_texts"],
        template="""
You are a government scheme advisor. Your job is to help match a user with one of the few schemes.

Below is a JSON array of scheme summaries:

{scheme_texts}

Generate 5 intelligent and diverse questions that will help you understand the user's background and needs.
Your goal is to narrow down the list by:
- Asking about the presence of specific documents such as Caste Certificate, Income Certificate, Disability Certificate, Farmer ID, Student ID, MSME Registration, etc. Avoid generic questions like “what other documents do you have?”
- User eligibility (student, widow, farmer, entrepreneur, etc.)

Avoid generic questions. Ask sharp, discriminative ones. Just return the questions in numbered list format.
"""
    )

    scheme_texts = json.dumps(scheme_summaries, indent=2)
    chain = prompt | llm
    questions_output = chain.invoke({"scheme_texts": scheme_texts})
    question_lines = re.findall(r"\d+\.\s*(.*)", questions_output.content)
    return [q.strip() for q in question_lines if q.strip()]

def embed_schemes(schemes: List[dict]) -> List[np.ndarray]:
    model = SentenceTransformer('all-mpnet-base-v2')
    texts = [get_scheme_semantic_text(s) for s in schemes]
    return model.encode(texts, convert_to_tensor=True)

def filter_by_similarity(user_answers: List[str], schemes: List[dict], scheme_embeddings: List[np.ndarray]) -> List[dict]:
    model = SentenceTransformer('all-mpnet-base-v2')
    joined = ". ".join(user_answers).lower()
    answer_embedding = model.encode(joined, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(answer_embedding, scheme_embeddings)[0].cpu().numpy()

    keywords = ["widow", "student", "farmer", "entrepreneur", "sports", "rural", "disability"]
    for idx, scheme in enumerate(schemes):
        for kw in keywords:
            if kw in joined and kw in json.dumps(scheme).lower():
                similarities[idx] += 0.05 

    top_indices = np.argsort(-similarities)[:3]
    return [schemes[i] for i in top_indices]

# Main function
def main():
    file_path = "schemes_2.json" 
    groq_api_key = os.getenv("GROQ_API_KEY_2") or input("Enter your GROQ API key: ")

    schemes = load_schemes(file_path)
    print(f"Loaded {len(schemes)} schemes for refinement.\n")

    summarized_schemes = [summarize_scheme(scheme, groq_api_key) for scheme in schemes]

    questions = generate_questions(summarized_schemes, groq_api_key)
    print("Please answer the following questions to help us filter the schemes further:\n")
    print("\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]))
    print("\nPlease provide your answers below. Separate each answer with a new line:\n")

    user_answers = []
    for i in range(len(questions)):
        ans = input(f"Answer {i+1}: ")
        user_answers.append(ans)

    scheme_embeddings = embed_schemes(schemes)
    final_schemes = filter_by_similarity(user_answers, schemes, scheme_embeddings)

    print("\n\n✅ Final Matched Schemes:")
    print(json.dumps(final_schemes, indent=2))

if __name__ == "__main__":
    main()
