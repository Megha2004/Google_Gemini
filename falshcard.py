import google.generativeai as genai
import json

genai.configure(api_key="AIzaSyAdR_Bfb5h-cnIk2xxxJtGc8naiHGuNFL4")

pdf_content = """
Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.
In plants, photosynthesis typically occurs within the chloroplasts, using chlorophyll. This process is crucial for the survival of plant life on Earth.
"""

# Function to get Gemini AI response
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Function to extract key concepts using Gemini
def extract_key_concepts(text):
    prompt = f"Extract key concepts and terms from the following text for creating flashcards:\n\n{text}\n\nProvide terms and short definitions."
    response = genai.generate_text(model="models/text-bison-001", prompt=prompt)
    return response['text'] if 'text' in response else None

# Get key concepts from the text
key_concepts = extract_key_concepts(pdf_content)
print("Extracted Key Concepts:", key_concepts)

# Function to create flashcards using Gemini
def create_flashcards(key_concept):
    # Craft a prompt to generate a question and answer format for a flashcard
    prompt = f"Create a flashcard with a question and answer based on this concept:\n\n'{key_concept}'\n\nQuestion:"
    response = genai.generate_text(prompt)
    
    # Split response into question and answer if needed
    question = response['text'].split("Answer:")[0].strip()
    answer = response['text'].split("Answer:")[1].strip() if "Answer:" in response['text'] else ""
    return {"question": question, "answer": answer}

# Example of creating flashcards based on extracted key concepts
flashcards = []
for concept in key_concepts.split("\n"):
    flashcard = create_flashcards(concept)
    flashcards.append(flashcard)

print("Generated Flashcards:", json.dumps(flashcards, indent=2))

# Save flashcards to a JSON file
with open("flashcards.json", "w") as f:
    json.dump(flashcards, f, indent=2)
print("Flashcards saved to flashcards.json")