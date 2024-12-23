import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the generative AI model
genai.configure(api_key=api_key)

# Load FAQs from JSON file
with open('faqs.json') as f:
    faqs = json.load(f)['faqs']

# Initialize vector store with FAQ questions
def create_vector_store(faqs):
    questions = [faq["question"] for faq in faqs]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(questions, embedding=embeddings)
    vector_store.save_local("faq_index")

# Create the vector store for FAQs
create_vector_store(faqs)

# Set up a conversational chain for fallback answers
def get_conversational_chain():
    prompt_template = """
    You are a helpful customer support chatbot. Answer the customer's question using the context provided.
    If the answer isn't available in the context, say "I'm sorry, I don't have the answer to that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to find the best FAQ answer or use a fallback response
def get_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faq_index", embeddings, allow_dangerous_deserialization=True)

    # Perform semantic search on the FAQ
    docs = vector_store.similarity_search(user_question, k=1)
    
    # If an FAQ match is found, retrieve the answer
    if docs:
        matching_question = docs[0].page_content
        for faq in faqs:
            if faq["question"] == matching_question:
                return faq["answer"]

    # Fallback to generative response if no relevant FAQ answer is found
    fallback_context = "Unfortunately, we do not have information on that in the FAQ."
    chain = get_conversational_chain()
    response = chain({"context": fallback_context, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main function to run Streamlit app
def main():
    st.title("Customer Support Chatbot 🤖")
    st.write("Ask your question, and I'll do my best to help!")

    user_question = st.text_input("Enter your question here:")
    
    if st.button("Submit"):
        if user_question:
            answer = get_answer(user_question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
