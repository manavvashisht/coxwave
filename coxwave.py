import csv
import torch
from transformers import RagRetriever, RagTokenForGeneration, RagTokenizer

def load_faq_dataset(file_path):
    faq_data = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question = row[0].strip()
            answer = row[1].strip()
            faq_data[question] = answer
    return faq_data

def initialize_rag_model():
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
    generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    return retriever, generator, tokenizer

def retrieve_answer(question, faq_data, retriever, generator, tokenizer, conversation_history=None):
    if conversation_history:
        context = " ".join(conversation_history)
        inputs = tokenizer([question], return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True)
        with torch.no_grad():
            retrieved_docs = retriever(**inputs)
            generated = generator.generate(
                input_ids=retrieved_docs["input_ids"],
                num_return_sequences=1,
                max_length=100,
                early_stopping=True,
                context=context
            )
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    # If no context available, simply retrieve answer
    if question in faq_data:
        return faq_data[question]
    else:
        return "I'm sorry, I couldn't find an answer to your question."

def main():
    faq_data = load_faq_dataset("naver_store_qna_en.csv")
    retriever, generator, tokenizer = initialize_rag_model()
    conversation_history = []

    print("Welcome to Smart Store FAQ Chatbot. Ask me anything about Smart Store!")
    while True:
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Retrieve answer based on user input
        answer = retrieve_answer(user_input, faq_data, retriever, generator, tokenizer, conversation_history)
        print("Bot:", answer)

        # Save conversation history
        conversation_history.append(user_input)

if __name__ == "__main__":
    main()
