import os
import re
import string
import math
from collections import Counter

# Standard FAQ dataset containing sample questions and accurate responses
FAQ_DATA = [
    {
        "question": "What is CodeAlpha?",
        "answer": "CodeAlpha is a software development and education platform providing virtual internships and hands-on project experience in emerging technologies like AI, Web Development, and Data Science."
    },
    {
        "question": "How do I submit my assigned internship tasks?",
        "answer": "You need to upload your source code to GitHub, share a project explanation video on LinkedIn tagging @CodeAlpha, and fill out the official task submission form provided in your WhatsApp group."
    },
    {
        "question": "How many tasks do I need to complete to get a certificate?",
        "answer": "You must complete at least 2 or 3 tasks out of the assigned domain task list to be eligible for the internship completion certificate."
    },
    {
        "question": "What perks do interns receive upon successful completion?",
        "answer": "Interns receive a Verified Completion Certificate, Unique ID Certificate, Offer Letter, and potentially a Letter of Recommendation (LOR) based on exceptional performance."
    },
    {
        "question": "What programming language is used for Artificial Intelligence tasks?",
        "answer": "Python is the primary language used for AI, Natural Language Processing, Computer Vision, and Deep Learning tasks in this internship."
    },
    {
        "question": "How can I contact CodeAlpha support for queries?",
        "answer": "You can reach out via email at services@codealpha.tech or contact their official WhatsApp helpline at +91 9336576683."
    }
]

def preprocess_text(text):
    """
    Cleans and tokenizes input text:
    1. Converts text to lowercase
    2. Removes punctuation and special characters
    3. Splits text into distinct word tokens
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    return tokens

def compute_tf(tokens):
    """Computes Term Frequency (TF) for a tokenized sequence."""
    tf_dict = {}
    total_tokens = len(tokens)
    if total_tokens == 0:
        return tf_dict
    counts = Counter(tokens)
    for word, count in counts.items():
        tf_dict[word] = count / float(total_tokens)
    return tf_dict

def compute_idf(doc_list):
    """Computes Inverse Document Frequency (IDF) across all documents."""
    idf_dict = {}
    N = len(doc_list)
    
    # Gather set of all unique words
    all_words = set(word for doc in doc_list for word in doc)
    
    for word in all_words:
        doc_count = sum(1 for doc in doc_list if word in doc)
        # Smooth IDF calculation to prevent division by zero
        idf_dict[word] = math.log((1 + N) / (1 + doc_count)) + 1
        
    return idf_dict

def vectorize(tokens, idf_dict, vocabulary):
    """Converts tokenized text into a TF-IDF vector."""
    tf = compute_tf(tokens)
    vector = []
    for word in vocabulary:
        vector.append(tf.get(word, 0.0) * idf_dict.get(word, 0.0))
    return vector

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two numerical vectors."""
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vec1))
    magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

class FAQChatbot:
    def __init__(self, faq_dataset):
        self.faq_dataset = faq_dataset
        self.processed_questions = [preprocess_text(item["question"]) for item in self.faq_dataset]
        self.vocabulary = list(set(word for doc in self.processed_questions for word in doc))
        self.idf_dict = compute_idf(self.processed_questions)
        
        # Precompute vectors for stored FAQ questions
        self.question_vectors = [
            vectorize(doc, self.idf_dict, self.vocabulary)
            for doc in self.processed_questions
        ]

    def get_response(self, user_query, similarity_threshold=0.2):
        """Processes user input query and returns the best matching response."""
        processed_query = preprocess_text(user_query)
        query_vector = vectorize(processed_query, self.idf_dict, self.vocabulary)
        
        best_score = 0.0
        best_index = -1
        
        for idx, q_vector in enumerate(self.question_vectors):
            score = cosine_similarity(query_vector, q_vector)
            if score > best_score:
                best_score = score
                best_index = idx
                
        if best_score >= similarity_threshold and best_index != -1:
            return self.faq_dataset[best_index]["answer"], best_score
        else:
            return "I'm sorry, I couldn't find an exact answer to your question. Please contact support at services@codealpha.tech.", best_score

if __name__ == "__main__":
    print("=" * 60)
    print("      CODEALPHA AI INTERNSHIP - TASK 2: FAQ CHATBOT      ")
    print("=" * 60)
    print("Initializing Chatbot Model...")
    bot = FAQChatbot(FAQ_DATA)
    print("Chatbot is ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                print("Bot: Thank you! Have a great day ahead.")
                break
            if not user_input.strip():
                continue
                
            response, confidence = bot.get_response(user_input)
            print(f"Bot: {response}")
            print(f"[Match Confidence Score: {confidence:.2f}]\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break
