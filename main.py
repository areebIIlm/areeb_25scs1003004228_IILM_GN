import json
import re
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# if not downloaded once run download automatically
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class FAQChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # FAQ Database can be set accordingly according to university
        self.faq_data = {
            "admission_requirements": {
                "questions": [
                    "what are admission requirements",
                    "how to apply for admission",
                    "admission process",
                    "what documents needed for admission",
                    "eligibility criteria"
                ],
                "answer": "The admission requirements for  IILM UNIVERSITY greater noida in BTECH you should have 75%le in jee mains and atleat 70% in class 12th. Direct admission are also available through 12th marks"
            },
            "application_deadline": {
                "questions": [
                    "when is application deadline",
                    "last date to apply",
                    "admission deadline",
                    "application dates"
                ],
                "answer": "The applications are available from february and last date to apply is 15th august"
            },
            "tuition_fees": {
                "questions": [
                    "how much is tuition fees",
                    "fees structure",
                    "cost of education",
                    "tuition fees per semester",
                    "how much does college cost"
                ],
                "answer": "Tuition fees vary from courses to courses semester vise fees for btech varises from 75,000 to 1,00,000 depending on the branch.For MBA semester wise fees is around 1,50,000. .For BcA it is 65,000. For MCA it is 85,000  "
            },
            "scholarships": {
                "questions": [
                    "are scholarships available",
                    "is there any scholarship "
                    "how to get scholarship",
                    "scholarship opportunities",
                    "funding options"
                ],
                "answer": "Yes! Scholarships are offered according to the Merit in various competitive examinations and class 12th marks "
            },
            "courses_offered": {
                "questions": [
                    "what courses are offered",
                    "available programs",
                    "courses offered",
                    "what can i study",
                    "degree programs"
                ],
                "answer": "IILM UNIVERSITY typically offer programs in: Engineering, Business, Arts & Sciences, Law, Computer Science, Social Sciences, Humanities, and more. Both bachelors and masters degree programs are available"
            },
            "campus_facilities": {
                "questions": [
                    "what facilities are available",
                    "how is campus facelities",
                    "library gym hostel",
                    "student facilities",
                    "campus infrastructure"
                ],
                "answer": "IILM UNIVERSITY campus offers various facilities like on campus hostel, basketball court, tennis court, badmintion court, football ground, cricet ground etc "
            },
            "housing_accommodation": {
                "questions": [
                    "is housing available",
                    "hostel facilities",
                    "on campus accommodation",
                    "where to stay",
                    "dormitory options"
                ],
                "answer": "IILM UNIVERSITY greater noida provide many hostel options within the campus and off campus hostel are also available hostel provide laundry servises, gym and mess and many more"
            },

            "campus_life": {
                "questions": [
                    "what is campus life like",
                    "student activities",
                    "clubs and organizations",
                    "extracurricular activities",
                    "social life"
                ],
                "answer": "IILM UNIVERSITY greater noida provide a vibrant Campus life which includes: student clubs, sports teams, cultural fests, student government, festivals, and networking events."
            },
            "career_services": {
                "questions": [
                    "career support",
                    "job help",
                    "internship opportunities",
                    "career counseling",
                    "job assistance"
                ],
                "answer": "IILM UNIVERSITY greater noida offer many career services including: resume/CV building, interview preparation, job fairs, internship placement, alumni networking, career counseling, and on-campus recruitment."
            },
            "Placements": {
                "questions": [
                    "what is average package in lpa",
                    "what is highest package",
                    "what is the median package",
                    "what are the placement oppurtinities"
                    
                ],
                "answer": "The average placement of IILM UNIVERSITY greater noida is 8LPA . Median is around 4.5LPA and highest placement is around 28LPA"
            },
            
            "grade_system": {
                "questions": [
                    "what is the grading system",
                    "gpa calculation",
                    "how are grades given",
                    "grade point average",
                    "marking scheme"
                ],
                "answer": "Grades or CGPA is calculated according to marks scored in theoretical exams , practical assesment and project based evaluation. MInimum of 7 CGPA is required to sit in placements and minimum of 4 CGPA is required to advance to new semester "
            },
            "student_support": {
                "questions": [
                    "academic support services",
                    "tutoring available",
                    "counseling services",
                    "student help",
                    "support services"
                ],
                "answer": "IILM UNIVERSITY greater noida provide various accademic and personal servises like every student is provided by carrer councellor and there are consellors for personal help etc "
            },
            
        }
    
    def preprocess_text(self, text: str):
        # preprocessing - lowercase,remving all comma etc, tokenising(word by word)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words]
        return tokens
    
    def calculate_similarity(self, query: str, faq_question: str) :
        
        query_tokens = set(self.preprocess_text(query))
        faq_tokens = set(self.preprocess_text(faq_question))
        
        if not query_tokens or not faq_tokens:
            return 0.0
        
        # findind similarity in order to give relevant answer
        similarity = query_tokens.intersection(faq_tokens)
        union = query_tokens.union(faq_tokens)
        jaccard = len(similarity) / len(union)
        
        #similarity in string
        string_similarity = SequenceMatcher(None, query.lower(), faq_question.lower()).ratio()
        
        # score help in relevency
        return 0.6 * jaccard + 0.4 * string_similarity
    
    def find_best_match(self, query: str): 
        # findind most relevant question
        best_match = None
        best_score = 0.0
        
        for faq_id, faq_content in self.faq_data.items():
            for question in faq_content["questions"]:
                score = self.calculate_similarity(query, question)
                if score > best_score:
                    best_score = score
                    best_match = faq_id
        
        return best_match, best_score
    
    def answer(self, query: str, threshold: float = 0.3):
        # fetching most relevant answer
        best_match, score = self.find_best_match(query)
        
        if score >= threshold and best_match:
            return self.faq_data[best_match]["answer"]
        else:
            return "I'm sorry, No relevant answer founds please elaborate your question or see all the freaquently asked topic by writing 'topics'"
    
    
    
    def list_topics(self):
        # List all the topics
        topics = []
        for faq_content in self.faq_data.values():
            topics.append(faq_content["questions"][0])
        return topics


def main():
    # main function 
    chatbot = FAQChatbot()
    
    print("=" * 55)
    print("\t\t     FAQ Chatbot")
    print("=" * 55)
    print("🤖 Ask me anything about the University\n")
    print("Type 'topics' to see all frequently asked questions")
    print("Type 'quit' or 'exit' to exit the chat")
    print("=" * 55)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n🤖: Thank you for chatting with us. Goodbye!😊 ")
            break
        
        if user_input.lower() == 'topics':
            print("\n🤖: Here are all the frequenly asked questions:")
            for i, topic in enumerate(chatbot.list_topics(), 1):
                print(f"  {i}. {topic.capitalize()}")
            continue
        
        response = chatbot.answer(user_input)
        print(f"\n🤖: {response}")


if __name__ == "__main__":
    main()
