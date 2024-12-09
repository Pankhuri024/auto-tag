from flask import Flask, request, jsonify
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
app = Flask(__name__)

import nltk
# Initialize stemmer
stemmer = PorterStemmer()
nltk.download('stopwords')

# Get a list of English stopwords
stop_words = set(stopwords.words('english'))

# Synonym mapping for research types
RESEARCH_TYPE_SYNONYMS = {
    "A/B (Split Test)": [ "experiment","tests","test", "testing", "A/B", "AB test"],
    "User Study": [ "user research", "usability study", "usability research"],
    "Market Research": [ "customer research", "customer interview", "stakeholder interview"],
    "Lead generation": [ "drove leads" ],
    "Copy": ["messaging framework", "copy", "messaging"],
    "Ecommerce": ["orders per visitor", "orders", "online orders", "average order value", "revenue per visitor", "online purchases", "purchases","discounted"],
    "Data Analysis": ["gradually increasing", "gradually decreasing", "order trend", "purchase trend", "purchasing trend"]

    # Add more research types and their synonyms here
}

# Function to normalize text
def clean_text(text):
    """Normalize text by removing special characters and extra spaces."""
    # Replace special characters with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to find keywords in the summary text
def check_keywords(text, keyword_list, synonyms=None):
    selected_keywords = []
    text_clean = clean_text(text.lower())  # Clean and lowercase the text
    
    # Stem the cleaned text into a list of stemmed words
    text_words = [stemmer.stem(word) for word in text_clean.split()]
    # List of words that should not be matched by their stemmed version (e.g., "research")
    excluded_words = {"research", "researching", "researched", "searched"}  # Add any other unwanted words here
    
    for keyword in keyword_list:
        keyword_clean = clean_text(keyword.lower())  # Clean and lowercase the keyword
        
        # Split and stem the cleaned keyword words
        keyword_words = [word for word in keyword_clean.split() if word not in stop_words]
        keyword_stems = [stemmer.stem(word) for word in keyword_words]
        
        # Check if the entire cleaned keyword phrase exists in the text
        if keyword_clean in text_clean:
            selected_keywords.append(keyword)
        # Match if ALL stemmed words in the multi-word keyword exist in the text (excluding stopwords)
        elif all(stem in text_words for stem in keyword_stems) and not any(word in excluded_words for word in keyword_stems):
            selected_keywords.append(keyword)
            # not for user
        # # Match if ANY single word from the keyword exists in the text (excluding stopwords)
        # elif any(stem in text_words for stem in keyword_stems):
        #     selected_keywords.append(keyword)
        
        # Check for synonyms if provided
        # Check for synonyms if provided
        if synonyms and keyword in synonyms:
            synonym_list = synonyms[keyword]
            
            # Loop through each synonym
            for synonym in synonym_list:
                synonym_clean = clean_text(synonym.lower())
                
                # Check if synonym exists as a whole phrase
                if synonym_clean in text_clean:
                    selected_keywords.append(keyword)
                    break  # Avoid duplicate additions
                
                # Stem the synonym for partial matching
                synonym_words = [stemmer.stem(word) for word in synonym_clean.split() if word not in stop_words]
                if all(stem in text_words for stem in synonym_words) and not any(word in excluded_words for word in synonym_words):
                    selected_keywords.append(keyword)
                    break

    
    return selected_keywords


@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        summary = data.get('summary', '')  # Default to empty string if not provided

        # Use the predefined values if they are not passed in the request
        goals = data.get('goals', [])
        categories = data.get('categories', [])
        tools = data.get('tools', [])
        elements = data.get('elements', [])
        research_types = data.get('research_types', [])
        industries = data.get('industries', [])

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        # Check for keywords in the provided summary
        selected_categories = check_keywords(summary, categories, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_elements = check_keywords(summary, elements, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_tools = check_keywords(summary, tools)
        selected_goals = check_keywords(summary, goals, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_research_types = check_keywords(summary, research_types, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_industries = check_keywords(summary, industries)

        # Return the auto-selected values
        return jsonify({
            "selected_categories": selected_categories,
            "selected_elements": selected_elements,
            "selected_tools": selected_tools,
            "selected_goals": selected_goals,
            "selected_research_types": selected_research_types,
            "selected_industries": selected_industries,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
