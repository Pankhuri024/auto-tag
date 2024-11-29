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


# Function to find keywords in the summary text
def check_keywords(text, keyword_list):
    selected_keywords = []
    text_lower = text.lower()
    
    # Stem the entire text into a list of stemmed words for single-word matching
    text_words = [stemmer.stem(word) for word in text_lower.split()]
    
    for keyword in keyword_list:
        keyword_lower = keyword.lower()
        
        keyword_words = [word for word in keyword_lower.split() if word not in stop_words]
        
        # Stem the filtered keyword words
        keyword_stems = [stemmer.stem(word) for word in keyword_words]
        
        # Check if the entire keyword phrase (multi-word) exists as an exact match in the text
        if keyword_lower in text_lower:
            selected_keywords.append(keyword)
        # Match if ALL stemmed words in the multi-word keyword exist in the text (excluding stopwords)
        elif all(stem in text_words for stem in keyword_stems):
            selected_keywords.append(keyword)
        # Match if ANY single word from the keyword exists in the text (excluding stopwords)
        elif any(stem in text_words for stem in keyword_stems):
            selected_keywords.append(keyword)
    
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
        selected_categories = check_keywords(summary, categories)
        selected_elements = check_keywords(summary, elements)
        selected_tools = check_keywords(summary, tools)
        selected_goals = check_keywords(summary, goals)
        selected_research_types = check_keywords(summary, research_types)
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
