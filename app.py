from flask import Flask, request, jsonify
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Initialize stemmer
stemmer = PorterStemmer()
nltk.download('stopwords')

# Get a list of English stopwords
stop_words = set(stopwords.words('english'))

# Define keyword synonyms for research types
keyword_mapping = {
    "A/B (Split Test)": ["test", "testing", "tested", "A/B", "AB"],
}

# Function to expand only research types with synonyms
def expand_research_types(research_types, keyword_mapping):
    expanded_keywords = []
    for keyword in research_types:
        expanded_keywords.append(keyword)
        if keyword in keyword_mapping:
            expanded_keywords.extend(keyword_mapping[keyword])
    return list(set(expanded_keywords))  # Remove duplicates

# Function to find keywords in the summary text
def check_keywords(text, keyword_list):
    selected_keywords = []
    text_lower = text.lower()
    text_words = [stemmer.stem(word) for word in text_lower.split()]

    for keyword in keyword_list:
        keyword_lower = keyword.lower()
        keyword_words = [word for word in keyword_lower.split() if word not in stop_words]
        keyword_stems = [stemmer.stem(word) for word in keyword_words]

        # Check for exact match, all words match, or any word match
        if keyword_lower in text_lower:
            selected_keywords.append(keyword)
        elif all(stem in text_words for stem in keyword_stems):
            selected_keywords.append(keyword)
        elif any(stem in text_words for stem in keyword_stems):
            selected_keywords.append(keyword)

    return selected_keywords

@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        summary = data.get('summary', '')

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        # Expand only research types with synonyms
        research_types = expand_research_types(data.get('research_types', []), keyword_mapping)

        # Check for keywords in the provided summary
        selected_research_types = check_keywords(summary, research_types)
        selected_categories = check_keywords(summary, data.get('categories', []))
        selected_elements = check_keywords(summary, data.get('elements', []))
        selected_tools = check_keywords(summary, data.get('tools', []))
        selected_goals = check_keywords(summary, data.get('goals', []))
        selected_industries = check_keywords(summary, data.get('industries', []))

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
