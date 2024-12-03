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

# Define keyword synonyms
keyword_mapping = {
    "A/B (Split Test)": ["test", "testing", "tested", "A/B", "AB"],
}

# Function to find keywords in the summary text
def check_keywords(text, keyword_mapping):
    selected_keywords = []
    text_lower = text.lower()

    # Stem the entire text into a list of stemmed words for single-word matching
    text_words = [stemmer.stem(word) for word in text_lower.split()]

    for main_keyword, synonyms in keyword_mapping.items():
        all_keywords = [main_keyword] + synonyms
        matched = False

        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            keyword_words = [word for word in keyword_lower.split() if word not in stop_words]
            keyword_stems = [stemmer.stem(word) for word in keyword_words]

            # Check if the entire keyword phrase exists as an exact match in the text
            if keyword_lower in text_lower:
                matched = True
            # Match if ALL stemmed words in the multi-word keyword exist in the text
            elif all(stem in text_words for stem in keyword_stems):
                matched = True
            # Match if ANY single word from the keyword exists in the text
            elif any(stem in text_words for stem in keyword_stems):
                matched = True

        # If a match is found for any synonym or the main keyword, add only the main keyword
        if matched and main_keyword not in selected_keywords:
            selected_keywords.append(main_keyword)

    return selected_keywords

@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        summary = data.get('summary', '')  # Default to empty string if not provided

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        # Check for keywords in the provided summary
        selected_research_types = check_keywords(summary, keyword_mapping)

        # Return the auto-selected values
        return jsonify({
            "selected_research_types": selected_research_types,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
