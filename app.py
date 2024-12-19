from flask import Flask, request, jsonify, Response
import re
import json  # For loading the JSON config
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from langchain_openai import ChatOpenAI
import openai
import os


app = Flask(__name__)

import nltk
# Initialize stemmer
stemmer = PorterStemmer()
nltk.download('stopwords')

# Get a list of English stopwords
stop_words = set(stopwords.words('english'))

# Load synonyms from config.json dynamically
try:
    with open("config.json", "r") as file:
        config_data  = json.load(file)
        RESEARCH_TYPE_SYNONYMS = config_data.get("research_type_synonyms", {})
        # print("Loaded config.json content:", json.dumps(RESEARCH_TYPE_SYNONYMS, indent=4))
except Exception as e:
    RESEARCH_TYPE_SYNONYMS = {}
    print(f"Error loading config.json: {e}")


# Function to normalize text
def clean_text(text):
    """Normalize text by removing special characters and extra spaces."""
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace special characters with a space
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    return text.strip()

# Function to find keywords in the summary text
def check_keywords(text, keyword_list, synonyms=None):
    selected_keywords = []
    text_clean = clean_text(text.lower())  # Clean and lowercase the text
    
    # Stem the cleaned text into a list of stemmed words
    text_words = [stemmer.stem(word) for word in text_clean.split()]
    excluded_stems = {stemmer.stem(word) for word in {"research", "researching", "researched", "searched","engaged"}}  # Stem excluded words
    # excluded_words = {"research", "researching", "researched", "searched"}  # Add any other unwanted words here
    
    for keyword in keyword_list:
        keyword_clean = clean_text(keyword.lower())  # Clean and lowercase the keyword
        keyword_words = [word for word in keyword_clean.split() if word not in stop_words]
        keyword_stems = [stemmer.stem(word) for word in keyword_words]

        if keyword_clean in text_clean:
            selected_keywords.append(keyword)
        elif all(stem in text_words for stem in keyword_stems) and not any(word in excluded_stems for word in keyword_stems):
            selected_keywords.append(keyword)

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
                if all(stem in text_words for stem in synonym_words) and not any(word in excluded_stems for word in synonym_words):
                    selected_keywords.append(keyword)
                    break
    
    return selected_keywords

def extract_lift_and_metric(summary, goals):
    """
    Extracts lift (percentage changes) and associated metrics from the summary text.
    Includes patterns like `x% increase in y`.
    """
    patterns = [
        (r'(\d+)%\s+(?:lift|uplift|increase|improvement|uptick|higher|more)\s+(?:in|of)\s+([\w\s]+)', '+'),
        (r'improvement\s+of\s+(\d+)%\s+(?:in|of)\s+([\w\s]+)', '+'),
        (r'increase\s+in\s+([\w\s]+)\s+of\s+(\d+)%', '+'),
        (r'(\d+)%\s+(?:increase|boost|gain)\s+in\s+([\w\s]+)', '+'),
        (r'(\d+)%\s+of\s+([\w\s]+)', '+'),
        (r'(\d+)%\s+(?:lower|less|fewer)\s+([\w\s]+)', '-'),
        (r'decrease\s+in\s+([\w\s]+)\s+of\s+(\d+)%', '-'),
        (r'reduction\s+of\s+(\d+)%\s+(?:in|of)\s+([\w\s]+)', '-'),
        (r'(\d+)%\s+decrease\s+in\s+([\w\s]+)', '-')
    ]

    results = []
    for pattern, sign in patterns:
        matches = re.findall(pattern, summary.lower())
        for match in matches:
            if len(match) == 2:
                lift, metric = match[0], match[1].strip()  # Ensure lift comes first
            else:
                metric, lift = match[0], match[1].strip()  # Handle alternate formats
            
            metric_cleaned = " ".join(metric.split())  # Clean up extra spaces
            
            # Filter by goals if provided
            if not goals or any(goal.lower() in metric_cleaned for goal in goals):
                results.append({"lift": int(lift), "metric": metric_cleaned})

    return results


def extract_confidence_level(text):
    """
    Extract a single confidence level from the text.
    Looks for patterns like:
    - x% stat sig
    - x% stat sig.
    - x% stat significance
    - x% statistical sig.
    - x% statistical significance
    - x% statistical sig
    Returns the first match or the maximum value if multiple matches exist.
    """
    pattern = r"(\d+)%\s*(?:stat(?:istical)?(?:\s+sig(?:nificance)?)?)"
    matches = re.findall(pattern, text.lower())  # Case-insensitive search
    
    if matches:
        # Convert the matches to integers
        confidence_levels = [int(match) for match in matches]
        # Return the first match or max(confidence_levels) if you want the maximum
        return confidence_levels[0]  # Use max(confidence_levels) if required
    return None  # Return None if no match is found

@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        summary = data.get('summary', '')  # Default to empty string if not provided
        goals = data.get('goals', [])
        categories = data.get('categories', [])
        tools = data.get('tools', [])
        elements = data.get('elements', [])
        research_types = data.get('research_types', [])
        industries = data.get('industries', [])
        # Extract lift and metric
        # lift_metric_pairs = extract_lift_and_metric_ai(summary, goals)
   

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        selected_categories = check_keywords(summary, categories, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_elements = check_keywords(summary, elements, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_tools = check_keywords(summary, tools)
        selected_goals = check_keywords(summary, goals, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_research_types = check_keywords(summary, research_types, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_industries = check_keywords(summary, industries)
        # Extract lift and metric using regex-based method
        lift_metric_pairs = extract_lift_and_metric(summary, goals)

        # Separate lifts and metrics
        lift_values = [item['lift'] for item in lift_metric_pairs]
        print("lift_values",lift_values)
        metric_values = [item['metric'] for item in lift_metric_pairs]
        print("metric_values",metric_values)
        # Extract confidence levels
        selected_confidence_levels = extract_confidence_level(summary)

        # Priority logic for A/B (Split Test)
        if lift_metric_pairs:  # If lift values are detected
            ab_split_test_synonyms = RESEARCH_TYPE_SYNONYMS.get("A/B (Split Test)", [])
            if any(keyword in summary.lower() for keyword in ab_split_test_synonyms):
                if "A/B (Split Test)" not in selected_research_types:
                    selected_research_types.insert(0, "A/B (Split Test)")  # Ensure priority

        return jsonify({
            "selected_categories": selected_categories,
            "selected_elements": selected_elements,
            "selected_tools": selected_tools,
            "selected_goals": selected_goals,
            "selected_research_types": selected_research_types,
            "selected_industries": selected_industries,
            "selected_lift": lift_values,
            "selected_metric": metric_values,
            "confidence_levels": selected_confidence_levels, 
            # "selected_metrics": metrics if metrics else "No metric found",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
