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


import re

# def extract_lift_and_metric(text, goals):
#     """
#     Extract lift (x%) and associated metric (y) from the text.
#     Handles patterns like:
#     - Positive lift: 
#       x% lift in (or of) y, x% uplift in (or of) y, x% increase in (or of) y,
#       x% improvement in (or of) y, improvement of x% in (or of) y,
#       increase in y of x%, x% higher y, x% uptick in (or of) y, x% more y
#     - Negative lift: 
#       x% less y, x% fewer y, increased [...] by x%, improved [...] by x%,
#       boosted [...] by x%, x% lower y
#     Returns a list of dictionaries with "lift" and "metric". Filters metrics against a predefined list of goals.
    
#     Args:
#         text (str): Input text containing lift and metrics information.
#         goals (list): List of valid goal metrics.
    
#     Returns:
#         list: A list of dictionaries with "lift" and "metric" that match the goals.
#     """
#     pattern = r"""
#         (\d+)%\s*(?:lift|uplift|increase|improvement|higher|uptick|more)\s*(?:in|of)?\s*(\w[\w\s]*?)\b
#         |improvement\s*of\s*(\d+)%\s*(?:in|of)\s*(\w[\w\s]*?)\b
#         |increase\s*in\s*(\w[\w\s]*?)\s*of\s*(\d+)%\b
#         |(\d+)%\s*(?:less|fewer|lower)\s*(\w[\w\s]*?)\b
#         |(?:increased|improved|boosted)\s*(?:.*?)\s*by\s*(\d+)%\b\s*(\w[\w\s]*?)\b
#         |(\d+)%\s*(?:lift)\s*(?:\s*\(or\s*of\s*\))?\s*(in|of)?\s*(\w[\w\s]*?)\b
#     """
    
#     matches = re.findall(pattern, text.lower(), re.VERBOSE)
#     results = []

#     # Normalize the goals to lowercase for comparison
#     normalized_goals = [goal.lower() for goal in goals]

#     for match in matches:
#         # Positive cases
#         if match[0] and match[1]:  # x% lift/uplift/increase/improvement/higher/uptick/more in y
#             lift, metric = match[0], match[1]
#         elif match[2] and match[3]:  # improvement of x% in y
#             lift, metric = match[2], match[3]
#         elif match[4] and match[5]:  # increase in y of x%
#             lift, metric = match[5], match[4]
#         # Negative cases
#         elif match[6] and match[7]:  # x% less/fewer/lower y
#             lift, metric = f"-{match[6]}", match[7]
#         elif match[8] and match[9]:  # increased/improved/boosted [...] by x%
#             lift, metric = f"-{match[8]}", match[9]
#         elif match[10] and match[11]:  # x% lift (or of) in y
#             lift, metric = match[10], match[11]
#         else:
#             continue

#         # Normalize metric for comparison
#         metric = metric.strip().lower()
#         print ("metric",metric)

#         # Match metric with goals using substring or fuzzy matching
#         matched_goal = next((goal for goal in normalized_goals if metric in goal), "")

#         # Append the matched goal or leave metric empty if no match found
#         results.append({"lift": f"{lift}%", "metric": matched_goal if matched_goal else ""})

#     return results

import re

def extract_lift_and_metric(text, goals):
    """
    Extract lift (x%) and associated metric (y) from the text.
    Match the extracted metric fully or partially with the predefined goals.
    
    Args:
        text (str): Input text containing lift and metrics information.
        goals (list): List of valid goal metrics.
    
    Returns:
        list: A list of dictionaries with "lift" and "metric" that match the goals.
    """
    pattern = r"""
        (\d+)%\s*(?:lift|uplift|increase|improvement|higher|uptick|more)\s*(?:in|of)?\s*(\w[\w\s]*?)\b
        |improvement\s*of\s*(\d+)%\s*(?:in|of)\s*(\w[\w\s]*?)\b
        |increase\s*in\s*(\w[\w\s]*?)\s*of\s*(\d+)%\b
        |(\d+)%\s*(?:less|fewer|lower)\s*(\w[\w\s]*?)\b
        |(?:increased|improved|boosted)\s*(?:.*?)(?:by\s*(\d+)%\b)\s*(\w[\w\s]*?)\b
        |(\d+)%\s*(?:lift)\s*(?:\s*\(or\s*of\s*\))?\s*(in|of)?\s*(\w[\w\s]*?)\b
    """
    
    matches = re.findall(pattern, text.lower(), re.VERBOSE)
    results = []

    # Normalize the goals to lowercase for comparison
    normalized_goals = [goal.lower() for goal in goals]

    def match_with_goals(metric, goals):
        """
        Check if the metric matches or is part of any goal.
        
        Args:
            metric (str): The extracted metric.
            goals (list): List of normalized goals.
        
        Returns:
            str: The best-matched goal, or an empty string if no match is found.
        """
        metric = metric.strip()
        for goal in goals:
            if metric in goal or goal in metric:  # Check for substring match
                return goal
        return ""

    for match in matches:
        # Positive cases
        if match[0] and match[1]:  # x% lift/uplift/increase/improvement/higher/uptick/more in y
            lift, metric = match[0], match[1]
        elif match[2] and match[3]:  # improvement of x% in y
            lift, metric = match[2], match[3]
        elif match[4] and match[5]:  # increase in y of x%
            lift, metric = match[5], match[4]
        # Negative cases
        elif match[6] and match[7]:  # x% less/fewer/lower y
            lift, metric = f"-{match[6]}", match[7]
        elif match[8] and match[9]:  # increased/improved/boosted [...] by x%
            lift, metric = f"-{match[8]}", match[9]
        elif match[10] and match[11]:  # x% lift (or of) in y
            lift, metric = match[10], match[11]
        else:
            continue

        # Normalize metric for comparison
        metric = metric.strip().lower()

        # Match metric with the goals
        best_match = match_with_goals(metric, normalized_goals)

        results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})

    return results






import re

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
    - x% confidence
    Returns the first match or the maximum value if multiple matches exist.
    """
    # Updated regex to include "confidence"
    pattern = r"(\d+)%\s*(?:stat(?:istical)?(?:\s+sig(?:nificance)?)?|confidence)"
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

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        selected_categories = check_keywords(summary, categories, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_elements = check_keywords(summary, elements, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_tools = check_keywords(summary, tools, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_goals = check_keywords(summary, goals, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_research_types = check_keywords(summary, research_types, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_industries = check_keywords(summary, industries)
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
            "selected_confidence_levels": selected_confidence_levels, 
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
