from flask import Flask, request, jsonify
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
    excluded_stems = {stemmer.stem(word) for word in {"research", "researching", "researched", "searched"}}  # Stem excluded words
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

# def extract_lift_and_metric(summary):
#     """
#     Extracts lift (x%) and associated metric (y) from the summary.
#     Looks for patterns like 'x% increase in y', 'x% improvement of y', etc.
#     """
#     pattern = re.compile(
#         r'(\d+\.?\d*)\s?%\s?(?:lift|uplift|increase|improvement)?\s?(?:in|of)?\s([a-zA-Z\s]+)',
#         re.IGNORECASE
#     )
#     matches = pattern.findall(summary)

#     lift_metric_pairs = []
#     for match in matches:
#         lift = float(match[0])  # Extract lift as a float
#         metric = match[1].strip()  # Extract and clean metric
#         lift_metric_pairs.append((lift, metric))
    
#     return lift_metric_pairs

# def filter_metrics_by_goals(lift_metric_pairs, goals):
#     """
#     Filters the extracted metrics based on alignment with organizational goals.
#     """
#     filtered_pairs = []
#     for lift, metric in lift_metric_pairs:
#         # Check if the metric aligns with any goal
#         if any(goal.lower() in metric.lower() for goal in goals):
#             filtered_pairs.append((lift, metric))
#     return filtered_pairs


def extract_lift_and_metric_ai(summary, goals):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return jsonify({'message': 'OPENAI_API_KEY environment variable is not set.'}), 500
    """
    Uses OpenAI GPT API to extract lift and metric based on organizational goals and specific patterns.
    """
    prompt = f"""
    Analyze the following summary text and extract:

Any percentage change mentioned using these patterns:

x% lift in (or of) y
x% uplift in (or of) y
x% increase in (or of) y
x% improvement in (or of) y
improvement of x% in (or of) y
increase in y of x%
x% uptick in (or of) y
x% higher y
x% more y
For these patterns, insert +x into the 'Lift' field.

Any percentage change mentioned using these patterns:

x% lower y
x% less y
x% fewer y
increased [...] by x%, 
improved [...] by x%, 
boosted [...] by x%
For these patterns, insert -x into the 'Lift' field.
    2. The associated metric it applies to.
    Only include metrics that align with these organizational goals: {', '.join(goals)}.

    Text: "{summary}"

    Output format:
    - Lift: [Percentage]
    - Metric: [Metric Name]
    """
    try:
        # Initialize OpenAI model
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        response = llm(prompt)
        print("response", response)
        return response.content
    except Exception as e:
        # Handle API errors or connection issues
        print(f"Error calling OpenAI API: {e}")
        return None  # Fallback to None if an error occurs



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
        lift_metric_pairs = extract_lift_and_metric_ai(summary, goals)
        # filtered_pairs = filter_metrics_by_goals(lift_metric_pairs, goals)

        # # Split filtered pairs into separate lists
        # lift = [pair[0] for pair in filtered_pairs]
        # metrics = [pair[1] for pair in filtered_pairs]

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        selected_categories = check_keywords(summary, categories, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_elements = check_keywords(summary, elements, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_tools = check_keywords(summary, tools)
        selected_goals = check_keywords(summary, goals, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_research_types = check_keywords(summary, research_types, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_industries = check_keywords(summary, industries)

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
            "selected_lift": lift_metric_pairs,
            # "selected_metrics": metrics if metrics else "No metric found",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
