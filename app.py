
from flask import Flask, request, jsonify, Response
import re
import json  # For loading the JSON config
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from langchain_openai import ChatOpenAI
import openai
import os
# from typing import List, Dict


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
    """Normalize text by removing special characters and extra spaces, but keep '%' as is."""
    text = re.sub(r'[^\w\s%]', '', text)  # Remove special characters except % and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()


# Function to find keywords in the summary text
def check_keywords(text, keyword_list, synonyms=None):
    selected_keywords = []
    # print('text',text)
    text_clean = clean_text(text.lower())  # Clean and lowercase the text
    # print('text_clean',text_clean)
    
    # Stem the cleaned text into a list of stemmed words
    text_words = [stemmer.stem(word) for word in text_clean.split()]
    excluded_stems = {stemmer.stem(word) for word in {"research", "researching", "researched", "searched","engaged"}}  # Stem excluded words
    # excluded_words = {"research", "researching", "researched", "searched"}  # Add any other unwanted words here
    
    
    for keyword in keyword_list:
        keyword_clean = clean_text(keyword.lower())  # Clean and lowercase the keyword
        keyword_words = [word for word in keyword_clean.split() if word not in stop_words]
        keyword_stems = [stemmer.stem(word) for word in keyword_words]

        # print(f"Processing keyword: {keyword}")
        # print(f"Keyword stems: {keyword_stems}, Text words: {text_words}")

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
                # print('synonym_clean',synonym_clean)

                # Check for dynamic patterns like "X%"
                # if "x%" in synonym_clean and re.search(r'\b\d+(\.\d+)?%\b', text_clean):  
                #     selected_keywords.append(keyword)
                #     break
                
                # Check if synonym exists as a whole phrase
                if synonym_clean in text_clean:
                    selected_keywords.append(keyword)
                    break  # Avoid duplicate additions
                
                # Stem the synonym for partial matching
                synonym_words = [stemmer.stem(word) for word in synonym_clean.split() if word not in stop_words]
                if all(stem in text_words for stem in synonym_words) and not any(word in excluded_stems for word in synonym_words):
                # if all(stem in text_words for stem in keyword_stems) and not any(word in excluded_stems for word in keyword_stems):
                    selected_keywords.append(keyword)
                    break
    
    return selected_keywords


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

    positive_patterns = [
        # r"(\d+\.\d+|\d+)%\s*(?:lift|uplift|increase|improvement|higher|uptick|more)\s*(?:in|of)?\s*(\w[\w\s]*?)\b",
        r"(\d+\.\d+|\d+)%\s*(?:lift|uplift|increase|improvement|higher|uptick|more)\s*(?:in|of)?\s*((?:\w+\s*){1,3})\b",
        r"improvement\s*of\s*(\d+\.\d+|\d+)%\s*(?:in|of)\s*(\w[\w\s]*?)\b",
        r"increase\s*in\s*(\w[\w\s]*?)\s*of\s*(\d+\.\d+|\d+)%\b",
        # r"(\d+\.\d+|\d+)%\s*(?:lift)\s*(?:\s*\(or\s*of\s*\))?\s*(in|of)?\s*(\w[\w\s]*?)\b",
    ]

    positive_patterns2 = [
        r"increase\s+in\s+(.*?)\s+of\s+(\d+(\.\d+)?)%",
        r"(\w+)\s+was\s+(\d+(\.\d+)?)%\s+(?:higher|more)",
    ]

    negative_patterns = [
        # r"(\d+\.\d+|\d+)%\s*(?:less|fewer|lower)\s*(\w[\w\s]*?)\b",
        r"(\d+\.\d+|\d+)%\s*(?:less|fewer|lower)\s*((?:\b\w+\b\s*){1,3})",
    ]

    negative_patterns_alt=[
       r"(reduced|decreased|declined|reduces|decreases|declines)\s+(.*?)\s+by\s+(\d+(\.\d+)?)%"
    ]

    positive_patterns3=[
        r"(\w+(\s+\w+){0,3})\s+increased\s+by\s+(\d+(\.\d+)?)%",
    ]

    positive_patterns4=[
        # r"(\w+)\s+was\s+(boosted|improved|increased)\s+by\s+(\d+(\.\d+)?)%",
        r"(\w+(\s+\w+){0,3})\s+was\s+(boosted|improved|increased)\s+by\s+(\d+(\.\d+)?)%",
    ]

    positive_patterns5=[
        # r"uptick\s+of\s+(\d+(\.\d+)?)%\s+in\s+([\w\s]+)",
        # r"uptick\s+of\s+(\d+(\.\d+)?)%\s+in\s+((?:[a-zA-Z]+\s*){1,3})"
        r"(?:uptick|lift|uplift)\s+of\s+(\d+(\.\d+)?)%\s+in\s+((?:[a-zA-Z]+\s*){1,3})"

    ]

    boosted_positive_pattern_alt = [
        r"(boosted|improved|increased|boosts|increases|improves)\s+(.*?)\s+by\s+(\d+(\.\d+)?)%",
    ]

    text = text.lower()
    print("Text being analyzed:", text)

    results = []
    normalized_goals = [goal.lower() for goal in goals]

    def match_with_goals(metric, goals):
        """
        Match metric with goals, considering partial matches based on stemming.
        Select three consecutive words if the first and last words match any goal.
        """
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        stop_words = {"the", "is", "am", "are", "a", "an", "of", "to", "in", "on", "at", "for", "by", "with", "and", "or"}
        
        # Remove stop words from the metric
        metric_words = [word for word in metric.split() if word.lower() not in stop_words]
        print('metric_words:', metric_words)

        # Stem the metric words
        stemmed_metric_words = [stemmer.stem(word.lower()) for word in metric_words]

        for goal in goals:
            # Stem the goal words
            stemmed_goal_words = [stemmer.stem(word.lower()) for word in goal.split()]

            for i in range(len(metric_words) - 2):
                # Check for three consecutive words
                segment = metric_words[i:i+3]
                segment_stemmed = stemmed_metric_words[i:i+3]
                
                # Match the first and last word with the goal
                if (segment_stemmed[0] in stemmed_goal_words and 
                    segment_stemmed[-1] in stemmed_goal_words):
                    matching_segment = " ".join(segment)
                    print('matching_segment:', matching_segment)
                    return matching_segment.title()

            # General match without the three-word rule
            matching_words = [
                metric_words[i]
                for i, stem_word in enumerate(stemmed_metric_words)
                if stem_word in stemmed_goal_words
            ]
            
            if matching_words:
                matching_segment = " ".join(matching_words)
                print('matching_segment:', matching_segment)
                return matching_segment.title()

        return ""


    def match_with_goals_2(metric, goals):
        """
        Match metric with goals, considering partial matches based on stemming.
        Exclude unnecessary words like 'the', 'is', 'am', 'are' from the metric before returning.
        """
        stemmer = PorterStemmer()

        stop_words = {"the", "is", "am", "are", "a", "an", "of", "to", "in", "on", "at", "for", "by", "with", "and", "or"}
        
        metric_words = [word for word in metric.split() if word.lower() not in stop_words]
        print('metric_words',metric_words)
        cleaned_metric = " ".join(metric_words)
        print('cleaned_metric',cleaned_metric)

        stemmed_metric_words = [stemmer.stem(word.lower()) for word in metric_words]

        for goal in goals:
            stemmed_goal = stemmer.stem(goal.lower())
            for stemmed_word in stemmed_metric_words:
                if stemmed_word in stemmed_goal:
                    print('stemmed_goal',stemmed_goal)
                    print('stemmed_word',stemmed_goal)
                    return cleaned_metric.title()

        return ""

    for pattern in positive_patterns:
        matches = re.findall(pattern, text)
        print(f"Matches for positive pattern '{pattern}':", matches)

        for match in matches:
            if match[0] and match[1]:
                lift, metric = match[0], match[1]
            elif match[2] and match[3]:
                lift, metric = match[2], match[3]
            else:
                continue
            
            metric = metric.strip().lower()
            best_match = match_with_goals(metric, goals)
            print(f"Best match for metric '{metric}': '{best_match}'")

            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})

    for pattern in positive_patterns2:
        matches = re.findall(pattern, text)
        print(f"Matches for positive pattern2 '{pattern}':", matches)

        for match in matches:
            if match[0] and match[1]:
                lift, metric = match[1], match[0]
            else:
                continue
            
            metric = metric.strip().lower()
            best_match = match_with_goals(metric, goals)
            print(f"Best match for metric '{metric}': '{best_match}'")

            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})
    
    for pattern in negative_patterns:
        matches = re.findall(pattern, text)
        print(f"Matches for negative pattern '{pattern}':", matches)

        for match in matches:
            print('match',match)
            if match[0] and match[1]:
                lift, metric = f"-{match[0]}", match[1]
            elif match[2] and match[3]:
                lift, metric = f"-{match[3]}", match[2]
            else:
                continue
            
            metric = metric.strip().lower()
            best_match = match_with_goals(metric, goals)
            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})

    for pattern in negative_patterns_alt:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        matches_list = list(matches)

        for match in matches_list:
            print('Match:', match)  
            
            if match:
                lift = f"-{match.group(3)}"
                metric = match.group(2).strip()
                metric = metric.strip().lower()
                best_match = match_with_goals(metric, goals)

                results.append({"lift": f"{lift}%", "metric": best_match})

    for pattern in positive_patterns3:
        matches = re.findall(pattern, text)
        print(f"Matches for posi pattern3'{pattern}':", matches)

        for match in matches:
            print('match',match)
            if match[0] and match[2]:
                lift, metric = match[2], match[0] 
            metric = metric.strip().lower()
            print('metric',metric)
            best_match = match_with_goals(metric, goals)
            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})

    for pattern in positive_patterns4:
        matches = re.findall(pattern, text)
        print(f"Matches for posi pattern4'{pattern}':", matches)

        for match in matches:
            print('match',match)
            if match[0] and match[2]:
                lift, metric = match[3], match[0] 
            metric = metric.strip().lower()
            print('metric',metric)
            best_match = match_with_goals(metric, goals)
            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})


    for pattern in positive_patterns5:
        matches = re.findall(pattern, text)
        print(f"Matches for posi pattern5'{pattern}':", matches)

        for match in matches:
            print('match',match)
            if match[0] and match[2]:
                lift, metric = match[0], match[2] 
            metric = metric.strip().lower()
            print('metric',metric)
            best_match = match_with_goals(metric, goals)
            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})

    for pattern in boosted_positive_pattern_alt:
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        matches_list = list(matches)
        print(f"Matches for pattern '{pattern}':", matches_list)
        
        for match in matches_list:
            print('Match:', match) 
            
            if match:
                lift = match.group(3) 
                metric = match.group(2).strip()
                metric = metric.strip().lower()
                best_match = match_with_goals_2(metric, goals)
                results.append({"lift": f"{lift}%", "metric": best_match})

    print("Final results:", results)
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
    - x% confidence
    Returns the first match or the maximum value if multiple matches exist.
    """
    pattern = r"(\d+)%\s*(?:stat(?:istical)?(?:\s+sig(?:nificance)?)?|confidence)"
    matches = re.findall(pattern, text.lower()) 
    
    if matches:
        confidence_levels = [int(match) for match in matches]
        return confidence_levels[0] 
    return None  


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
        # Attempt to extract lift and metric
        try:
            lift_metric_pairs = extract_lift_and_metric(summary, goals)
            lift_values = [item['lift'] for item in lift_metric_pairs]
            metric_values = [item['metric'] for item in lift_metric_pairs]
        except Exception as e:
            lift_metric_pairs = []
            lift_values = []
            metric_values = []
            print(f"Error extracting lift and metric: {e}")

        selected_confidence_levels = extract_confidence_level(summary)

        if lift_metric_pairs: 
            ab_split_test_synonyms = RESEARCH_TYPE_SYNONYMS.get("A/B (Split Test)", [])
            if any(keyword in summary.lower() for keyword in ab_split_test_synonyms):
                if "A/B (Split Test)" not in selected_research_types:
                    selected_research_types.insert(0, "A/B (Split Test)") 

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
