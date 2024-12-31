
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
    # Positive patterns for lift and metric extraction
    positive_patterns = [
        r"(\d+\.\d+|\d+)%\s*(?:lift|uplift|increase|improvement|higher|uptick|more)\s*(?:in|of)?\s*(\w[\w\s]*?)\b",
        r"improvement\s*of\s*(\d+\.\d+|\d+)%\s*(?:in|of)\s*(\w[\w\s]*?)\b",
        r"increase\s*in\s*(\w[\w\s]*?)\s*of\s*(\d+\.\d+|\d+)%\b",
        r"(\d+\.\d+|\d+)%\s*(?:lift)\s*(?:\s*\(or\s*of\s*\))?\s*(in|of)?\s*(\w[\w\s]*?)\b",
        
    ]

    positive_patterns2 = [
        r"increase\s+in\s+(.*?)\s+of\s+(\d+(\.\d+)?)%",
        r"(\w+)\s+was\s+(\d+(\.\d+)?)%\s+(?:higher|more)",

    ]

    
    # Negative patterns for lift and metric extraction
    negative_patterns = [
        r"(\d+\.\d+|\d+)%\s*(?:less|fewer|lower)\s*(\w[\w\s]*?)\b",
    ]

    negative_patterns2=[
        # r"(\w+)\s+was\s+(boosted|improved|increased)\s+by\s+(\d+(\.\d+)?)%",
        # r"(\w+(\s+\w+){0,1})\s+increased\s+by\s+(\d+(\.\d+)?)%",
    ]

    positive_patterns3=[
        # r"(\w+)\s+was\s+(boosted|improved|increased)\s+by\s+(\d+(\.\d+)?)%",
        r"(\w+(\s+\w+){0,1})\s+increased\s+by\s+(\d+(\.\d+)?)%",
    ]

    positive_patterns4=[
        r"(\w+)\s+was\s+(boosted|improved|increased)\s+by\s+(\d+(\.\d+)?)%",
        # r"(\w+(\s+\w+){0,1})\s+increased\s+by\s+(\d+(\.\d+)?)%",
    ]

    boosted_positive_pattern_alt = [
        r"(boosted|improved|increased|boosts|increases|improves)\s+(.*?)\s+by\s+(\d+(\.\d+)?)%",
    ]
  
    
    text = text.lower()
    print("Text being analyzed:", text)
    
    # Extract matches from positive patterns
    results = []
    normalized_goals = [goal.lower() for goal in goals]

    # def match_with_goals(metric, goals):
    #     metric_words = metric.split()  # Limit to max 3 words
    #     for goal in goals:
    #         for word in metric_words:
    #             if word.lower() in goal.lower():
    #                 return metric  # Return the original metric if it aligns with any goal
    #     return ""

    # def match_with_goals(metric, goals):
    #     """
    #     Match metric with goals, considering partial matches based on a minimum of 4 consecutive characters.
    #     Exclude unnecessary words like 'the', 'is', 'am', 'are' from the metric before returning.
    #     """
    #     # Define a set of stop words to exclude
    #     stop_words = {"the", "is", "am", "are", "a", "an", "of", "to", "in", "on", "at", "for", "by", "with", "and", "or"}
        
    #     # Split the metric into individual words and filter out stop words
    #     metric_words = [word for word in metric.split() if word.lower() not in stop_words]
    #     print('metric_words',metric_words)
    #     cleaned_metric = " ".join(metric_words)  # Join the cleaned words back into a string
    #     print('cleaned_metric',cleaned_metric)

    #     for goal in goals:
    #         print('goal',goal)
    #         for word in metric_words:
    #             print('word',word)
    #             word = word.lower()
    #             goal = goal.lower()
                
    #             if word.lower() in goal.lower():
    #                 return cleaned_metric  # Return the cleaned metric if it aligns with any goal

    #     return ""

    def match_with_goals(metric, goals):
        """
        Match metric with goals, considering partial matches based on stemming.
        Exclude unnecessary words like 'the', 'is', 'am', 'are' from the metric before returning.
        """
        # Initialize stemmer
        stemmer = PorterStemmer()

        # Define a set of stop words to exclude
        stop_words = {"the", "is", "am", "are", "a", "an", "of", "to", "in", "on", "at", "for", "by", "with", "and", "or"}
        
        # Split the metric into individual words and filter out stop words
        metric_words = [word for word in metric.split() if word.lower() not in stop_words]
        cleaned_metric = " ".join(metric_words)  # Join the cleaned words back into a string

        # Stem the metric words
        stemmed_metric_words = [stemmer.stem(word.lower()) for word in metric_words]

        # Iterate through the goals to find matches
        for goal in goals:
            stemmed_goal = stemmer.stem(goal.lower())
            for stemmed_word in stemmed_metric_words:
                if stemmed_word in stemmed_goal:
                    return cleaned_metric  # Return the cleaned metric if it aligns with any goal

        return ""


    # Process positive patterns
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
        print(f"Matches for positive pattern '{pattern}':", matches)

        for match in matches:
            if match[0] and match[1]:
                lift, metric = match[1], match[0]
            else:
                continue
            
            metric = metric.strip().lower()
            best_match = match_with_goals(metric, goals)
            print(f"Best match for metric '{metric}': '{best_match}'")

            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})
    
    # Process negative patterns
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

        # Process negative patterns
    for pattern in negative_patterns2:
        matches = re.findall(pattern, text)
        print(f"Matches for negative pattern2 '{pattern}':", matches)

        for match in matches:
            print('match',match)
            if match[0] and match[2]:
                lift, metric = f"-{match[2]}", match[1] 
            metric = metric.strip().lower()
            print('metric',metric)
            best_match = match_with_goals(metric, goals)
            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})


        # Process negative patterns
    for pattern in positive_patterns3:
        matches = re.findall(pattern, text)
        print(f"Matches for posi pattern3'{pattern}':", matches)

        for match in matches:
            print('match',match)
            if match[0] and match[2]:
                lift, metric = match[2], match[1] 
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
                lift, metric = match[2], match[0] 
            metric = metric.strip().lower()
            print('metric',metric)
            best_match = match_with_goals(metric, goals)
            results.append({"lift": f"{lift}%", "metric": best_match if best_match else ""})

    for pattern in boosted_positive_pattern_alt:
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        matches_list = list(matches)
        print(f"Matches for pattern '{pattern}':", matches_list)
        
        # Now iterate over matches to process them
        for match in matches_list:
            print('Match:', match)  # Debug: should print the match object if matches exist
            
            if match:
                lift = match.group(3) 
                metric = match.group(2).strip()
                metric = metric.strip().lower()
                best_match = match_with_goals(metric, goals)

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
