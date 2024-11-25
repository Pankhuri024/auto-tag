from flask import Flask, request, jsonify

app = Flask(__name__)

# Predefined categories, elements, aspects, tools, goals, and research types
# Tools
tools = ["FullStory","Google Analytics","Hotjar","Convert"]

# Categories
categories = ["Branding","Blogs/Content Marketing","Email","Events","ABM & Personalization","Persona Development","Search","Social Media","Sponsorships","Clients","Contractors/Suppliers","Business Risk & Liability","Contractor Prequalification","Cybersecurity","ESG & Sustainability","Health & Safety","Supply Chain Risk","Worker Compliance"]

# Elements & Aspects
elements = ["Images","Copy","Layout","Design","Video","Functional","Navigation"]
# Research Types
research_types = ["General","Data Analysis","User Study","Survey","A/B (Split Test)","Market Research"]
# Goals
goals = ["Contact Sales","Logins","Gated Asset Registrations","Chat Starts","Event Registrations","Engagement","Site Traffic"]


# Function to find keywords in the summary text
def check_keywords(text, keyword_list):
    selected_keywords = []
    for keyword in keyword_list:
        if keyword.lower() in text.lower():
            selected_keywords.append(keyword)
    return selected_keywords

@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        summary = data.get('summary', '')

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        # Check for keywords in the provided summary
        selected_categories = check_keywords(summary, categories)
        selected_elements = check_keywords(summary, elements)
        selected_tools = check_keywords(summary, tools)
        selected_goals = check_keywords(summary, goals)
        selected_research_types = check_keywords(summary, research_types)

        # Return the auto-selected values
        return jsonify({
            "selected_categories": selected_categories,
            "selected_elements": selected_elements,
            "selected_tools": selected_tools,
            "selected_goals": selected_goals,
            "selected_research_types": selected_research_types
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
