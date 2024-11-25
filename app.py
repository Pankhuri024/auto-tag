from flask import Flask, request, jsonify
import openai  # Install the OpenAI Python library using `pip install openai`

app = Flask(__name__)

# Predefined categories, elements, aspects, tools, goals, and research types
tools = ["FullStory", "Google Analytics", "Hotjar", "Convert"]
categories = ["Branding", "Blogs/Content Marketing", "Email", "Events", "ABM & Personalization",
              "Persona Development", "Search", "Social Media", "Sponsorships", "Clients", "Contractors/Suppliers",
              "Business Risk & Liability", "Contractor Prequalification", "Cybersecurity", "ESG & Sustainability",
              "Health & Safety", "Supply Chain Risk", "Worker Compliance"]
elements = ["Images", "Copy", "Layout", "Design", "Video", "Functional", "Navigation"]
research_types = ["General", "Data Analysis", "User Study", "Survey", "A/B (Split Test)", "Market Research"]
goals = ["Contact Sales", "Logins", "Gated Asset Registrations", "Chat Starts", "Event Registrations", "Engagement", "Site Traffic"]

# Configure OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        app.logger.info(f"Received request data: {data}")
        summary = data.get('summary', '')

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        # Use GPT model to analyze the summary and suggest categories, tools, goals, etc.
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes insights based on context."},
                {
                    "role": "user",
                    "content": f"""Analyze the following summary and suggest categories, tools, goals, elements, and research types:
                    Summary: {summary}
                    Categories: {categories}
                    Tools: {tools}
                    Goals: {goals}
                    Elements: {elements}
                    Research Types: {research_types}
                    Provide the response as a JSON object with keys: selected_categories, selected_tools, selected_goals, selected_elements, and selected_research_types."""
                }
            ]
        )

        # Extract AI's response
        ai_response = response['choices'][0]['message']['content']
        app.logger.info(f"AI Response: {ai_response}")

        # Convert the response to JSON
        return jsonify({"ai_suggestions": ai_response})

    except Exception as e:
        app.logger.error(f"Error processing insight: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
