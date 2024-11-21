from flask import Flask, request, jsonify
from dotenv import load_dotenv, find_dotenv
from functions import chunk_data, load_embeddings_chroma, ask_questions, main 

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

# Load the LLM and data embeddings for analysis
sales_data = main()

@app.route('/api/rep_performance', methods=['GET'])
def rep_performance():
    """
    Endpoint to analyze the performance of a specific sales representative.
    Parameters:
        - rep_id: Unique identifier for the sales representative.
    Returns:
        - Performance analysis and feedback as JSON.
    """
    rep_id = request.args.get('rep_id')
    if not rep_id:
        return jsonify({'error': 'Missing parameter: rep_id'}), 400

    query_text = f"Analyze the performance of sales representative with ID {rep_id}."
    result = ask_questions(query_text, sales_data)
    return jsonify({'rep_id': rep_id, 'analysis': result.get('answer', 'No data available')})

@app.route('/api/team_performance', methods=['GET'])
def team_performance():
    """
    Endpoint to summarize overall sales team performance.
    Returns:
        - Summary of the sales team's performance as JSON.
    """
    query_text = "Provide an analysis of the overall sales team's performance."
    result = ask_questions(query_text, sales_data)
    return jsonify({'team_performance': result.get('answer', 'No data available')})

@app.route('/api/performance_trends', methods=['GET'])
def performance_trends():
    """
    Endpoint to analyze sales trends and forecast future performance.
    Parameters:
        - time_period: Time period for trend analysis (e.g., 'monthly', 'quarterly').
    Returns:
        - Trends and forecast as JSON.
    """
    time_period = request.args.get('time_period', 'monthly')
    query_text = f"Analyze sales trends and forecast performance for the {time_period} period."
    result = ask_questions(query_text, sales_data)
    return jsonify({'time_period': time_period, 'trends': result.get('answer', 'No data available')})

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=False)
