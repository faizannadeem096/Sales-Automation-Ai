# Sales-Automation-Ai
This repository features an AI-driven solution for automating sales data analysis and generating actionable insights. By combining HuggingFace embeddings and GPT models, the application processes raw CSV sales data into vectorized embeddings and uses natural language queries to deliver detailed performance analysis and trend forecasts.
Automated Sales Analysis and Insights

Overview
This project automates the analysis of sales data, transforming raw CSV files into actionable insights using vector embeddings and GPT-based models. It supports REST API endpoints for querying performance metrics and sales trends.

Features:
Converts sales CSV data into vector embeddings.
Utilizes Chroma for efficient retrieval.
Leverages GPT models for natural language insights.

Provides RESTful API for querying:
/api/rep_performance: Analyze a sales representative's performance.
/api/team_performance: Summarize team performance.
/api/performance_trends: Analyze trends and forecast future sales.

Technologies Used:
LangChain: Document loaders, text splitters, and vector stores.
HuggingFace: Embedding models for vectorization.
OpenAI: GPT-based content generation.

Flask: API backend.
Chroma: Vector store for efficient retrieval.
Setup and Installation
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/sales-automation.git
cd sales-automation

Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Add your API keys to a .env file:
env
Copy code
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
OPENAI_API_KEY=your_openai_api_key

Run the application:
bash
Copy code
python app.py
Usage

Use endpoints with query parameters:
Example: /api/rep_performance?rep_id=123
Example: /api/performance_trends?time_period=monthly
Directory Structure
bash
Copy code
sales-automation/
│
├── app.py                      # Flask app with API endpoints
├── functions.py                # Helper functions for data processing and embeddings
├── requirements.txt            # Dependencies
├── .env                        # Environment variables (add your keys here)
├── sales_data.csv              # Example sales data
└── README.md                   # Documentation
Future Improvements
Add more complex trend analysis using machine learning models.
Enhance performance for large datasets.
Extend APIs for custom report generation.
License
MIT License.

