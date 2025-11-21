# Problem 1: Two-Agent System with Function Calling & Structured Output

## Overview
This solution implements a two-agent system where:
- **Info Agent**: Retrieves flight information and returns structured JSON data
- **QA Agent**: Processes user queries, calls the Info Agent, and returns user-friendly responses

## Features
- Multi-agent coordination with function calling
- **Groq LLM Integration**: Uses Llama 3.1 for enhanced natural language understanding
- Structured JSON output for all responses
- Flight number extraction from natural language queries
- Mock flight database for testing
- Comprehensive error handling
- Fallback to regex-based extraction if LLM unavailable

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Groq API key:
```bash
# Edit api_keys.env and add your Groq API key
# Get your free API key from: https://console.groq.com
GROQ_API_KEY=your_actual_groq_api_key
```

**Note**: The system works without an API key using regex-based extraction, but LLM mode provides better natural language understanding.

## Usage

Run the main script to test all functions:
```bash
python main.py
```

## Functions

### 1. `get_flight_info(flight_number: str) -> dict`
Retrieves flight information from the mock database.

**Example:**
```python
result = get_flight_info("AI123")
# Returns: {"flight_number": "AI123", "departure_time": "08:00 AM", "destination": "Delhi", "status": "Delayed"}
```

### 2. `info_agent_request(flight_number: str) -> str`
Info Agent that returns flight data as a JSON string.

**Example:**
```python
result = info_agent_request("AI123")
# Returns: '{"flight_number": "AI123", "departure_time": "08:00 AM", "destination": "Delhi", "status": "Delayed"}'
```

### 3. `qa_agent_respond(user_query: str) -> str`
QA Agent that processes natural language queries and returns structured responses.

**Example:**
```python
result = qa_agent_respond("When does Flight AI123 depart?")
# Returns: '{"answer": "Flight AI123 departs at 08:00 AM to Delhi. Current status: Delayed."}'
```

## Test Cases

The script includes comprehensive test cases covering:
- Valid flight queries
- Non-existent flights
- Various query formats
- Edge cases

## Implementation Details

- **LLM Provider**: Groq (llama-3.3-70b-versatile model)
- **Flight Database**: Mock database with 3 sample flights (AI123, AI456, AI789)
- **Dual Mode**: 
  - LLM mode: Uses Groq for query understanding and response generation
  - Regex mode: Fallback pattern matching for flight number extraction
- **JSON Validation**: All outputs are valid JSON strings
- **Error Handling**: Graceful handling of missing flights, invalid queries, and API errors

## Getting Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to `api_keys.env`

Groq offers:
- Fast inference speeds
- Free tier available
- Multiple open-source models (Llama, Mixtral, Gemma)

## Architecture

```
User Query → QA Agent → Info Agent → Flight Database
                ↓           ↓
            Extract     Retrieve
            Flight#     Flight Info
                ↓           ↓
            Format ← Parse JSON
            Response
                ↓
            Return JSON
```
