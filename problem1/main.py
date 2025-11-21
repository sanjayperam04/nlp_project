"""
Two-Agent System for Airline Flight Information
Problem 1 - Multi-agent coordination with function calling
"""

import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv('api_keys.env')

# Simple database to store flight info
FLIGHT_DATABASE = {
    "AI123": {
        "flight_number": "AI123",
        "departure_time": "08:00 AM",
        "destination": "Delhi",
        "status": "Delayed"
    },
    "AI456": {
        "flight_number": "AI456",
        "departure_time": "10:30 AM",
        "destination": "Mumbai",
        "status": "On Time"
    },
    "AI789": {
        "flight_number": "AI789",
        "departure_time": "02:15 PM",
        "destination": "Bangalore",
        "status": "On Time"
    }
}


def get_flight_info(flight_number):
    """Get flight information from database"""
    return FLIGHT_DATABASE.get(flight_number.upper())


def info_agent_request(flight_number):
    """Info Agent - returns flight data as JSON string only"""
    flight_info = get_flight_info(flight_number)
    
    if flight_info:
        return json.dumps(flight_info)
    else:
        return json.dumps({"error": "Flight not found"})


def qa_agent_respond(user_query, use_llm=False):
    """QA Agent - processes queries and returns JSON response"""
    if use_llm:
        return qa_agent_respond_with_llm(user_query)
    
    # Try to find flight number in the query
    pattern = r'(?:flight\s+)?([A-Z]{2}\d{3}|\d{3})'
    match = re.search(pattern, user_query, re.IGNORECASE)
    
    if not match:
        return json.dumps({
            "answer": "Could not identify a flight number in your query. Please provide a valid flight number."
        })
    
    flight_num = match.group(1)
    
    # Add AI prefix if just numbers
    if flight_num.isdigit():
        flight_num = f"AI{flight_num}"
    else:
        flight_num = flight_num.upper()
    
    # Get flight info from Info Agent
    info_response = info_agent_request(flight_num)
    flight_data = json.loads(info_response)
    
    if "error" in flight_data:
        return json.dumps({
            "answer": f"Flight {flight_num} not found in database."
        })
    
    # Build response message
    answer = (
        f"Flight {flight_data['flight_number']} departs at {flight_data['departure_time']} "
        f"to {flight_data['destination']}. Current status: {flight_data['status']}."
    )
    
    return json.dumps({"answer": answer})


def qa_agent_respond_with_llm(user_query):
    """QA Agent with LLM support for better understanding"""
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key or api_key == 'your_groq_api_key_here':
            return qa_agent_respond(user_query, use_llm=False)
        
        client = Groq(api_key=api_key)
        
        # Ask LLM to extract flight number
        extraction_prompt = f"""Extract the flight number from this query. 
If you find a flight number, respond with ONLY the flight number (e.g., AI123).
If no flight number is found, respond with "NONE".

Query: {user_query}

Flight number:"""
        
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        flight_num = resp.choices[0].message.content.strip().upper()
        
        if flight_num == "NONE" or not flight_num:
            return json.dumps({
                "answer": "Could not identify a flight number in your query. Please provide a valid flight number."
            })
        
        # Clean it up
        flight_num = re.sub(r'[^A-Z0-9]', '', flight_num)
        if flight_num.isdigit():
            flight_num = f"AI{flight_num}"
        
        # Get the flight info
        info_response = info_agent_request(flight_num)
        flight_data = json.loads(info_response)
        
        if "error" in flight_data:
            return json.dumps({
                "answer": f"Flight {flight_num} not found in database."
            })
        
        # Generate a nice response using LLM
        response_prompt = f"""You are a helpful airline customer service agent. 
Based on this flight information, answer the user's query in a friendly, natural way.

Flight Information:
{json.dumps(flight_data, indent=2)}

User Query: {user_query}

Provide a clear, concise answer (1-2 sentences max):"""
        
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": response_prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        answer = resp.choices[0].message.content.strip()
        return json.dumps({"answer": answer})
        
    except Exception as e:
        print(f"LLM error: {e}. Falling back to regex.")
        return qa_agent_respond(user_query, use_llm=False)


def test_system():
    """Run tests for the agent system"""
    print("=" * 80)
    print("Testing Two-Agent System")
    print("=" * 80)
    
    api_key = os.getenv('GROQ_API_KEY')
    use_llm = api_key and api_key != 'your_groq_api_key_here'
    
    if use_llm:
        print("\nUsing LLM mode (Groq API detected)")
    else:
        print("\nUsing regex mode (no API key found)")
    
    # Test 1
    print("\n1. Testing get_flight_info('AI123'):")
    result = get_flight_info("AI123")
    print(json.dumps(result, indent=2))
    
    # Test 2
    print("\n2. Testing info_agent_request('AI123'):")
    result = info_agent_request("AI123")
    print(result)
    
    # Test 3
    print("\n3. Testing qa_agent_respond('When does Flight AI123 depart?'):")
    result = qa_agent_respond("When does Flight AI123 depart?", use_llm=use_llm)
    print(result)
    data = json.loads(result)
    print(f"Answer: {data['answer']}")
    
    # Test 4
    print("\n4. Testing qa_agent_respond('What is the status of Flight AI999?'):")
    result = qa_agent_respond("What is the status of Flight AI999?", use_llm=use_llm)
    print(result)
    data = json.loads(result)
    print(f"Answer: {data['answer']}")
    
    # Test 5
    print("\n5. Testing qa_agent_respond('Tell me about flight AI456'):")
    result = qa_agent_respond("Tell me about flight AI456", use_llm=use_llm)
    print(result)
    data = json.loads(result)
    print(f"Answer: {data['answer']}")
    
    # Test 6
    print("\n6. Testing qa_agent_respond('Is AI789 on time?'):")
    result = qa_agent_respond("Is AI789 on time?", use_llm=use_llm)
    print(result)
    data = json.loads(result)
    print(f"Answer: {data['answer']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_system()
