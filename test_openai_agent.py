#!/usr/bin/env python3
"""
Test OpenAI Agent with your API key
"""

import os
from local_agent_swarm import OpenAIProvider, SmartAgent, AgentSwarm

def test_single_agent():
    """Test a single OpenAI agent"""
    print("Testing Single OpenAI Agent")
    print("=" * 30)

    # Create OpenAI provider
    provider = OpenAIProvider()

    # Create agent
    coder = SmartAgent("GPT-Coder", provider, "Python developer")
    coder.add_capability("coding")
    coder.add_capability("debugging")

    # Test the agent
    prompt = "Write a simple Python function to calculate the factorial of a number"
    print(f"Prompt: {prompt}")
    print("\nAgent response:")

    response = coder.think(prompt)
    print(response)

    return coder

def test_swarm():
    """Test multi-agent swarm with OpenAI"""
    print("\n" + "=" * 50)
    print("Testing Multi-Agent Swarm")
    print("=" * 50)

    # Create swarm
    swarm = AgentSwarm()

    # Create multiple OpenAI agents with different roles
    provider = OpenAIProvider()

    # Coder agent
    coder = SmartAgent("Coder", provider, "Python developer")
    coder.add_capability("coding")
    swarm.add_agent(coder)

    # Reviewer agent
    reviewer = SmartAgent("Reviewer", provider, "senior code reviewer")
    reviewer.add_capability("review")
    swarm.add_agent(reviewer)

    # Architect agent
    architect = SmartAgent("Architect", provider, "software architect")
    architect.add_capability("planning")
    architect.add_capability("architecture")
    swarm.add_agent(architect)

    print(f"Created swarm with {len(swarm.agents)} agents")

    # Create tasks
    tasks = [
        ("Design a REST API for a todo application", "architecture"),
        ("Write a Python function to validate email addresses", "coding"),
        ("Review the email validation function for security", "review")
    ]

    for desc, capability in tasks:
        swarm.create_task(desc, capability)

    print(f"Created {len(swarm.tasks)} tasks")

    # Execute tasks
    for i, task in enumerate(swarm.tasks):
        print(f"\n--- Task {i+1}: {task.description} ---")
        print(f"Assigned to: {task.assigned_to}")

        result = swarm.execute_task(task)
        print(f"Result preview: {result[:200]}...")
        print(f"Status: {task.status}")

    return swarm

def test_conversation():
    """Test multi-turn conversation"""
    print("\n" + "=" * 50)
    print("Testing Multi-turn Conversation")
    print("=" * 50)

    provider = OpenAIProvider()
    assistant = SmartAgent("Assistant", provider, "helpful coding assistant")

    # Multi-turn conversation
    conversations = [
        "What is a Python decorator?",
        "Can you show me an example of a decorator?",
        "How would I use that decorator with a function?",
        "What are some common use cases for decorators?"
    ]

    for i, prompt in enumerate(conversations):
        print(f"\nTurn {i+1}: {prompt}")
        response = assistant.think(prompt)
        print(f"Response: {response[:150]}...")

def main():
    """Run all tests"""
    print("OpenAI Agent Framework Test")
    print("API Key:", os.getenv('OPENAI_API_KEY', 'Not set')[:20] + "..." if os.getenv('OPENAI_API_KEY') else "Not set")
    print()

    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not set!")
        return

    try:
        # Test single agent
        agent = test_single_agent()

        # Test swarm
        swarm = test_swarm()

        # Test conversation
        test_conversation()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("The agent framework is working with your OpenAI API key.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()