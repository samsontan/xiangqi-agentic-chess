#!/usr/bin/env python3
"""
Coding Agent Demo - Real agentic coding with your OpenAI API
"""

import os
from local_agent_swarm import OpenAIProvider, SmartAgent, AgentSwarm

def create_coding_swarm():
    """Create a specialized coding swarm"""
    swarm = AgentSwarm()
    provider = OpenAIProvider(model="gpt-4o-mini")  # Use faster, cheaper model

    # Senior Developer
    senior_dev = SmartAgent("SeniorDev", provider, "senior Python developer with 10+ years experience")
    senior_dev.add_capability("coding")
    senior_dev.add_capability("architecture")
    senior_dev.add_capability("mentoring")
    swarm.add_agent(senior_dev)

    # Code Reviewer
    reviewer = SmartAgent("Reviewer", provider, "meticulous code reviewer focused on best practices")
    reviewer.add_capability("review")
    reviewer.add_capability("security")
    reviewer.add_capability("performance")
    swarm.add_agent(reviewer)

    # Documentation Expert
    doc_expert = SmartAgent("DocExpert", provider, "technical writer specializing in code documentation")
    doc_expert.add_capability("documentation")
    doc_expert.add_capability("examples")
    swarm.add_agent(doc_expert)

    # Tester
    tester = SmartAgent("Tester", provider, "QA engineer focused on comprehensive testing")
    tester.add_capability("testing")
    tester.add_capability("debugging")
    swarm.add_agent(tester)

    return swarm

def demo_web_api_project():
    """Demo: Build a complete web API project"""
    print("🚀 Agentic Coding Demo: Building a Web API")
    print("=" * 50)

    swarm = create_coding_swarm()

    # Project tasks in order
    project_tasks = [
        ("Design a Flask REST API for a book library system with CRUD operations", "architecture"),
        ("Implement the Flask application with proper error handling and validation", "coding"),
        ("Review the code for security vulnerabilities and best practices", "review"),
        ("Create comprehensive unit tests for the API endpoints", "testing"),
        ("Write detailed API documentation with usage examples", "documentation")
    ]

    print(f"Creating {len(project_tasks)} project tasks...")

    # Create and execute tasks
    for i, (description, capability) in enumerate(project_tasks):
        print(f"\n{'='*60}")
        print(f"TASK {i+1}: {description}")
        print(f"{'='*60}")

        task = swarm.create_task(description, capability)
        print(f"📋 Assigned to: {task.assigned_to}")

        # Execute the task
        result = swarm.execute_task(task)

        # Show relevant portion of the result
        if len(result) > 500:
            print(f"📄 Result preview:\n{result[:500]}...")
            print(f"\n[Full result: {len(result)} characters]")
        else:
            print(f"📄 Result:\n{result}")

        print(f"✅ Status: {task.status}")

    return swarm

def demo_algorithm_challenge():
    """Demo: Solve an algorithmic challenge"""
    print("\n🧠 Algorithm Challenge Demo")
    print("=" * 50)

    provider = OpenAIProvider(model="gpt-4o-mini")
    coder = SmartAgent("AlgorithmExpert", provider, "competitive programming expert")

    challenge = """
    Problem: Find the longest common subsequence (LCS) between two strings.

    Example:
    - String 1: "ABCDGH"
    - String 2: "AEDFHR"
    - LCS: "ADH" (length = 3)

    Requirements:
    1. Implement efficient dynamic programming solution
    2. Include time/space complexity analysis
    3. Add test cases
    4. Explain the algorithm
    """

    print("🎯 Challenge:")
    print(challenge)
    print("\n🤖 Agent solving...")

    response = coder.think(challenge)
    print(f"\n💡 Solution:\n{response}")

def demo_code_refactoring():
    """Demo: Refactor messy code"""
    print("\n🔧 Code Refactoring Demo")
    print("=" * 50)

    messy_code = '''
def calc(x,y,op):
    if op=="add":
        return x+y
    elif op=="sub":
        return x-y
    elif op=="mul":
        return x*y
    elif op=="div":
        if y!=0:
            return x/y
        else:
            return "Error"
    else:
        return "Invalid"

def process_list(lst):
    result=[]
    for i in range(len(lst)):
        if lst[i]%2==0:
            result.append(lst[i]*2)
        else:
            result.append(lst[i]*3)
    return result
'''

    provider = OpenAIProvider(model="gpt-4o-mini")
    refactor_agent = SmartAgent("Refactorer", provider, "expert at clean code and refactoring")

    prompt = f"""
    Please refactor this messy Python code to follow best practices:

    {messy_code}

    Requirements:
    1. Add proper type hints
    2. Improve naming and structure
    3. Add docstrings
    4. Handle errors properly
    5. Make it more Pythonic
    6. Add example usage
    """

    print("🗑️ Original messy code:")
    print(messy_code)
    print("\n🤖 Agent refactoring...")

    refactored = refactor_agent.think(prompt)
    print(f"\n✨ Refactored code:\n{refactored}")

def main():
    """Run coding demos"""
    print("🤖 Agentic Swarm Coding Demos")
    print("=" * 60)
    print(f"Using OpenAI API: {os.getenv('OPENAI_API_KEY', 'Not set')[:20]}...")

    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        return

    try:
        # Demo 1: Complete web API project
        demo_web_api_project()

        # Demo 2: Algorithm challenge
        demo_algorithm_challenge()

        # Demo 3: Code refactoring
        demo_code_refactoring()

        print("\n" + "=" * 60)
        print("🎉 All coding demos completed!")
        print("Your agentic coding swarm is ready for real projects.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()