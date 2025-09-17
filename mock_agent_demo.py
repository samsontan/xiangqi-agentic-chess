#!/usr/bin/env python3
"""
Mock Agent Demo - Working example with simulated responses
"""

from local_agent_swarm import SmartAgent, AgentSwarm, LLMProvider
from typing import List, Dict


class MockProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def __init__(self, agent_name: str = "MockAgent"):
        self.agent_name = agent_name
        self.responses = {
            "coding": "Here's a simple Python function:\n\ndef reverse_string(s):\n    return s[::-1]\n\n# Example usage:\nprint(reverse_string('hello'))  # Output: olleh",
            "planning": "I'll break this down into steps:\n1. Analyze requirements\n2. Design architecture\n3. Implement core features\n4. Test and debug\n5. Deploy and monitor",
            "review": "Code review feedback:\n✓ Good: Clean function structure\n⚠ Consider: Add input validation\n⚠ Consider: Add type hints\n✓ Good: Clear variable names",
            "architecture": "Recommended architecture:\n- Use MVC pattern\n- Implement dependency injection\n- Add logging layer\n- Use configuration management\n- Include error handling",
            "testing": "Test plan:\n1. Unit tests for core functions\n2. Integration tests for API endpoints\n3. Performance testing\n4. Security testing\n5. User acceptance testing"
        }

    def chat(self, messages: List[Dict], **kwargs) -> str:
        # Extract the latest user message
        user_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"].lower()

        # Generate appropriate response based on keywords
        if "reverse" in user_msg or "string" in user_msg or "function" in user_msg:
            return self.responses["coding"]
        elif "plan" in user_msg or "break down" in user_msg or "steps" in user_msg:
            return self.responses["planning"]
        elif "review" in user_msg or "feedback" in user_msg:
            return self.responses["review"]
        elif "architecture" in user_msg or "design" in user_msg:
            return self.responses["architecture"]
        elif "test" in user_msg:
            return self.responses["testing"]
        else:
            return f"Hello! I'm {self.agent_name}. I can help with coding, planning, review, architecture, and testing tasks."


def demo_basic_agents():
    """Demo basic agent functionality"""
    print("=== Basic Agent Demo ===")

    # Create agents with mock providers
    coder = SmartAgent("Alice", MockProvider("Alice"), "Python developer")
    coder.add_capability("coding")
    coder.add_capability("debugging")

    reviewer = SmartAgent("Bob", MockProvider("Bob"), "code reviewer")
    reviewer.add_capability("review")
    reviewer.add_capability("testing")

    # Test individual agents
    print(f"\n{coder.name} (Coder):")
    response = coder.think("Write a function to reverse a string")
    print(response)

    print(f"\n{reviewer.name} (Reviewer):")
    response = reviewer.think("Please review this code for best practices")
    print(response)


def demo_swarm_coordination():
    """Demo swarm coordination"""
    print("\n=== Swarm Coordination Demo ===")

    # Create swarm
    swarm = AgentSwarm()

    # Create specialized agents
    planner = SmartAgent("Planner", MockProvider("Planner"), "project planner")
    planner.add_capability("planning")
    planner.add_capability("architecture")

    coder = SmartAgent("Coder", MockProvider("Coder"), "developer")
    coder.add_capability("coding")
    coder.add_capability("debugging")

    tester = SmartAgent("Tester", MockProvider("Tester"), "quality assurance")
    tester.add_capability("testing")
    tester.add_capability("review")

    # Add to swarm
    swarm.add_agent(planner)
    swarm.add_agent(coder)
    swarm.add_agent(tester)

    print("Swarm created with agents:", list(swarm.agents.keys()))

    # Create tasks
    task1 = swarm.create_task("Plan a web application project", "planning")
    task2 = swarm.create_task("Write a function to reverse a string", "coding")
    task3 = swarm.create_task("Create test cases for the string function", "testing")

    print(f"\nCreated {len(swarm.tasks)} tasks")

    # Execute tasks
    for i, task in enumerate(swarm.tasks):
        print(f"\n--- Task {i+1}: {task.description} ---")
        print(f"Assigned to: {task.assigned_to}")
        result = swarm.execute_task(task)
        print(f"Result: {result[:150]}...")
        print(f"Status: {task.status}")

    # Show final status
    print(f"\n{swarm.status_report()}")


def demo_agent_memory():
    """Demo agent memory and capabilities"""
    print("\n=== Agent Memory Demo ===")

    agent = SmartAgent("Memory_Agent", MockProvider("Memory_Agent"), "smart assistant")
    agent.add_capability("coding")
    agent.add_capability("memory")

    # Store some information
    agent.remember("project_name", "MyWebApp")
    agent.remember("language", "Python")
    agent.remember("framework", "Flask")

    print("Agent memory:")
    for key, value in agent.memory.items():
        print(f"  {key}: {value}")

    print(f"\nAgent capabilities: {agent.capabilities}")
    print(f"Can code: {agent.can_do('coding')}")
    print(f"Can fly: {agent.can_do('flying')}")

    # Test memory in conversation
    response = agent.think(f"I'm working on {agent.recall('project_name')} using {agent.recall('language')}")
    print(f"\nAgent response: {response}")


def demo_advanced_swarm():
    """Demo advanced swarm features"""
    print("\n=== Advanced Swarm Demo ===")

    swarm = AgentSwarm()

    # Create diverse team
    agents_config = [
        ("ProductManager", "planning", "product manager"),
        ("BackendDev", "coding", "backend developer"),
        ("FrontendDev", "coding", "frontend developer"),
        ("DevOps", "deployment", "DevOps engineer"),
        ("QA", "testing", "quality assurance")
    ]

    for name, capability, role in agents_config:
        agent = SmartAgent(name, MockProvider(name), role)
        agent.add_capability(capability)
        swarm.add_agent(agent)

    # Simulate a software development project
    project_tasks = [
        ("Define project requirements", "planning"),
        ("Design system architecture", "planning"),
        ("Implement backend API", "coding"),
        ("Create frontend interface", "coding"),
        ("Set up CI/CD pipeline", "deployment"),
        ("Write test suite", "testing")
    ]

    print(f"Creating {len(project_tasks)} project tasks...")

    for desc, capability in project_tasks:
        swarm.create_task(desc, capability)

    # Execute all tasks
    print("\nExecuting project tasks:")
    for i, task in enumerate(swarm.tasks):
        print(f"\n{i+1}. {task.description}")
        print(f"   Assigned to: {task.assigned_to}")
        swarm.execute_task(task)
        print(f"   Status: {task.status}")

    # Show completion stats
    completed = sum(1 for task in swarm.tasks if task.status == "completed")
    print(f"\nProject completion: {completed}/{len(swarm.tasks)} tasks completed")


def main():
    """Run all demos"""
    print("Agentic Swarm Coding Framework Demo")
    print("=" * 40)

    demo_basic_agents()
    demo_swarm_coordination()
    demo_agent_memory()
    demo_advanced_swarm()

    print("\n" + "=" * 40)
    print("Demo completed! This framework works with:")
    print("• OpenAI API (set OPENAI_API_KEY)")
    print("• Anthropic API (set ANTHROPIC_API_KEY)")
    print("• Local Ollama (install ollama)")
    print("• Mock providers (for testing)")


if __name__ == "__main__":
    main()