#!/usr/bin/env python3
"""
Simple AI Agent Framework for Termux/Android
A minimal pure Python implementation for agentic coding
"""

import json
import requests
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentResponse:
    content: str
    role: str = "assistant"
    metadata: Dict[str, Any] = None


class SimpleAgent:
    """Minimal AI Agent using HTTP API calls"""

    def __init__(self, name: str, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.name = name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.conversation_history = []
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def call_api(self, prompt: str, system_prompt: str = None) -> AgentResponse:
        """Make API call to LLM"""
        if not self.api_key:
            return AgentResponse("Error: No API key provided", metadata={"error": True})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Update conversation history
            self.add_message("user", prompt)
            self.add_message("assistant", content)

            return AgentResponse(content=content, metadata={"tokens": result.get('usage', {})})

        except requests.exceptions.RequestException as e:
            return AgentResponse(f"API Error: {str(e)}", metadata={"error": True})
        except Exception as e:
            return AgentResponse(f"Unexpected Error: {str(e)}", metadata={"error": True})


class AgentSwarm:
    """Simple multi-agent orchestration"""

    def __init__(self):
        self.agents = {}
        self.shared_context = {}

    def add_agent(self, agent: SimpleAgent):
        """Add agent to swarm"""
        self.agents[agent.name] = agent

    def coordinate(self, task: str, agent_names: List[str] = None) -> Dict[str, AgentResponse]:
        """Coordinate multiple agents on a task"""
        if agent_names is None:
            agent_names = list(self.agents.keys())

        results = {}

        for name in agent_names:
            if name in self.agents:
                agent = self.agents[name]
                system_prompt = f"You are {name}. Work on this task: {task}"

                # Include shared context
                context_str = ""
                if self.shared_context:
                    context_str = f"\nShared context: {json.dumps(self.shared_context, indent=2)}"

                response = agent.call_api(task + context_str, system_prompt)
                results[name] = response

                # Update shared context with results
                if not response.metadata or not response.metadata.get("error"):
                    self.shared_context[f"{name}_result"] = response.content[:200]  # First 200 chars

        return results


def demo():
    """Demo the simple agent framework"""
    print("Simple AI Agent Framework Demo")
    print("=" * 40)

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: No OPENAI_API_KEY found. Set it with:")
        print("export OPENAI_API_KEY='your-key-here'")
        return

    # Create agents
    coder = SimpleAgent("Coder", api_key)
    reviewer = SimpleAgent("Reviewer", api_key)

    # Create swarm
    swarm = AgentSwarm()
    swarm.add_agent(coder)
    swarm.add_agent(reviewer)

    # Test single agent
    print("\n1. Single Agent Test:")
    response = coder.call_api("Write a simple Python function to calculate fibonacci numbers")
    print(f"Coder: {response.content[:100]}...")

    # Test swarm coordination
    print("\n2. Swarm Coordination Test:")
    results = swarm.coordinate("Create a simple Python web server", ["Coder", "Reviewer"])

    for agent_name, response in results.items():
        print(f"\n{agent_name}:")
        print(f"Response: {response.content[:150]}...")
        if response.metadata:
            print(f"Metadata: {response.metadata}")


if __name__ == "__main__":
    demo()