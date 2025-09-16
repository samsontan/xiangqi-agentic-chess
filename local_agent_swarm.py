#!/usr/bin/env python3
"""
Local Agent Swarm Framework for Termux/Android
Supports multiple LLM providers including local models via Ollama
"""

import json
import requests
import os
import subprocess
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class AgentMessage:
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    description: str
    assigned_to: str = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama provider"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def chat(self, messages: List[Dict], **kwargs) -> str:
        try:
            data = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "No response")

        except Exception as e:
            return f"Ollama Error: {str(e)}"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model

    def chat(self, messages: List[Dict], **kwargs) -> str:
        if not self.api_key:
            return "Error: No OpenAI API key provided"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except Exception as e:
            return f"OpenAI Error: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model

    def chat(self, messages: List[Dict], **kwargs) -> str:
        if not self.api_key:
            return "Error: No Anthropic API key provided"

        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }

            # Convert messages format for Anthropic
            system_msg = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)

            data = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "messages": user_messages
            }

            if system_msg:
                data["system"] = system_msg

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result['content'][0]['text']

        except Exception as e:
            return f"Anthropic Error: {str(e)}"


class SmartAgent:
    """Enhanced AI Agent with multiple provider support"""

    def __init__(self, name: str, provider: LLMProvider, role: str = "assistant"):
        self.name = name
        self.provider = provider
        self.role = role
        self.conversation_history = []
        self.memory = {}
        self.capabilities = set()

    def add_capability(self, capability: str):
        """Add a capability to this agent"""
        self.capabilities.add(capability)

    def remember(self, key: str, value: Any):
        """Store information in agent memory"""
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        """Retrieve information from agent memory"""
        return self.memory.get(key)

    def think(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Process a prompt and return response"""
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif self.role:
            messages.append({"role": "system", "content": f"You are a {self.role} named {self.name}."})

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Get response from provider
        response = self.provider.chat(messages, **kwargs)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response

    def can_do(self, capability: str) -> bool:
        """Check if agent has a specific capability"""
        return capability in self.capabilities


class AgentSwarm:
    """Advanced multi-agent coordination system"""

    def __init__(self):
        self.agents = {}
        self.tasks = []
        self.shared_memory = {}
        self.task_queue = []

    def add_agent(self, agent: SmartAgent):
        """Add agent to swarm"""
        self.agents[agent.name] = agent

    def create_task(self, description: str, required_capability: str = None) -> AgentTask:
        """Create a new task"""
        task = AgentTask(description=description)

        # Auto-assign to capable agent
        if required_capability:
            for name, agent in self.agents.items():
                if agent.can_do(required_capability):
                    task.assigned_to = name
                    break

        self.tasks.append(task)
        return task

    def assign_task(self, task: AgentTask, agent_name: str):
        """Manually assign task to specific agent"""
        if agent_name in self.agents:
            task.assigned_to = agent_name

    def execute_task(self, task: AgentTask) -> str:
        """Execute a single task"""
        if not task.assigned_to or task.assigned_to not in self.agents:
            return "Error: No suitable agent assigned"

        agent = self.agents[task.assigned_to]
        task.status = "in_progress"

        try:
            # Include shared context
            context = f"Task: {task.description}\n"
            if self.shared_memory:
                context += f"Shared context: {json.dumps(self.shared_memory, indent=2)}\n"

            result = agent.think(context)
            task.result = result
            task.status = "completed"

            # Update shared memory with key insights
            self.shared_memory[f"task_{len(self.tasks)}_result"] = result[:200]

            return result

        except Exception as e:
            task.status = "failed"
            task.result = f"Task failed: {str(e)}"
            return task.result

    def coordinate(self, objective: str) -> Dict[str, str]:
        """Coordinate multiple agents toward an objective"""
        # Break down objective into tasks
        planner = self.agents.get("planner")
        if planner:
            plan_prompt = f"Break down this objective into specific tasks: {objective}"
            plan = planner.think(plan_prompt)
            self.shared_memory["current_plan"] = plan

        # Execute tasks
        results = {}
        for task in self.tasks:
            if task.status == "pending":
                result = self.execute_task(task)
                results[task.assigned_to or "unassigned"] = result

        return results

    def status_report(self) -> str:
        """Generate status report of all agents and tasks"""
        report = f"Swarm Status Report\n{'='*20}\n"
        report += f"Agents: {len(self.agents)}\n"
        report += f"Tasks: {len(self.tasks)}\n\n"

        for name, agent in self.agents.items():
            report += f"Agent: {name}\n"
            report += f"  Role: {agent.role}\n"
            report += f"  Capabilities: {', '.join(agent.capabilities)}\n"
            report += f"  Memory items: {len(agent.memory)}\n\n"

        for i, task in enumerate(self.tasks):
            report += f"Task {i+1}: {task.description[:50]}...\n"
            report += f"  Status: {task.status}\n"
            report += f"  Assigned to: {task.assigned_to}\n\n"

        return report


def setup_demo_swarm():
    """Set up a demo swarm with different providers"""
    swarm = AgentSwarm()

    # Try to create agents with available providers
    providers_tried = []

    # Try Ollama (local)
    try:
        ollama = OllamaProvider("llama2")
        coder = SmartAgent("LocalCoder", ollama, "Python developer")
        coder.add_capability("coding")
        coder.add_capability("debugging")
        swarm.add_agent(coder)
        providers_tried.append("Ollama (local)")
    except:
        pass

    # Try OpenAI
    if os.getenv('OPENAI_API_KEY'):
        openai_provider = OpenAIProvider()
        architect = SmartAgent("Architect", openai_provider, "software architect")
        architect.add_capability("planning")
        architect.add_capability("architecture")
        swarm.add_agent(architect)
        providers_tried.append("OpenAI")

    # Try Anthropic
    if os.getenv('ANTHROPIC_API_KEY'):
        anthropic_provider = AnthropicProvider()
        reviewer = SmartAgent("Reviewer", anthropic_provider, "code reviewer")
        reviewer.add_capability("review")
        reviewer.add_capability("testing")
        swarm.add_agent(reviewer)
        providers_tried.append("Anthropic")

    return swarm, providers_tried


def main():
    """Main demo function"""
    print("Local Agent Swarm Framework")
    print("=" * 30)

    # Setup swarm
    swarm, providers = setup_demo_swarm()

    print(f"Providers available: {', '.join(providers) if providers else 'None'}")
    print(f"Agents created: {len(swarm.agents)}")

    if not swarm.agents:
        print("\nNo agents available. To use this framework:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Pull a model: ollama pull llama2")
        print("3. Or set API keys: export OPENAI_API_KEY='...' or ANTHROPIC_API_KEY='...'")
        return

    # Demo tasks
    print("\n" + swarm.status_report())

    # Create and execute a task
    task = swarm.create_task("Write a simple Python function to reverse a string", "coding")
    if task.assigned_to:
        print(f"Executing task with {task.assigned_to}...")
        result = swarm.execute_task(task)
        print(f"Result: {result[:200]}...")
    else:
        print("No suitable agent found for coding task")


if __name__ == "__main__":
    main()