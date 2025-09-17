#!/usr/bin/env python3
"""
Chinese Chess (Xiangqi) Game Development using Agentic Swarm
"""

import os
from local_agent_swarm import OpenAIProvider, SmartAgent, AgentSwarm

def create_xiangqi_dev_team():
    """Create specialized development team for Chinese chess game"""
    swarm = AgentSwarm()
    provider = OpenAIProvider(model="gpt-4o-mini")

    # Game Architect
    architect = SmartAgent("GameArchitect", provider, "expert game developer specializing in board games and chess engines")
    architect.add_capability("architecture")
    architect.add_capability("design")
    swarm.add_agent(architect)

    # Core Developer
    core_dev = SmartAgent("CoreDeveloper", provider, "senior Python developer expert in game logic and algorithms")
    core_dev.add_capability("coding")
    core_dev.add_capability("algorithms")
    swarm.add_agent(core_dev)

    # AI Specialist
    ai_specialist = SmartAgent("AISpecialist", provider, "AI expert specializing in game AI, minimax, and evaluation functions")
    ai_specialist.add_capability("ai")
    ai_specialist.add_capability("minimax")
    swarm.add_agent(ai_specialist)

    # UI Developer
    ui_dev = SmartAgent("UIDeveloper", provider, "frontend developer expert in Python GUI frameworks like tkinter")
    ui_dev.add_capability("ui")
    ui_dev.add_capability("interface")
    swarm.add_agent(ui_dev)

    # QA Tester
    tester = SmartAgent("GameTester", provider, "QA engineer specializing in game testing and validation")
    tester.add_capability("testing")
    tester.add_capability("debugging")
    swarm.add_agent(tester)

    return swarm

def develop_xiangqi_game():
    """Coordinate the development of Chinese chess game"""
    print("🎮 Developing Chinese Chess (Xiangqi) Game Engine")
    print("=" * 60)

    swarm = create_xiangqi_dev_team()

    # Development tasks in order
    development_tasks = [
        ("Design the overall architecture for a Chinese chess game with AI opponent, including board representation, piece types, and game flow", "architecture"),
        ("Implement the core game board, piece classes, and basic movement logic for all Xiangqi pieces", "coding"),
        ("Create comprehensive move validation, game rules enforcement, and win condition detection", "coding"),
        ("Develop an AI engine using minimax algorithm with alpha-beta pruning and position evaluation", "ai"),
        ("Build a user-friendly interface using tkinter for playing against the AI", "ui"),
        ("Create comprehensive tests for game logic, AI behavior, and edge cases", "testing")
    ]

    print(f"📋 Creating {len(development_tasks)} development tasks...")

    # Execute development tasks
    code_files = {}

    for i, (description, capability) in enumerate(development_tasks):
        print(f"\n{'='*80}")
        print(f"🔧 TASK {i+1}: {description[:80]}...")
        print(f"{'='*80}")

        task = swarm.create_task(description, capability)
        print(f"👤 Assigned to: {task.assigned_to}")

        # Add context from previous tasks
        if code_files:
            context = f"\nPrevious development context:\n"
            for filename, content in code_files.items():
                context += f"File: {filename}\n{content[:300]}...\n\n"
            task.description += context

        # Execute the task
        result = swarm.execute_task(task)

        # Save significant code output
        if "```python" in result and len(result) > 200:
            filename = f"task_{i+1}_{task.assigned_to.lower()}.py"
            code_files[filename] = result

        # Show result
        if len(result) > 1000:
            print(f"📝 Result preview:\n{result[:800]}...")
            print(f"\n[Full result: {len(result)} characters]")
        else:
            print(f"📝 Result:\n{result}")

        print(f"✅ Status: {task.status}")

    return swarm, code_files

def extract_and_save_code(code_files):
    """Extract Python code from agent responses and save to files"""
    print(f"\n🔧 Extracting and consolidating code...")

    all_code = ""

    for filename, content in code_files.items():
        print(f"Processing: {filename}")

        # Extract Python code blocks
        lines = content.split('\n')
        in_code_block = False
        extracted_code = []

        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                continue
            elif in_code_block:
                extracted_code.append(line)

        if extracted_code:
            code_content = '\n'.join(extracted_code)
            all_code += f"\n# === {filename} ===\n"
            all_code += code_content + "\n"

    return all_code

def create_integrated_game():
    """Create a final integrated game file"""
    print(f"\n🎯 Creating integrated Chinese chess game...")

    swarm = create_xiangqi_dev_team()

    integration_prompt = """
    Create a complete, playable Chinese Chess (Xiangqi) game in Python with the following requirements:

    1. **Board Representation**: 9x10 board with proper Chinese chess layout
    2. **Pieces**: All 7 types - General(King), Advisor, Elephant, Horse, Chariot(Rook), Cannon, Soldier(Pawn)
    3. **Movement Rules**: Implement correct Xiangqi movement for each piece type
    4. **Game Rules**: River crossing, palace restrictions, flying general rule
    5. **AI Opponent**: Minimax with alpha-beta pruning (depth 3-4)
    6. **User Interface**: Text-based interface that's easy to use in terminal
    7. **Game Flow**: Turn management, move input, game state display
    8. **Win Conditions**: Checkmate, stalemate detection

    Make it a single, complete Python file that runs immediately and allows human vs AI gameplay.
    Use clear piece symbols and intuitive coordinate system (like a1-i10).
    Include helpful prompts and move format examples.
    """

    # Use the core developer for integration
    task = swarm.create_task(integration_prompt, "coding")
    result = swarm.execute_task(task)

    return result

def main():
    """Main development orchestration"""
    print("🤖 Agentic Swarm Development: Chinese Chess Engine")
    print("=" * 70)

    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        return

    try:
        # Phase 1: Collaborative development
        swarm, code_files = develop_xiangqi_game()

        # Phase 2: Extract code
        extracted_code = extract_and_save_code(code_files)

        # Phase 3: Create integrated game
        print(f"\n{'='*70}")
        print("🎮 Creating Final Integrated Game")
        print(f"{'='*70}")

        final_game = create_integrated_game()

        # Save the final game
        with open('/data/data/com.termux/files/home/xiangqi_game.py', 'w') as f:
            # Extract code from the final response
            lines = final_game.split('\n')
            in_code_block = False

            for line in lines:
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    in_code_block = False
                    continue
                elif in_code_block:
                    f.write(line + '\n')

        print("\n🎉 Chinese Chess game development completed!")
        print("📁 Game saved as: xiangqi_game.py")
        print("🚀 Run with: python xiangqi_game.py")

        # Show preview
        print(f"\n📝 Game preview:")
        with open('/data/data/com.termux/files/home/xiangqi_game.py', 'r') as f:
            preview = f.read()[:1000]
            print(preview + "...")

    except Exception as e:
        print(f"❌ Error during development: {e}")

if __name__ == "__main__":
    main()