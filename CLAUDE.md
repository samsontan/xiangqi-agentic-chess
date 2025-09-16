# Xiangqi (Chinese Chess) Game Project

## Overview
A complete Chinese Chess (Xiangqi) implementation developed using agentic swarm coding techniques on Android/Termux. The project evolved from exploring agentic frameworks to creating a fully functional game with AI opponent.

## Project Evolution

### Initial Goals
- Explore agentic swarm coding systems compatible with Android/Termux
- Install and test claude-flow (failed due to native dependencies)
- Create alternative pure Python agentic framework
- Use multi-agent approach to develop Chinese Chess game

### Key Challenges Solved
1. **Native Dependencies Issue**: claude-flow requires Android NDK for compilation
   - Solution: Created pure Python agentic framework `local_agent_swarm.py`

2. **AI Performance**: Initial minimax AI was too slow (getting stuck in calculations)
   - Solution: Created time-limited AI with 2-second thinking limit
   - Final solution: Proper move validation to prevent illegal moves

3. **Board Alignment**: Major challenge with visual alignment of pieces and labels
   - Issue: Chinese characters are double-width, causing misalignment
   - Solution: Bracketed file labels [a][b][c] with proper spacing offset

4. **Illegal AI Moves**: AI was making invalid moves (e.g., horse moving horizontally)
   - Issue: Simplified validation `return True` for speed
   - Solution: Implemented proper Xiangqi move validation rules

## Final Implementation

### Game Files
- **`xiangqi_fixed.py`** - Main game with proper move validation and perfect alignment
- **`xiangqi_perfect_final.py`** - Perfect board alignment with circles and bracketed labels
- **`xiangqi_fast.py`** - Base game engine with time-limited AI
- **`local_agent_swarm.py`** - Pure Python agentic framework

### Features
- ✅ Perfect visual alignment with bracketed file labels [a]-[i]
- ✅ Proper Xiangqi move validation for all piece types
- ✅ Fast AI opponent (2-second thinking limit)
- ✅ Beautiful Chinese characters with color coding
- ✅ Stable game logic that handles invalid moves gracefully
- ✅ Familiar move notation (e4-e5 format)

### AI Move Validation Rules
- **Horse (馬)**: L-shape movement only (2+1 or 1+2 squares)
- **Chariot (車)**: Horizontal or vertical movement
- **Cannon (炮)**: Like chariot, needs platform piece to capture
- **Soldier (兵/卒)**: Forward only, sideways after crossing river
- **General (帥/將)**: One step within palace
- **Advisor (仕/士)**: Diagonal movement within palace
- **Elephant (相/象)**: Diagonal 2 points, cannot cross river

## Technical Achievements

### Agentic Swarm Framework
```python
class SmartAgent:
    def __init__(self, name: str, provider: LLMProvider, role: str = "assistant"):
        # Multi-agent coordination system
```

### Perfect Board Alignment
```python
# Accounts for rank number spacing (2 chars + 1 space)
print("   [a][b][c][d][e][f][g][h][i]")
```

### Move Validation Example
```python
def is_valid_move(self, start, end):
    if piece_type == 'horse':
        # Horse moves in L-shape only
        if not ((abs(dr) == 2 and abs(dc) == 1) or (abs(dr) == 1 and abs(dc) == 2)):
            return False
```

## Commands to Run

### Play the Game
```bash
python xiangqi_fixed.py
```

### Test Board Alignment
```bash
python test_board_alignment.py
```

### Debug AI Moves
```bash
python debug_ai_move.py
```

## Development Process
1. **Multi-agent architecture**: Used architect, developer, AI specialist, UI developer, tester agents
2. **Iterative improvement**: Started with complex AI, simplified for performance
3. **User feedback integration**: Fixed alignment and validation issues based on testing
4. **Progressive enhancement**: Added features while maintaining stability

## Key Learnings
- Agentic swarm coding effective for complex game development
- Terminal alignment requires careful consideration of character widths
- Game AI needs proper validation to prevent illegal moves
- User testing crucial for catching edge cases and visual issues

## Files Structure
```
├── xiangqi_fixed.py              # Final game with all fixes
├── xiangqi_perfect_final.py      # Perfect alignment version
├── xiangqi_fast.py              # Base fast AI engine
├── local_agent_swarm.py         # Agentic framework
├── debug_ai_move.py             # AI debugging tools
├── test_board_alignment.py      # Alignment testing
└── CLAUDE.md                    # This documentation
```

## Success Metrics
- ✅ Fully functional Chinese Chess game
- ✅ Legal AI moves only
- ✅ Perfect visual alignment
- ✅ Fast, responsive gameplay
- ✅ Stable, crash-free experience
- ✅ Pure Python, Termux-compatible

This project demonstrates successful use of agentic swarm coding to create a complete, polished game from scratch, overcoming multiple technical challenges through iterative development and user feedback.