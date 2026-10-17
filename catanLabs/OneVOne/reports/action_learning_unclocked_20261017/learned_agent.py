from pathlib import Path
import sys
sys.path.insert(0, str((Path(__file__).parent / '../../hackathon').resolve()))
from learning_agent import Agent as LearningAgent
class Agent(LearningAgent):
    def __init__(self):
        super().__init__(model=str(Path(__file__).parent / 'action_model.json'), trace_path=str(Path(__file__).parent / 'traces' / 'learned'))
