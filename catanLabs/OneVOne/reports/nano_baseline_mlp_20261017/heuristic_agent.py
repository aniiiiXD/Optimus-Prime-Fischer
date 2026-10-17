from pathlib import Path
import sys
sys.path.insert(0, str((Path(__file__).parent / '../../hackathon').resolve()))
from learning_agent import Agent as LearningAgent
class Agent(LearningAgent):
    def __init__(self):
        super().__init__(config_path=str(Path(__file__).parent / 'frozen_config.json'), model=None, trace_path=str(Path(__file__).parent / 'traces' / 'heuristic'))
