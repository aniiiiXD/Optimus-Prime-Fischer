from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from priority_policy import Agent as PriorityAgent

class Agent(PriorityAgent):

    def __init__(self):
        super().__init__(priorities=[1.25, 1.25, 0.8, 0.95, 0.75])
