from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from priority_policy import Agent as PriorityAgent

class Agent(PriorityAgent):

    def __init__(self):
        super().__init__(priorities=[1.0855982403668598, 1.40265572948628, 1.03097389452267, 0.6693633007505899, 0.8114088348735998])
