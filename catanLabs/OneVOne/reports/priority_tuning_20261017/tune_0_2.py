from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from priority_policy import Agent as PriorityAgent

class Agent(PriorityAgent):

    def __init__(self):
        super().__init__(priorities=[1.5624476842210275, 1.5033568253029552, 0.5858535010941827, 0.6454948771486329, 0.7028471122332021])
