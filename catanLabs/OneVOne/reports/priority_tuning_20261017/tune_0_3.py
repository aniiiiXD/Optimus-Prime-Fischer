from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hackathon'))
from priority_policy import Agent as PriorityAgent

class Agent(PriorityAgent):

    def __init__(self):
        super().__init__(priorities=[1.502592365023513, 0.9632729925231673, 0.7715077854579828, 0.7762279274410565, 0.9863989295542802])
