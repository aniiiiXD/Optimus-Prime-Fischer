"""Frozen scorer variant for the official evaluator."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent import Agent as CoreAgent

class Agent(CoreAgent):
    def __init__(self):
        super().__init__(Path(__file__).with_name("config.json"))
