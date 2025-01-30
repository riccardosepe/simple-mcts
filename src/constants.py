import os

PROJECT_NAME = "simple-mcts"
ENTITY_NAME = "riccardos"
PROJECT_PATH = os.getcwd()

while os.path.basename(PROJECT_PATH) != PROJECT_NAME:
    PROJECT_PATH = os.path.dirname(PROJECT_PATH)
    if PROJECT_PATH == '/':
        raise RuntimeError("Project folder not found")

