"""Shared constants for imitation-learning code."""

ACTION_FORWARD = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
NUM_ACTIONS = 3

ACTION_ID_TO_NAME = {
    ACTION_FORWARD: "FORWARD",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}

ACTION_NAME_TO_ID = {name: idx for idx, name in ACTION_ID_TO_NAME.items()}
