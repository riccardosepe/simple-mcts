# Simple implementation of MCTS


# NB: a node has to retain the value as it is seen by its father. This means that in a node in which the agent has to take the decisions, its children, which represent human choices, must retain the values in the perspective of the father. A child node that leads to all human wins, even if it is a human node, must have a positive value.