from torch.utils.hipify.hipify_python import value

from src.evaluators.frozen_lake_evaluator import FrozenLakeEvaluator
from src.tree.chance_tree import ChoiceNode


class FrozenLakeExplainer:
    @staticmethod
    def explain(tree, action):
        # TODO: env?
        subtrees = FrozenLakeExplainer.find_subtrees(tree)

        # 1. safety <= 50%
        values_low_safety = {}
        for a, subtree in subtrees.items():
            values_low_safety[a] = FrozenLakeExplainer.low_safety(subtree)
        # 2. safety > 50%
        # 3. distance <= 50%
        # 4. distance > 50%
        # 5. goal reached
        # 6. goal not reached
        # 7. fell in hole
        # 8. didn't fall in hole
        print()

    @staticmethod
    def low_safety(nodes):
        low_safety_nodes = filter(lambda x: x[0] <= 0.5, map(lambda n: (n.features['safe'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], low_safety_nodes))

    @staticmethod
    def find_subtrees(tree):
        subtrees = dict.fromkeys(tree.root.children, set())
        for a, node in tree.root.children.items():
            subtree = set()
            if node is None:
                continue
            FrozenLakeExplainer._dfs(node, subtree)
            subtrees[a] = subtree

        return subtrees

    @staticmethod
    def _dfs(node, subtree):
        if isinstance(node, ChoiceNode):
            assert node is not None
            subtree.add(node)
        for child in node.children.values():
            if child is None:
                continue
            FrozenLakeExplainer._dfs(child, subtree)
