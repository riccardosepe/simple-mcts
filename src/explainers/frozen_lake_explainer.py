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
        values_high_safety = {}
        for a, subtree in subtrees.items():
            values_high_safety[a] = FrozenLakeExplainer.high_safety(subtree)

        # 3. proximity <= 50%
        values_low_proximity = {}
        for a, subtree in subtrees.items():
            values_low_proximity[a] = FrozenLakeExplainer.low_proximity(subtree)

        # 4. proximity > 50%
        values_high_proximity = {}
        for a, subtree in subtrees.items():
            values_high_proximity[a] = FrozenLakeExplainer.high_proximity(subtree)

        # 5. goal reached
        values_goal_reached = {}
        for a, subtree in subtrees.items():
            values_goal_reached[a] = FrozenLakeExplainer.goal_reached(subtree)

        # 6. goal not reached
        values_goal_not_reached = {}
        for a, subtree in subtrees.items():
            values_goal_not_reached[a] = FrozenLakeExplainer.goal_not_reached(subtree)

        # 7. fell in hole
        values_hole_fall = {}
        for a, subtree in subtrees.items():
            values_hole_fall[a] = FrozenLakeExplainer.hole_fall(subtree)

        # 8. didn't fall in hole
        values_hole_not_fall = {}
        for a, subtree in subtrees.items():
            values_hole_not_fall[a] = FrozenLakeExplainer.hole_not_fall(subtree)
        print()

    @staticmethod
    def low_safety(nodes):
        low_safety_nodes = filter(lambda x: x[0] <= 0.5, map(lambda n: (n.features['safe'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], low_safety_nodes))

    @staticmethod
    def high_safety(nodes):
        high_safety_nodes = filter(lambda x: x[0] > 0.5, map(lambda n: (n.features['safe'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], high_safety_nodes))

    @staticmethod
    def low_proximity(nodes):
        low_proximity_nodes = filter(lambda x: x[0] <= 0.5, map(lambda n: (n.features['dist'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], low_proximity_nodes))

    @staticmethod
    def high_proximity(nodes):
        high_proximity_nodes = filter(lambda x: x[0] > 0.5, map(lambda n: (n.features['dist'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], high_proximity_nodes))

    @staticmethod
    def goal_reached(nodes):
        goal_reached_nodes = filter(lambda x: x[0], map(lambda n: (n.features['goal'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], goal_reached_nodes))

    @staticmethod
    def goal_not_reached(nodes):
        goal_not_reached_nodes = filter(lambda x: not x[0], map(lambda n: (n.features['goal'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], goal_not_reached_nodes))

    @staticmethod
    def hole_fall(nodes):
        hole_fall_nodes = filter(lambda x: x[0], map(lambda n: (n.features['hole'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], hole_fall_nodes))

    @staticmethod
    def hole_not_fall(nodes):
        hole_not_fall_nodes = filter(lambda x: not x[0], map(lambda n: (n.features['hole'], n.value), nodes))
        return sum(map(lambda n: n[0] * n[1], hole_not_fall_nodes))

    @staticmethod
    def find_subtrees(tree):
        subtrees = dict.fromkeys(tree.root.children, set())
        for a, node in tree.root.children.items():
            subtree = subtrees[a]
            if node is None:
                continue
            FrozenLakeExplainer._dfs(node, subtree)

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
