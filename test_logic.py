from logic_engine import KnowledgeBase


def test_forward_chaining():
    kb = KnowledgeBase()

    # Add Domain Rules
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    # Test Case 1: Safe Engagement
    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.forward_chain()
    assert 'SafeToEngage' in kb.facts, "Test 1 Failed: Should deduce SafeToEngage"
    assert 'Retreat' not in kb.facts, "Test 1 Failed: Should NOT deduce Retreat"

    # Test Case 2: Unsafe Engagement (Bloodseeker Missing)
    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.tell_fact('BloodseekerMissing')
    kb.forward_chain()
    assert 'Retreat' in kb.facts, "Test 2 Failed: Should deduce Retreat"

    print("All Logic Engine Test Cases Passed!")


def test_agent_default_avoids_retreat_tile():
    """Lab 07: the default agent must use the KB-aware A* and avoid (1, 0)."""
    from agent import SearchAgent

    agent = SearchAgent()
    percept = {
        'agent_pos': [0, 0],
        'all_food': [(1, 0)],
        'walls': [],
        'grid_size': (2, 2),
    }
    action = agent.sense_and_act(percept)
    assert action != 'Right', "Agent tried to move onto the unsafe tile (1, 0)"


def test_astar_never_enters_retreat_tile():
    """A* must route around a tile where the KB derives Retreat."""
    from agent import SearchAgent

    agent = SearchAgent()
    tile_facts = {
        (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],
        (0, 1): ['TargetVisible', 'HasDust'],
    }
    path = agent.astar_search((0, 0), (2, 0), [], (3, 2), tile_facts=tile_facts)
    assert path is not None, "A* should find a safe route around (1, 0)"

    pos = (0, 0)
    visited = [pos]
    for action in path:
        dx, dy = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}[action]
        pos = (pos[0] + dx, pos[1] + dy)
        visited.append(pos)

    assert (1, 0) not in visited, "A* path entered the Retreat tile (1, 0)"
    assert pos == (2, 0), "A* path did not reach the goal"


def test_astar_rejects_unsafe_goal():
    """A* must return no path when the goal itself is a Retreat tile."""
    from agent import SearchAgent

    agent = SearchAgent()
    tile_facts = {
        (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],
    }
    path = agent.astar_search((0, 0), (1, 0), [], (2, 2), tile_facts=tile_facts)
    assert path is None, "A* should not return a path onto an unsafe goal tile"


def test_visual_game_does_not_spawn_food_on_retreat_tile():
    """Lab 07 demo: no food on (1, 0), because the agent must never enter it."""
    import random
    from visual_grid_game import VisualGridHuntGame

    for seed in range(50):
        random.seed(seed)
        env = VisualGridHuntGame(width=6, height=6, num_food=10, num_opponents=0)
        assert (1, 0) not in env.food_positions, \
            f"Seed {seed} placed food on the unsafe tile (1, 0)"


def test_visual_environment_agent_avoids_retreat_tile():
    """The visual game's percepts must drive the A* agent safely past (1, 0)."""
    import random
    from agent import SearchAgent
    from visual_grid_game import VisualGridHuntGame

    random.seed(0)
    env = VisualGridHuntGame(width=8, height=8, num_food=6, num_opponents=0)
    agent = SearchAgent()

    visited = []
    while not env.is_done():
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        visited.append(tuple(env.agent_pos))

    assert (1, 0) not in visited, "Agent stepped onto the unsafe tile (1, 0)"


if __name__ == "__main__":
    test_forward_chaining()
    test_agent_default_avoids_retreat_tile()
    test_astar_never_enters_retreat_tile()
    test_astar_rejects_unsafe_goal()
    test_visual_game_does_not_spawn_food_on_retreat_tile()
    test_visual_environment_agent_avoids_retreat_tile()
