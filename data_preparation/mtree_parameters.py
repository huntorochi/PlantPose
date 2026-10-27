"""Example Mtree parameter draws in Blender scene units; no graph conversion."""
import random


def sample_parameters(seed=10):
    rng = random.Random(seed)
    return {
        'trunk': {
            'seed': rng.randint(0, 1_000_000),
            'length': rng.uniform(20.0, 30.0),
            'radius': rng.uniform(0.4, 0.6),
            'end_radius': rng.uniform(0.0, 0.1),
            'resolution': rng.uniform(1.0, 1.2),
            'shape': rng.uniform(0.7, 1.1),
            'randomness': rng.uniform(0.08, 0.4),
            'axis_attraction': rng.uniform(0.2, 0.3),
        },
        'branch': {
            'seed': rng.randint(0, 1_000_000),
            'amount': rng.randint(6, 15),
            'split_angle': rng.uniform(0.5, 0.7),
            'max_split_number': rng.randint(2, 3),
            'radius': rng.uniform(0.5, 0.8),
            'end_radius': rng.uniform(0.0, 0.1),
            'min_height': rng.uniform(0.08, 0.12),
            'length': rng.uniform(5.0, 13.0),
            'shape_start': rng.uniform(0.9, 1.1),
            'shape_end': rng.uniform(0.9, 1.1),
            'shape_convexity': rng.uniform(0.2, 0.4),
            'resolution': rng.uniform(1.0, 1.2),
            'randomness': rng.uniform(0.08, 0.4),
            'split_proba': rng.uniform(0.1, 0.5),
            'split_flatten': rng.uniform(0.4, 0.6),
            'gravity_strength': rng.uniform(0.2, 0.4),
        },
    }


if __name__ == '__main__':
    import json
    print(json.dumps(sample_parameters(), indent=2))
