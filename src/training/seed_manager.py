TRAIN_SEEDS = [1, 2, 3]
EVAL_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


def seed_schedule(train_seeds=None, eval_seeds=None):
    return train_seeds or TRAIN_SEEDS, eval_seeds or EVAL_SEEDS
