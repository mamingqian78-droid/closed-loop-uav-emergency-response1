from __future__ import annotations


def closed_loop_loop(env, policy):
    obs, _ = env.reset()
    done = False
    records = []
    while not done:
        try:
            action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
        except TypeError:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        records.append((int(action), float(reward), info))
        done = terminated or truncated
    return records
