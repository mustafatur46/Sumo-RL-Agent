"""
Action-Mask-Funktion für sb3_contrib.ActionMasker.

Eigenes Top-Level-Modul, damit SubprocVecEnv die Funktion unter Windows
(spawn) picklen kann.
"""


def mask_fn(env):
    return env.unwrapped.action_masks()
