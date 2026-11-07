"""Exploratory re-decode of the both-reasoning paraphrases with the writer's instructions shown to the decoder.

Reuses that run's paraphrases, job ids, option orders, models, and acceptance decisions;
only the decoder prompts change, by showing the writer's instructions with the source
message and the secret bit withheld.
"""
