"""Add recurrent depth to pretrained models.

`program` describes which parent layers run, in what order; it needs no torch.
`runner` installs a program on a loaded Hugging Face model, sharing the parent's
weights and giving every executed position its own attention-cache slot.
"""
