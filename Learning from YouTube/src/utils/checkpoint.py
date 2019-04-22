"""
This file contains the logic for generating checkpoint rewards for an agent to
recieve while training.
"""

def get_checkpoints(vid_embeddings, N):
"""
Generates list of checkpoints every N timesteps
"""
    for embedding in range(0, len(vid_embeddings)):
        if embedding

def get_reward(obs_embedding, checkpoints):
