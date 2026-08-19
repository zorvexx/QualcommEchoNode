import os
import pandas as pd
import numpy as np

def session_aware_split(df_features, val_ratio=0.2, test_ratio=0.2):
    """
    Splits feature dataset by session_id ensuring test set contains both healthy and abnormal sessions.
    """
    sessions = list(df_features['session_id'].unique())
    
    # Check sessions with abnormal labels
    abnormal_sessions = df_features[df_features['label'] == 1]['session_id'].unique()
    healthy_sessions = [s for s in sessions if s not in abnormal_sessions]
    
    # Guarantee at least one abnormal session in test set if present
    if len(abnormal_sessions) > 0:
        test_sessions = [abnormal_sessions[0]]
        val_sessions = healthy_sessions[:1] if len(healthy_sessions) > 1 else []
        train_sessions = [s for s in sessions if s not in test_sessions and s not in val_sessions]
    else:
        test_sessions = [sessions[0]]
        val_sessions = [sessions[1]] if len(sessions) > 1 else []
        train_sessions = sessions[2:] if len(sessions) > 2 else sessions
        
    train_df = df_features[df_features['session_id'].isin(train_sessions)]
    val_df = df_features[df_features['session_id'].isin(val_sessions)]
    test_df = df_features[df_features['session_id'].isin(test_sessions)]
    
    return train_df, val_df, test_df
