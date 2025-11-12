"""
Synthetic User Behavior Data Generator

Generates realistic user behavior data for testing bad actor detection systems.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from datetime import datetime, timedelta


class BehaviorGenerator:
    """
    Generate synthetic user behavior data with normal and malicious patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_normal_user(self, n_sessions: int = 30) -> Dict:
        """
        Generate behavior profile for a normal user.
        
        Args:
            n_sessions: Number of sessions to simulate
            
        Returns:
            Dictionary of behavioral features
        """
        # Temporal patterns
        login_times = np.random.normal(14, 4, n_sessions)  # Centered around 2 PM
        login_times = np.clip(login_times, 0, 24)
        
        session_durations = np.random.lognormal(3, 0.5, n_sessions)  # Minutes
        session_durations = np.clip(session_durations, 1, 120)
        
        # Interaction patterns
        pages_per_session = np.random.poisson(15, n_sessions)
        clicks_per_session = np.random.poisson(25, n_sessions)
        
        # Transaction patterns (if applicable)
        transactions_per_session = np.random.binomial(1, 0.1, n_sessions)  # 10% chance
        transaction_amounts = np.random.lognormal(3, 1, n_sessions) * transactions_per_session
        
        # API usage
        api_calls_per_session = np.random.poisson(20, n_sessions)
        
        # Device and location
        device_changes = np.random.binomial(1, 0.05, n_sessions)  # 5% device change
        location_changes = np.random.binomial(1, 0.03, n_sessions)  # 3% location change
        
        # Aggregate features
        features = {
            # Temporal features
            'avg_login_hour': np.mean(login_times),
            'std_login_hour': np.std(login_times),
            'avg_session_duration': np.mean(session_durations),
            'std_session_duration': np.std(session_durations),
            'sessions_per_day': n_sessions / 30,
            
            # Interaction features
            'avg_pages_per_session': np.mean(pages_per_session),
            'avg_clicks_per_session': np.mean(clicks_per_session),
            'click_to_page_ratio': np.mean(clicks_per_session) / (np.mean(pages_per_session) + 1e-8),
            
            # Transaction features
            'transaction_frequency': np.mean(transactions_per_session),
            'avg_transaction_amount': np.mean(transaction_amounts[transaction_amounts > 0]) if any(transaction_amounts > 0) else 0,
            'max_transaction_amount': np.max(transaction_amounts),
            
            # API features
            'avg_api_calls': np.mean(api_calls_per_session),
            'api_call_variance': np.var(api_calls_per_session),
            
            # Stability features
            'device_change_rate': np.mean(device_changes),
            'location_change_rate': np.mean(location_changes),
            
            # Advanced patterns
            'weekend_activity_ratio': np.random.uniform(0.3, 0.7),  # Normal users use less on weekends
            'night_activity_ratio': np.random.uniform(0.05, 0.15),  # Low night activity
            'burst_activity_score': np.random.uniform(0.1, 0.3),  # Low burst activity
        }
        
        return features
    
    def generate_bot_user(self, n_sessions: int = 100) -> Dict:
        """
        Generate behavior profile for a bot user.
        
        Bots typically show:
        - Very regular patterns
        - High frequency
        - Unusual timing
        - Consistent behavior
        """
        # Very regular timing
        login_times = np.random.normal(12, 1, n_sessions)  # Very consistent timing
        
        # Short, consistent sessions
        session_durations = np.random.normal(2, 0.5, n_sessions)
        session_durations = np.clip(session_durations, 0.5, 5)
        
        # High, consistent interaction rates
        pages_per_session = np.random.poisson(30, n_sessions)  # Higher than normal
        clicks_per_session = np.random.poisson(50, n_sessions)  # Very high
        
        # Unusual transaction patterns
        transactions_per_session = np.random.binomial(1, 0.5, n_sessions)  # High frequency
        transaction_amounts = np.random.uniform(10, 50, n_sessions) * transactions_per_session
        
        # Very high API usage
        api_calls_per_session = np.random.poisson(100, n_sessions)  # Bot-like
        
        # No device/location changes (same machine)
        device_changes = np.zeros(n_sessions)
        location_changes = np.zeros(n_sessions)
        
        features = {
            'avg_login_hour': np.mean(login_times),
            'std_login_hour': np.std(login_times),  # Very low variance
            'avg_session_duration': np.mean(session_durations),
            'std_session_duration': np.std(session_durations),
            'sessions_per_day': n_sessions / 30,  # High frequency
            
            'avg_pages_per_session': np.mean(pages_per_session),
            'avg_clicks_per_session': np.mean(clicks_per_session),
            'click_to_page_ratio': np.mean(clicks_per_session) / (np.mean(pages_per_session) + 1e-8),
            
            'transaction_frequency': np.mean(transactions_per_session),
            'avg_transaction_amount': np.mean(transaction_amounts[transaction_amounts > 0]),
            'max_transaction_amount': np.max(transaction_amounts),
            
            'avg_api_calls': np.mean(api_calls_per_session),
            'api_call_variance': np.var(api_calls_per_session),
            
            'device_change_rate': np.mean(device_changes),
            'location_change_rate': np.mean(location_changes),
            
            'weekend_activity_ratio': np.random.uniform(0.4, 0.6),  # Bots work 24/7
            'night_activity_ratio': np.random.uniform(0.3, 0.5),  # High night activity
            'burst_activity_score': np.random.uniform(0.7, 0.9),  # High burst activity
        }
        
        return features
    
    def generate_fraudster_user(self, n_sessions: int = 50) -> Dict:
        """
        Generate behavior profile for a fraudulent user.
        
        Fraudsters typically:
        - Try to appear normal initially
        - Have sudden behavior changes
        - Focus on high-value transactions
        - May use VPNs (location changes)
        """
        # Mixed timing patterns
        login_times = np.random.uniform(0, 24, n_sessions)
        
        # Variable session durations
        session_durations = np.random.lognormal(2, 1.5, n_sessions)
        
        # Lower page views, focused navigation
        pages_per_session = np.random.poisson(8, n_sessions)  # Lower, targeted
        clicks_per_session = np.random.poisson(12, n_sessions)
        
        # High-value transactions
        transactions_per_session = np.random.binomial(1, 0.3, n_sessions)
        transaction_amounts = np.random.lognormal(5, 1.5, n_sessions) * transactions_per_session  # High amounts
        
        # Moderate API usage
        api_calls_per_session = np.random.poisson(35, n_sessions)
        
        # Frequent location changes (VPN/proxy)
        device_changes = np.random.binomial(1, 0.15, n_sessions)
        location_changes = np.random.binomial(1, 0.4, n_sessions)  # High location change
        
        features = {
            'avg_login_hour': np.mean(login_times),
            'std_login_hour': np.std(login_times),
            'avg_session_duration': np.mean(session_durations),
            'std_session_duration': np.std(session_durations),
            'sessions_per_day': n_sessions / 30,
            
            'avg_pages_per_session': np.mean(pages_per_session),
            'avg_clicks_per_session': np.mean(clicks_per_session),
            'click_to_page_ratio': np.mean(clicks_per_session) / (np.mean(pages_per_session) + 1e-8),
            
            'transaction_frequency': np.mean(transactions_per_session),
            'avg_transaction_amount': np.mean(transaction_amounts[transaction_amounts > 0]),
            'max_transaction_amount': np.max(transaction_amounts),
            
            'avg_api_calls': np.mean(api_calls_per_session),
            'api_call_variance': np.var(api_calls_per_session),
            
            'device_change_rate': np.mean(device_changes),
            'location_change_rate': np.mean(location_changes),  # High
            
            'weekend_activity_ratio': np.random.uniform(0.4, 0.8),
            'night_activity_ratio': np.random.uniform(0.2, 0.5),
            'burst_activity_score': np.random.uniform(0.5, 0.8),
        }
        
        return features
    
    def generate_abuser_user(self, n_sessions: int = 40) -> Dict:
        """
        Generate behavior profile for an abusive user (spam, harassment).
        
        Abusers typically:
        - High message/post frequency
        - Short sessions with specific goals
        - Repetitive actions
        """
        login_times = np.random.uniform(8, 22, n_sessions)
        
        # Short, focused sessions
        session_durations = np.random.lognormal(1.5, 0.8, n_sessions)
        session_durations = np.clip(session_durations, 1, 30)
        
        # High interaction focused on specific features
        pages_per_session = np.random.poisson(5, n_sessions)  # Low page variety
        clicks_per_session = np.random.poisson(40, n_sessions)  # High clicks (spam actions)
        
        # No or low transactions
        transactions_per_session = np.zeros(n_sessions)
        transaction_amounts = np.zeros(n_sessions)
        
        # High API usage (posting, messaging)
        api_calls_per_session = np.random.poisson(80, n_sessions)  # Very high
        
        device_changes = np.random.binomial(1, 0.08, n_sessions)
        location_changes = np.random.binomial(1, 0.1, n_sessions)
        
        features = {
            'avg_login_hour': np.mean(login_times),
            'std_login_hour': np.std(login_times),
            'avg_session_duration': np.mean(session_durations),
            'std_session_duration': np.std(session_durations),
            'sessions_per_day': n_sessions / 30,
            
            'avg_pages_per_session': np.mean(pages_per_session),
            'avg_clicks_per_session': np.mean(clicks_per_session),
            'click_to_page_ratio': np.mean(clicks_per_session) / (np.mean(pages_per_session) + 1e-8),
            
            'transaction_frequency': 0,
            'avg_transaction_amount': 0,
            'max_transaction_amount': 0,
            
            'avg_api_calls': np.mean(api_calls_per_session),  # Very high
            'api_call_variance': np.var(api_calls_per_session),
            
            'device_change_rate': np.mean(device_changes),
            'location_change_rate': np.mean(location_changes),
            
            'weekend_activity_ratio': np.random.uniform(0.5, 0.8),
            'night_activity_ratio': np.random.uniform(0.15, 0.35),
            'burst_activity_score': np.random.uniform(0.6, 0.9),  # High burst
        }
        
        return features


def generate_sample_data(
    n_users: int = 1000,
    bad_actor_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Generate sample dataset of user behaviors.
    
    Args:
        n_users: Total number of users to generate
        bad_actor_ratio: Fraction of users that are bad actors
        seed: Random seed
        
    Returns:
        user_ids: List of user IDs
        features: Feature matrix (n_users, n_features)
        labels: Binary labels (0=normal, 1=bad actor)
    """
    generator = BehaviorGenerator(seed=seed)
    
    n_bad = int(n_users * bad_actor_ratio)
    n_normal = n_users - n_bad
    
    # Generate different types of bad actors
    n_bots = int(n_bad * 0.4)
    n_fraudsters = int(n_bad * 0.3)
    n_abusers = n_bad - n_bots - n_fraudsters
    
    users_data = []
    labels = []
    
    # Generate normal users
    for i in range(n_normal):
        features = generator.generate_normal_user()
        users_data.append(features)
        labels.append(0)
    
    # Generate bots
    for i in range(n_bots):
        features = generator.generate_bot_user()
        users_data.append(features)
        labels.append(1)
    
    # Generate fraudsters
    for i in range(n_fraudsters):
        features = generator.generate_fraudster_user()
        users_data.append(features)
        labels.append(1)
    
    # Generate abusers
    for i in range(n_abusers):
        features = generator.generate_abuser_user()
        users_data.append(features)
        labels.append(1)
    
    # Convert to arrays
    feature_names = list(users_data[0].keys())
    feature_matrix = np.array([[user[fn] for fn in feature_names] for user in users_data])
    labels = np.array(labels)
    user_ids = list(range(n_users))
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_users)
    user_ids = [user_ids[i] for i in shuffle_idx]
    feature_matrix = feature_matrix[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return user_ids, feature_matrix, labels


def get_feature_names() -> List[str]:
    """Get list of feature names."""
    return [
        'avg_login_hour', 'std_login_hour',
        'avg_session_duration', 'std_session_duration',
        'sessions_per_day',
        'avg_pages_per_session', 'avg_clicks_per_session',
        'click_to_page_ratio',
        'transaction_frequency', 'avg_transaction_amount',
        'max_transaction_amount',
        'avg_api_calls', 'api_call_variance',
        'device_change_rate', 'location_change_rate',
        'weekend_activity_ratio', 'night_activity_ratio',
        'burst_activity_score'
    ]
