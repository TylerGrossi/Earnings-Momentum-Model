"""
Win Probability Predictor
=========================
Machine learning module to predict win probability for stocks entering the database.
Uses historical returns_tracker data to train a model that predicts whether a stock
will have a positive 5-day return (win) based on financial metrics.

Features:
- P/E ratio (with low/high categorization)
- Beta (with low/high categorization)
- Market Cap (with large/mid/small cap categorization)
- Sector
- Earnings Timing (BMO/AMC)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import os
import streamlit as st

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ------------------------------------
# FEATURE ENGINEERING
# ------------------------------------

def parse_market_cap(val):
    """Parse market cap string to numeric value."""
    if pd.isna(val) or val == 'N/A' or val == '':
        return np.nan
    val = str(val).upper().replace(',', '').replace('$', '').strip()
    multiplier = 1
    if 'T' in val:
        multiplier = 1_000_000_000_000
        val = val.replace('T', '')
    elif 'B' in val:
        multiplier = 1_000_000_000
        val = val.replace('B', '')
    elif 'M' in val:
        multiplier = 1_000_000
        val = val.replace('M', '')
    elif 'K' in val:
        multiplier = 1_000
        val = val.replace('K', '')
    try:
        return float(val) * multiplier
    except:
        return np.nan


def categorize_pe(pe_value):
    """
    Categorize P/E ratio to match PowerBI PE Group:
    - Blank or negative -> "Negative EPS"
    - pe < 15 -> "Low P/E"
    - pe < 25 -> "Medium P/E"
    - else -> "High P/E"
    """
    if pd.isna(pe_value) or pe_value == '' or pe_value == 'N/A':
        return 'Negative EPS'
    try:
        pe = float(pe_value)
        if pe < 0:
            return 'Negative EPS'
        elif pe < 15:
            return 'Low P/E'
        elif pe < 25:
            return 'Medium P/E'
        else:
            return 'High P/E'
    except (ValueError, TypeError):
        return 'Negative EPS'


def categorize_beta(beta_value):
    """Categorize Beta into low, medium, high."""
    if pd.isna(beta_value):
        return 'Unknown'
    try:
        beta = float(beta_value)
        if beta < 0.8:
            return 'Low Beta'
        elif beta < 1.5:
            return 'Medium Beta'
        else:
            return 'High Beta'
    except:
        return 'Unknown'


def categorize_market_cap(mcap_value):
    """
    Categorize Market Cap to match PowerBI Market Cap Category:
    - < 2e9 -> "Small Cap"
    - >= 2e9 and < 10e9 -> "Mid Cap"
    - >= 10e9 -> "Large Cap"
    - else -> "Other"
    """
    if pd.isna(mcap_value) or mcap_value == '' or mcap_value == 'N/A':
        return 'Other'
    mcap_numeric = parse_market_cap(mcap_value)
    if pd.isna(mcap_numeric) or mcap_numeric < 0:
        return 'Other'
    
    if mcap_numeric < 2e9:  # < $2B
        return 'Small Cap'
    elif mcap_numeric < 10e9:  # >= $2B and < $10B
        return 'Mid Cap'
    else:  # >= $10B
        return 'Large Cap'


def prepare_features(df):
    """
    Prepare features for ML model from returns_tracker dataframe.
    Rows where Date Check == "DATE PASSED" are removed before training.
    
    Args:
        df: DataFrame with columns: P/E, Beta, Market Cap, Sector, Earnings Timing, 3D Return, Date Check
    
    Returns:
        X: Feature matrix
        y: Target vector (1 if 3D Return > 0, else 0)
        feature_names: List of feature names
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    df = df.copy()
    
    # Remove rows where Date Check is DATE PASSED (keep only rows that are not DATE PASSED)
    if "Date Check" in df.columns:
        df = df[df["Date Check"] != "DATE PASSED"].copy()
    
    # Use Forward P/E if present (matches PowerBI), else P/E
    pe_col = 'Forward P/E' if 'Forward P/E' in df.columns else 'P/E'
    df['P/E Numeric'] = pd.to_numeric(df[pe_col], errors='coerce')
    # Fill missing/negative P/E with -1 so "Negative EPS" rows stay in training
    df.loc[df['P/E Numeric'].isna() | (df['P/E Numeric'] < 0), 'P/E Numeric'] = -1
    df['Beta Numeric'] = pd.to_numeric(df['Beta'], errors='coerce')
    df['Market Cap Numeric'] = df['Market Cap'].apply(parse_market_cap)
    
    # Create categorical features (PE Group and Market Cap Category match PowerBI)
    df['P/E Category'] = df[pe_col].apply(categorize_pe)
    df['Beta Category'] = df['Beta'].apply(categorize_beta)
    df['Market Cap Category'] = df['Market Cap'].apply(categorize_market_cap)
    
    # Handle Sector
    df['Sector'] = df['Sector'].fillna('Unknown')
    
    # Handle Earnings Timing
    df['Earnings Timing'] = df['Earnings Timing'].fillna('Unknown')
    df['Earnings Timing'] = df['Earnings Timing'].astype(str).str.upper()
    df['Earnings Timing'] = df['Earnings Timing'].apply(
        lambda x: 'BMO' if 'BMO' in x else ('AMC' if 'AMC' in x else 'Unknown')
    )
    
    # Filter to rows with all required features and valid target
    required_cols = ['P/E Numeric', 'Beta Numeric', 'Market Cap Numeric',
                     'P/E Category', 'Beta Category', 'Market Cap Category',
                     'Sector', 'Earnings Timing', '3D Return']
    
    df_clean = df[required_cols].copy()
    df_clean = df_clean.dropna(subset=['P/E Numeric', 'Beta Numeric', 'Market Cap Numeric', '3D Return'])
    
    if len(df_clean) == 0:
        raise ValueError("No valid data rows after cleaning")
    
    # Create target variable (1 if positive return, 0 otherwise)
    df_clean['Win'] = (df_clean['3D Return'] > 0).astype(int)
    
    # Encode categorical features
    le_pe = LabelEncoder()
    le_beta = LabelEncoder()
    le_cap = LabelEncoder()
    le_sector = LabelEncoder()
    le_timing = LabelEncoder()
    
    df_clean['P/E Category Encoded'] = le_pe.fit_transform(df_clean['P/E Category'])
    df_clean['Beta Category Encoded'] = le_beta.fit_transform(df_clean['Beta Category'])
    df_clean['Market Cap Category Encoded'] = le_cap.fit_transform(df_clean['Market Cap Category'])
    df_clean['Sector Encoded'] = le_sector.fit_transform(df_clean['Sector'])
    df_clean['Earnings Timing Encoded'] = le_timing.fit_transform(df_clean['Earnings Timing'])
    
    # Select features
    feature_cols = [
        'P/E Numeric',
        'Beta Numeric',
        'Market Cap Numeric',
        'P/E Category Encoded',
        'Beta Category Encoded',
        'Market Cap Category Encoded',
        'Sector Encoded',
        'Earnings Timing Encoded'
    ]
    
    X = df_clean[feature_cols].values
    y = df_clean['Win'].values
    
    # Store encoders for later use
    encoders = {
        'pe': le_pe,
        'beta': le_beta,
        'cap': le_cap,
        'sector': le_sector,
        'timing': le_timing
    }
    
    feature_names = feature_cols
    
    return X, y, encoders, feature_names


# ------------------------------------
# MODEL TRAINING
# ------------------------------------

def train_model(X, y, feature_names, model_type='random_forest'):
    """
    Train a classification model to predict win probability.
    
    Args:
        X: Feature matrix
        y: Target vector (1 for win, 0 for loss)
        feature_names: List of feature names (for importance output)
        model_type: 'random_forest' or 'gradient_boosting'
    
    Returns:
        model: Trained model
        metrics: Dictionary of performance metrics and feature importance
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Choose model (original hyperparameters)
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'win_rate_train': float(y_train.mean()),
        'win_rate_test': float(y_test.mean()),
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics['cv_accuracy_mean'] = float(cv_scores.mean())
    metrics['cv_accuracy_std'] = float(cv_scores.std())
    
    # Feature importance (for Streamlit display)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        metrics['feature_importance'] = dict(zip(feature_names, [float(x) for x in imp]))
    else:
        metrics['feature_importance'] = {}
    
    return model, metrics


# ------------------------------------
# PREDICTION
# ------------------------------------

def predict_win_probability(model, encoders, stock_data):
    """
    Predict win probability for a single stock or multiple stocks.
    
    Args:
        model: Trained model
        encoders: Dictionary of label encoders
        stock_data: Dict or DataFrame with stock features:
                   {'P/E': value, 'Beta': value, 'Market Cap': value, 
                    'Sector': value, 'Earnings Timing': value}
    
    Returns:
        win_probability: Probability of win (0-1) or array of probabilities
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    # Handle single stock dict
    if isinstance(stock_data, dict):
        stock_data = pd.DataFrame([stock_data])
    
    df = stock_data.copy()
    
    # Parse numeric features (match training: missing/negative P/E -> -1 for Negative EPS)
    df['P/E Numeric'] = pd.to_numeric(df['P/E'], errors='coerce')
    df.loc[df['P/E Numeric'].isna() | (df['P/E Numeric'] < 0), 'P/E Numeric'] = -1
    df['Beta Numeric'] = pd.to_numeric(df['Beta'], errors='coerce')
    df['Market Cap Numeric'] = df['Market Cap'].apply(parse_market_cap)
    
    # Create categorical features
    df['P/E Category'] = df['P/E'].apply(categorize_pe)
    df['Beta Category'] = df['Beta'].apply(categorize_beta)
    df['Market Cap Category'] = df['Market Cap'].apply(categorize_market_cap)
    
    # Handle Sector
    df['Sector'] = df['Sector'].fillna('Unknown')
    
    # Handle Earnings Timing
    df['Earnings Timing'] = df['Earnings Timing'].fillna('Unknown')
    df['Earnings Timing'] = df['Earnings Timing'].astype(str).str.upper()
    df['Earnings Timing'] = df['Earnings Timing'].apply(
        lambda x: 'BMO' if 'BMO' in x else ('AMC' if 'AMC' in x else 'Unknown')
    )
    
    # Encode categorical features (handle unseen categories)
    def safe_encode(encoder, values, default='Unknown'):
        """Encode values, handling unseen categories."""
        result = []
        for val in values:
            if val in encoder.classes_:
                result.append(encoder.transform([val])[0])
            else:
                # Use default if available, otherwise use first class
                if default in encoder.classes_:
                    result.append(encoder.transform([default])[0])
                else:
                    result.append(0)  # Use first class as fallback
        return np.array(result)
    
    df['P/E Category Encoded'] = safe_encode(encoders['pe'], df['P/E Category'])
    df['Beta Category Encoded'] = safe_encode(encoders['beta'], df['Beta Category'])
    df['Market Cap Category Encoded'] = safe_encode(encoders['cap'], df['Market Cap Category'])
    df['Sector Encoded'] = safe_encode(encoders['sector'], df['Sector'])
    df['Earnings Timing Encoded'] = safe_encode(encoders['timing'], df['Earnings Timing'])
    
    # Select features (must match training)
    feature_cols = [
        'P/E Numeric',
        'Beta Numeric',
        'Market Cap Numeric',
        'P/E Category Encoded',
        'Beta Category Encoded',
        'Market Cap Category Encoded',
        'Sector Encoded',
        'Earnings Timing Encoded'
    ]
    
    # Fill missing numeric values
    for col in ['P/E Numeric', 'Beta Numeric', 'Market Cap Numeric']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_cols].values
    
    # Predict probabilities
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (win)
    
    # Return single value if input was single dict, else return array
    if isinstance(stock_data, dict) or len(probabilities) == 1:
        return float(probabilities[0])
    else:
        return probabilities


# ------------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------------

def _train_win_probability_model_impl(returns_df, model_type='random_forest'):
    """
    Core training logic (no cache). Used when running as library (e.g. Earnings Momentum Model).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    if returns_df is None or returns_df.empty:
        raise ValueError("No data provided for training")
    
    # Prepare features
    X, y, encoders, feature_names = prepare_features(returns_df)
    
    if len(X) < 10:
        raise ValueError(f"Not enough data for training. Need at least 10 samples, got {len(X)}")
    
    # Train model
    model, metrics = train_model(X, y, feature_names, model_type=model_type)
    
    return model, encoders, metrics


def train_win_probability_model(returns_df, model_type='random_forest', use_cache=True):
    """
    Train win probability model from returns_tracker data.
    
    Args:
        returns_df: DataFrame with historical returns data
        model_type: 'random_forest' or 'gradient_boosting'
        use_cache: If True, use Streamlit cache (for Streamlit apps). If False, always train fresh (for library/script use).
    
    Returns:
        model: Trained model
        encoders: Dictionary of label encoders
        metrics: Performance metrics
    """
    if use_cache:
        return _train_win_probability_model_cached(returns_df, model_type)
    return _train_win_probability_model_impl(returns_df, model_type)


@st.cache_data(ttl=3600)
def _train_win_probability_model_cached(returns_df, model_type='random_forest'):
    """Cached wrapper for Streamlit apps."""
    return _train_win_probability_model_impl(returns_df, model_type)


# ------------------------------------
# BATCH PREDICTION FOR STOCK SCREENER
# ------------------------------------

def predict_batch(model, encoders, stock_list):
    """
    Predict win probabilities for a batch of stocks.
    
    Args:
        model: Trained model
        encoders: Dictionary of label encoders
        stock_list: List of dicts, each with stock features:
                   [{'Ticker': 'AAPL', 'P/E': '25.5', 'Beta': '1.2', 
                     'Market Cap': '3.5T', 'Sector': 'Technology', 
                     'Earnings Timing': 'AMC'}, ...]
    
    Returns:
        List of win probabilities (0-1)
    """
    if len(stock_list) == 0:
        return []
    
    probabilities = predict_win_probability(model, encoders, pd.DataFrame(stock_list))
    
    # Ensure it's a list
    if isinstance(probabilities, np.ndarray):
        return probabilities.tolist()
    elif isinstance(probabilities, (int, float)):
        return [probabilities]
    else:
        return list(probabilities)
