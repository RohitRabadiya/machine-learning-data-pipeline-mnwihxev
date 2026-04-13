import pandas as pd

def create_features(data: pd.DataFrame):
    # Example feature: log transformation of a column
    data['log_feature'] = data['feature'].
        apply(lambda x: np.log(x + 1))
    
    # Define features and target
    features = data.drop('target', axis=1)
    target = data['target']
    
    return features, target