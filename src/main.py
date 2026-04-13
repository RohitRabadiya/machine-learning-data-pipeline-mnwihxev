import pandas as pd
from data_cleaning import clean_data
from feature_engineering import create_features
from model_training import train_model


def main():
    # Load data
    data = pd.read_csv('data/dataset.csv')
    
    # Clean data
    clean_data(data)
    
    # Create features
    features, target = create_features(data)
    
    # Train model
    model = train_model(features, target)
    
    # Save model
    model.save('output/model.pkl')


if __name__ == '__main__':
    main()