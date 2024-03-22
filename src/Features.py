import ta
import pandas as pd


class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def calculate_technical_indicators(self):
        """
        Calculate specified technical indicators for the dataset.
        """
        self.data['rsi'] = ta.momentum.rsi(self.data['Dernier'], window=14)
        self.data['macd'] = ta.trend.macd_diff(self.data['Dernier'])
        self.data['ema20'] = ta.trend.ema_indicator(self.data['Dernier'], window=20)
        self.data['ema50'] = ta.trend.ema_indicator(self.data['Dernier'], window=50)

    def normalize_features(self):
        """
        Normalize features to have a similar scale.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = ['rsi', 'macd', 'ema20', 'ema50']  # Specify the features to normalize
        self.data[features] = scaler.fit_transform(self.data[features])

    def integrate_economic_indicators(self, economic_data):
        """
        Integrate economic indicators with market data.
        This function assumes economic_data is a DataFrame where columns are indicators and rows align with self.data's timeline.
        """
        self.data = pd.concat([self.data, economic_data], axis=1)

    def construct_feature_vector(self):
        """
        Combine all features into a single vector for each timestep.
        Assuming all necessary features are already columns in self.data,
        this function will return a numpy array representation of the DataFrame.
        """
        return self.data.values