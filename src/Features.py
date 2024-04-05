import ta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    def __init__(self, df):
        """
        Initialize the FeatureEngineer object with a DataFrame.
        """
        self.df = df.copy()

    def apply_preprocessing(self):
        """
        Apply preprocessing steps to the dataset.
        """
        self.replace_commas()
        self.sort_by_date()
        self.calculate_technical_indicators()  # Calculate RSI, MACD, EMA20, EMA50
        self.normalise_technical_indicators()
        self.convert_to_numeric()  # Convert 'K' and 'M' suffixes to numeric values
        self.fill_and_drop()  # Fill NaN values and drop rows with NaN values

    def replace_commas(self):
        lists = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Vol.", "Variation %"]
        for i in lists:
            self.df[i] = self.df[i].str.replace(",", ".")

    def sort_by_date(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values(by="Date")
        self.df.reset_index(drop=True)
        # self.df['Date'] = pd.to_numeric(self.df['Date'])

    def calculate_technical_indicators(self):
        self.df["Dernier"] = pd.to_numeric(self.df["Dernier"], errors="coerce")
        self.df["rsi"] = ta.momentum.rsi(self.df["Dernier"], window=14)
        self.df["macd"] = ta.trend.macd_diff(self.df["Dernier"])
        self.df["ema20"] = ta.trend.ema_indicator(self.df["Dernier"], window=20)
        self.df["ema50"] = ta.trend.ema_indicator(self.df["Dernier"], window=50)

    def normalise_technical_indicators(self):

        scaler = MinMaxScaler()
        features = ["rsi", "macd", "ema20", "ema50"]
        self.df[features] = scaler.fit_transform(self.df[features])

    def convert_k_m_to_numeric(self, value):
        """
        Convert values with 'K' or 'M' suffix to float numbers.
        Args:
        - value: The string or numeric value to convert.

        Returns:
        - The converted value as float if 'K' or 'M' was found; otherwise, the original value.
        """
        if isinstance(value, str):  # Only process strings
            if value.endswith("K"):
                return float(value[:-1]) * 1e3
            elif value.endswith("M"):
                return float(value[:-1]) * 1e6
            elif value.endswith("%"):
                return float(value[:-1]) / 100
        return float(value)

    def convert_to_numeric(self):
        for column in ["Ouv.", " Plus Haut", "Plus Bas", "Vol.", "Variation %"]:
            self.df[column] = self.df[column].apply(self.convert_k_m_to_numeric)

    def fill_and_drop(self):
        self.df["Vol."] = self.df["Vol."].fillna(self.df["Vol."].median())
        self.df = self.df.dropna()
