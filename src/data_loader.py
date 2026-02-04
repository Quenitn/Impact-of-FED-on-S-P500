import pandas as pd
import yfinance as yf
from src.config import MARKET_TICKER, START_DATE

class DataLoader:
    """Gère le chargement des données de discours et de marché."""

    def load_speeches(self, filepath: str) -> pd.DataFrame:
        """Charge et nettoie le fichier CSV des discours."""
        try:
            df = pd.read_csv(filepath)
            # Standardisation des colonnes
            df = df.rename(columns={"Date": "date", "Text": "texte", "Type": "type"})
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ {len(df)} discours chargés.")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Fichier non trouvé : {filepath}")

    def fetch_market_data(self) -> pd.DataFrame:
        """Télécharge les données du marché via yfinance."""
        print(f"📥 Téléchargement des données pour {MARKET_TICKER}...")
        df = yf.download(MARKET_TICKER, start=START_DATE, progress=False)
        
        # Calcul du rendement journalier (Close to Close)
        # Note: yfinance retourne parfois un MultiIndex, on s'assure d'avoir 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        
        df = df.reset_index()
        # On renomme pour avoir une colonne 'date' et 'return'
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        
        # Si df est une Series après extraction, on la remet en DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(name='Close').reset_index()

        # Calcul des retours
        # On suppose que la colonne de prix s'appelle soit 'Close', soit le ticker
        price_col = 'Close' if 'Close' in df.columns else MARKET_TICKER
        df['return'] = df[price_col].pct_change()
        
        df = df.dropna()
        print(f"✅ Données marché chargées : {len(df)} jours de trading.")
        return df
