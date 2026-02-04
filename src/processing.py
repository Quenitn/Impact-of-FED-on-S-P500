import pandas as pd
from datetime import timedelta
from src.config import RELEVANT_TYPES, MIN_TEXT_LENGTH, KEYWORDS

class DataProcessor:
    def filter_speeches(self, df_speeches: pd.DataFrame) -> pd.DataFrame:
        df = df_speeches[df_speeches['type'].isin(RELEVANT_TYPES)].copy()
        df = df[df['texte'].str.len() > MIN_TEXT_LENGTH]
        pattern = '|'.join(KEYWORDS)
        df = df[df['texte'].str.contains(pattern, case=False, na=False)]
        return df

    def align_market_data(self, df_speeches: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
        results = []
        df_market['date'] = pd.to_datetime(df_market['date'])
        
        for _, row in df_speeches.iterrows():
            date_discours = row['date']
            next_day = date_discours + timedelta(days=1)
            
            # Trouver le prochain jour de trading valide
            # On cherche les dates de marché supérieures ou égales à next_day
            future_market = df_market[df_market['date'] >= next_day]
            
            if not future_market.empty:
                # On prend la première date disponible (le lendemain ou le lundi suivant)
                idx = future_market.index[0]
                market_return = df_market.loc[idx, 'return']
                
                results.append({
                    'finbert_score': row.get('finbert_score', 0),
                    'market_return': market_return,
                    'date': date_discours,
                    'type': row['type']
                })

        return pd.DataFrame(results)