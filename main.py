import sys
import os
import pandas as pd
from tqdm import tqdm
from src.config import DATA_PATH
from src.data_loader import DataLoader
from src.sentiment import SentimentAnalyzer
from src.processing import DataProcessor
from src.model import StrategyModel

def main():
    print("=== Démarrage du Pipeline Fed Sentiment ===\n")

    # 1. Chemins des fichiers
    # On définit le nom du fichier de cache (résultats FinBERT déjà calculés)
    cache_path = os.path.join("data", "communication_with_sentiment.csv")

    # 2. Chargement des données de base
    loader = DataLoader()
    try:
        df_market = loader.fetch_market_data()
    except Exception as e:
        print(f"❌ Erreur marché : {e}")
        sys.exit(1)

    # 3. Sentiment Analysis (avec gestion du Cache)
    if os.path.exists(cache_path):
        print(f"♻️  Fichier de cache trouvé ! Chargement des scores déjà calculés...")
        df_filtered = pd.read_csv(cache_path)
        df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    else:
        print(f"🔍 Aucun cache trouvé. Lancement de l'analyse complète...")
        try:
            df_speeches = loader.load_speeches(DATA_PATH)
            processor = DataProcessor()
            df_filtered = processor.filter_speeches(df_speeches)
            
            analyzer = SentimentAnalyzer()
            tqdm.pandas(desc="Analyse FinBERT")
            print("\n⏳ Calcul des scores (FinBERT) - Uniquement au premier lancement...")
            df_filtered['finbert_score'] = df_filtered['texte'].progress_apply(analyzer.predict_score)
            
            # On sauvegarde pour la prochaine fois
            df_filtered.to_csv(cache_path, index=False)
            print(f"💾 Scores sauvegardés dans {cache_path}")
        except Exception as e:
            print(f"❌ Erreur lors du calcul : {e}")
            sys.exit(1)

    # 4. Alignement Données NLP <-> Marché
    processor = DataProcessor() # On en a besoin pour aligner
    df_final = processor.align_market_data(df_filtered, df_market)
    print(f"✅ Dataset final prêt : {len(df_final)} échantillons.")

    # 5. Modélisation et Évaluation
    if len(df_final) > 5:
        strategy = StrategyModel()
        X, y = strategy.prepare_data(df_final)
        X_test, y_test = strategy.train(X, y)
        strategy.evaluate(X_test, y_test)
    else:
        print("⚠️ Pas assez de données alignées pour l'entraînement.")

if __name__ == "__main__":
    main()