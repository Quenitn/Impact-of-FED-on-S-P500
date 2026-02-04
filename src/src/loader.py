import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        """
        Initialise le chargeur avec le chemin du fichier.
        """
        self.file_path = file_path

    def load_data(self):
        """
        Charge le CSV et retourne un DataFrame.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Le fichier {self.file_path} est introuvable.")
        
        df = pd.read_csv(self.file_path)
        print(f"✅ Données chargées : {len(df)} lignes trouvées.")
        return df

    def get_official_prices(self, start_date, end_date):
        """
        Génère ou charge les prix officiels (la 'Vérité terrain').
        """
        # Ici on peut mettre ta logique de création de prix officiels 
        # que nous avons vue ensemble précédemment.
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # ... logique de simulation ou import Excel ...
        return pd.DataFrame({'price': 60.0}, index=dates) # Exemple simplifié
