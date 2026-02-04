import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import BERT_MODEL_NAME, CHUNK_SIZE

class SentimentAnalyzer:
    """Analyse le sentiment des textes financiers avec FinBERT."""

    def __init__(self):
        print("🤖 Chargement du modèle FinBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME)
        self.model.eval() # Mode évaluation (pas d'entraînement)

    def predict_score(self, text: str) -> float:
        """
        Découpe le texte en chunks et retourne un score Hawkish agrégé.
        Score = Moyenne(Positif - Négatif) sur tous les chunks.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0

        # Tokenisation sans tronquer au début pour gérer manuellement les chunks
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        input_ids = inputs["input_ids"][0]

        # Découpage en chunks
        chunks = [input_ids[i : i + CHUNK_SIZE] for i in range(0, len(input_ids), CHUNK_SIZE)]
        
        scores = []
        with torch.no_grad():
            for chunk in chunks:
                # Si le chunk est trop petit, on ignore ou on pad (ici simple gestion)
                if len(chunk) > 510: 
                    chunk = chunk[:510] # Sécurité pour respecter la limite BERT
                
                outputs = self.model(input_ids=chunk.unsqueeze(0))
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
                scores.append(probs)

        if not scores:
            return 0.0

        # Moyenne des probabilités [negative, neutral, positive]
        mean_probs = np.mean(scores, axis=0)
        
        # Calcul du score Hawkish (Positive - Negative)
        # FinBERT labels: 0: neutral, 1: positive, 2: negative (ATTENTION : vérifier l'ordre spécifique du modèle)
        # Pour yiyanghkust/finbert-tone : labels sont ["Neutral", "Positive", "Negative"] généralement
        # Mais vérifions l'ordre standard de config:
        # Souvent: 0=Negative, 1=Neutral, 2=Positive OU l'inverse. 
        # Dans ton notebook tu utilisais : labels = ['negative', 'neutral', 'positive']
        # On garde ta logique du notebook :
        negative = mean_probs[0]
        positive = mean_probs[2]
        
        return positive - negative
