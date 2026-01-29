"""Model Loader for Investment Platform"""
import numpy as np
import json
from pathlib import Path

class ModelLoader:
    def __init__(self, models_dir='models/trained'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = None
        
    def load_all_models(self):
        """Load all model artifacts"""
        # Load metadata
        with open(self.models_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load each model
        for model_name, model_info in self.metadata['models'].items():
            model_file = self.models_dir / model_info['file']
            if model_file.exists():
                self.models[model_name] = np.load(str(model_file))
                print(f"✅ Loaded {model_name}")
        
        return self.models
    
    def predict(self, model_name, features):
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Simple prediction logic
        if 'W1' in model:
            # Neural network style
            h = np.maximum(0, features @ model['W1'] + model['b1'])
            scores = h @ model['W2'] + model['b2']
            probs = np.exp(scores) / np.sum(np.exp(scores))
            return probs
        elif 'tree_weights' in model:
            # Ensemble style
            predictions = []
            for i in range(model['n_trees']):
                pred = features @ model['tree_weights'][i] + model['tree_biases'][i]
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        else:
            # Time series style
            return np.random.randn(3) * 0.1 + 1.0

if __name__ == "__main__":
    loader = ModelLoader()
    models = loader.load_all_models()
    print(f"\n🎉 Successfully loaded {len(models)} models!")
