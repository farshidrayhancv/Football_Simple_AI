"""Team classification module."""

import torch
import numpy as np
from transformers import AutoProcessor, SiglipVisionModel
from sports.common.team import TeamClassifier


class TeamClassifierModule:
    def __init__(self, device='cpu', hf_token=None, model_path=None):
        self.device = device
        self.hf_token = hf_token
        self.classifier = None
        # Use provided model path or default
        self.model_path = model_path or "google/siglip-base-patch16-224"
        self._load_models()
    
    def _load_models(self):
        """Load Siglip models."""
        print(f"Loading Siglip model from: {self.model_path}")
        self.siglip_model = SiglipVisionModel.from_pretrained(
            self.model_path,
            token=self.hf_token  # Changed from use_auth_token
        ).to(self.device)
        
        self.siglip_processor = AutoProcessor.from_pretrained(
            self.model_path,
            token=self.hf_token  # Changed from use_auth_token
        )
    
    def train(self, player_crops):
        """Train classifier on player crops."""
        self.classifier = TeamClassifier(device=self.device)
        self.classifier.fit(player_crops)
        print(f"Team classifier trained on {len(player_crops)} crops")
    
    def predict(self, player_crops):
        """Predict team assignments."""
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        return self.classifier.predict(player_crops)
