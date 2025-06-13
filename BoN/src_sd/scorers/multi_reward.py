import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from scorers import AestheticScorer,CompressibilityScorer,FaceRecognitionScorer,ClipScorer,ImageRewardScorer,PickScoreScorer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class MultiReward:
    def __init__(self, sc1, sc2, scorer_weight_1=1, scorer_weight_2=1,device=device):
        self.scorer_names = [sc1, sc2]
        self.scorer_weight_1 = scorer_weight_1
        self.scorer_weight_2 = scorer_weight_2
        self.scorer1 = self._init_scorer(sc1, device)
        self.scorer2 = self._init_scorer(sc2, device)

    def _init_scorer(self, name, device):
        from scorers import ClipTextScorer
        scorer_map = {
            "aesthetic": AestheticScorer,
            "compress": CompressibilityScorer,
            "facedetector": FaceRecognitionScorer,
            "styletransfer": ClipScorer,
            "strokegen": ClipScorer,
            "clip": ClipTextScorer,
            "imagereward": ImageRewardScorer,
            "pickscore": PickScoreScorer,
        }

        if name not in scorer_map:
            raise ValueError(f"Unknown scorer type: {name}")

        ScorerClass = scorer_map[name]

        # Only pass device if the scorer requires it
        if name in ["imagereward","pickscore"]:
            return ScorerClass(device=device)
        else:
            return ScorerClass()
        
    def score(self,image,prompt,return_all=False):
        score1 = self._score_with_scorer(self.scorer1, self.scorer_names[0], image, prompt)
        score2 = self._score_with_scorer(self.scorer2, self.scorer_names[1], image, prompt)
        # print(f"{self.scorer_names[0]} score is {score1}")
        # print(f"{self.scorer_names[1]} score is {score2}")
        if return_all:
            return self.scorer_weight_1 * score1.cpu() + self.scorer_weight_2 * score2.cpu(),score1.cpu(), score2.cpu()
        return self.scorer_weight_1 * score1.cpu() + self.scorer_weight_2 * score2.cpu()
    
    def _score_with_scorer(self, scorer, name, image, prompt):
        if name in ["aesthetic", "compress"]:
            return scorer.score(image)
        else:
            return scorer.score(image, prompt)
        
    def loss_fn(self,image,prompt):
        loss1 = self._loss(self.scorer1, self.scorer_names[0], image, prompt)
        loss2 = self._loss(self.scorer2, self.scorer_names[1], image, prompt)
        return self.scorer_weight_1*loss1.cpu() + self.scorer_weight_2 * loss2.cpu()
    
    def _loss(self, scorer, name, image, prompt):
        if name in ["aesthetic", "compress"]:
            return scorer.loss_fn(image)
        else:
            return scorer.loss_fn(image, prompt)