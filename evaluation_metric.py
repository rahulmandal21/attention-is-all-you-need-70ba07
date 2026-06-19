import torch
import torch.nn as nn
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from typing import List, Tuple

class BLEUEvaluationMetric:
    """
    A class used to evaluate the quality of generated translations using the BLEU score.
    """

    def __init__(self):
        """
        Initializes the BLEUEvaluationMetric class.
        """
        pass

    def calculate_bleu_score(self, predictions: List[List[int]], references: List[List[List[int]]]) -> float:
        """
        Calculates the BLEU score for a list of predicted translations and their corresponding references.

        Args:
        predictions (List[List[int]]): A list of predicted translations, where each translation is a list of word indices.
        references (List[List[List[int]]]): A list of references, where each reference is a list of lists of word indices.

        Returns:
        float: The BLEU score.
        """
        return corpus_bleu(references, predictions)

    def calculate_sentence_bleu_score(self, prediction: List[int], references: List[List[int]]) -> float:
        """
        Calculates the BLEU score for a single predicted translation and its corresponding references.

        Args:
        prediction (List[int]): A predicted translation, where the translation is a list of word indices.
        references (List[List[int]]): A list of references, where each reference is a list of word indices.

        Returns:
        float: The BLEU score.
        """
        return sentence_bleu(references, prediction)

    def get_predicted_translations(self, logits: torch.Tensor) -> List[List[int]]:
        """
        Gets the predicted translations from a tensor of logits.

        Args:
        logits (torch.Tensor): A tensor of logits, where each logit corresponds to a word index.

        Returns:
        List[List[int]]: A list of predicted translations, where each translation is a list of word indices.
        """
        return torch.argmax(logits, dim=-1).tolist()

    def convert_to_tensors(self, predictions: List[List[int]]) -> List[torch.Tensor]:
        """
        Converts a list of predicted translations to tensors.

        Args:
        predictions (List[List[int]]): A list of predicted translations, where each translation is a list of word indices.

        Returns:
        List[torch.Tensor]: A list of tensors, where each tensor corresponds to a predicted translation.
        """
        return [torch.tensor(prediction) for prediction in predictions]

if __name__ == "__main__":
    evaluation_metric = BLEUEvaluationMetric()
    predictions = [[1, 2, 3], [4, 5, 6]]
    references = [[[1, 2, 3], [1, 2, 4]], [[4, 5, 6], [4, 5, 7]]]
    bleu_score = evaluation_metric.calculate_bleu_score(predictions, references)
    print(f"BLEU score: {bleu_score}")
    sentence_bleu_score = evaluation_metric.calculate_sentence_bleu_score(predictions[0], references[0])
    print(f"Sentence BLEU score: {sentence_bleu_score}")
    logits = torch.randn(2, 10)
    predicted_translations = evaluation_metric.get_predicted_translations(logits)
    print(f"Predicted translations: {predicted_translations}")
    tensors = evaluation_metric.convert_to_tensors(predicted_translations)
    print(f"Tensors: {tensors}")