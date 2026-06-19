import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DataPreprocessor:
    """
    A class used to preprocess input data by adding positional encodings to the input embeddings.
    
    Attributes:
    ----------
    d_model : int
        The dimensionality of the input embeddings.
    max_len : int
        The maximum length of the input sequences.
    embedding : nn.Embedding
        The input embeddings.
    pe : torch.Tensor
        The positional encodings.
    """

    def __init__(self, d_model: int, max_len: int, vocab_size: int):
        """
        Initializes the DataPreprocessor with the given parameters.
        
        Parameters:
        ----------
        d_model : int
            The dimensionality of the input embeddings.
        max_len : int
            The maximum length of the input sequences.
        vocab_size : int
            The size of the vocabulary.
        """
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = self.positional_encoding(max_len, d_model)

    def positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Calculates the positional encodings for the given maximum length and dimensionality.
        
        Parameters:
        ----------
        max_len : int
            The maximum length of the input sequences.
        d_model : int
            The dimensionality of the input embeddings.
        
        Returns:
        -------
        torch.Tensor
            The positional encodings.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def add_positional_encoding(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Adds the positional encodings to the input embeddings.
        
        Parameters:
        ----------
        input_embeddings : torch.Tensor
            The input embeddings.
        
        Returns:
        -------
        torch.Tensor
            The input embeddings with positional encodings.
        """
        return input_embeddings + self.pe[:input_embeddings.size(0), :]

    def normalize_embeddings(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input embeddings.
        
        Parameters:
        ----------
        input_embeddings : torch.Tensor
            The input embeddings.
        
        Returns:
        -------
        torch.Tensor
            The normalized input embeddings.
        """
        return F.normalize(input_embeddings, p=2, dim=-1)

    def preprocess(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input data by adding positional encodings to the input embeddings and normalizing the embeddings.
        
        Parameters:
        ----------
        input_ids : torch.Tensor
            The input ids.
        
        Returns:
        -------
        torch.Tensor
            The preprocessed input embeddings.
        """
        input_embeddings = self.embedding(input_ids)
        input_embeddings = self.add_positional_encoding(input_embeddings)
        input_embeddings = self.normalize_embeddings(input_embeddings)
        return input_embeddings


if __name__ == "__main__":
    preprocessor = DataPreprocessor(d_model=512, max_len=100, vocab_size=1000)
    input_ids = torch.randint(0, 1000, (10, 20))
    preprocessed_embeddings = preprocessor.preprocess(input_ids)
    print(preprocessed_embeddings.shape)