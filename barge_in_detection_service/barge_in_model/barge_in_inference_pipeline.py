import torch
import numpy as np

class BargeInClassificationError(Exception):
    """Base exception for errors related to barge-in classification."""
    pass

class EmbeddingModelError(BargeInClassificationError):
    """Exception raised when there's an error in the embedding model processing."""
    pass

class BargeInModelError(BargeInClassificationError):
    """Exception raised when there's an error in the barge-in classifier model processing."""
    pass

def embedding_preprocess(audio_segment, processor):
    """
    Converts audio to mono if stereo and uses provided processor to process audio.

    :param audio_segment: An AudioSegment object from the pydub library.
    :param processor: Processor function to convert audio data to embeddings.
    :returns: Processed inputs suitable for embedding model.
    """
    # expects an AudioSegment object from pydub library
    if audio_segment.channels == 2:
        audio_segment = audio_segment.set_channels(1)
        print('Audio was stereo')
    speech_array = np.array(audio_segment.get_array_of_samples()).astype(np.float64) # bug fix, introduced astype because hubert expected double not float, this is handled by librosa in training code
    inputs = processor(speech_array, return_tensors="pt", sampling_rate=16000)
    return inputs

def resize_embedding(embedding):
    """
    Resizes the given embedding to a fixed length by truncating or zero-padding on the right hand side.

    :param embedding: Tensor representing the embedding.
    :returns: Embedding tensor of fixed length.
    """
    FIXED_LENGTH = 250 # fixed sequence length that the model expects as an input
    # Truncate or zero-pad all sequences to a fixed length
    print('Original embedding shape: ', embedding.shape[0])
    if embedding.shape[0] > FIXED_LENGTH:
        embedding = embedding[:FIXED_LENGTH, :]
    else:
        padding = torch.zeros((FIXED_LENGTH - embedding.shape[0], embedding.shape[1]))
        embedding = torch.cat((embedding, padding))
    embedding = embedding.unsqueeze(0) # currently we have shape (L, C), we need (N, L, C) which will be later changed to (N, C, L)
    return embedding

def check_barge_in(audio_segment, embedding_model, embedding_processor, classifier_model, average_embeddings=False, threshold=0.5):
    """
    Checks for barged-in audio in the provided audio segment using given models. THRESHOLD is an important 

    :param audio_segment: An AudioSegment object from the pydub library.
    :param embedding_model: Model to get embeddings from audio data.
    :param embedding_processor: Processor function to preprocess audio data.
    :param classifier_model: Model to classify the embeddings for barge-in detection.
    :param average_embeddings: Boolean to decide if embeddings should be averaged or not. Default is False.
    "param threshold: threshold for which we must exceed to deem a Barge-in.
    :returns: Boolean indicating if barge-in is detected or not.
    :raises EmbeddingModelError: Raised when there's an error in the embedding model processing.
    :raises BargeInModelError: Raised when there's an error in the barge-in classifier model processing.
    """
    
    try:
        inputs = embedding_preprocess(audio_segment, embedding_processor)
        with torch.no_grad():
            outputs = embedding_model(inputs.input_values)
        embedding = torch.squeeze(outputs.last_hidden_state)
    except Exception as e:
        raise EmbeddingModelError(f"Error during embedding model processing: {str(e)}") from e

    try:
        if average_embeddings:
            embedding = torch.mean(embedding, dim=0)
            embedding = embedding.unsqueeze(0)
        else:
            embedding = resize_embedding(embedding)
        with torch.no_grad():
            output = classifier_model(embedding).squeeze(1)
        pred = torch.sigmoid(output)
        print('Interruption assessed, confidence: ', pred[0])
        if float(pred[0]) > threshold:
            return True
        return False
    except Exception as e:
        raise BargeInModelError(f"Error during barge-in classifier model processing: {str(e)}") from e