import torch
from transformers import HubertModel, Wav2Vec2Processor
from .averaging_classifier import AverageModel
from .pattern_classifier import PatternModel

def load_barge_in_models(average_model=True, local=False, model_weights_path = ''):
    """
    Loads and returns either the average-based or pattern-based barge-in detection models alongside embeddings processor and model.

    :param average_model: If set to True, loads the AverageModel; otherwise, loads the PatternModel. Default is True.
    :returns: A tuple containing the loaded Hubert model, Wav2Vec2 processor, and the selected classifier model.
    """
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

    if average_model:
        classifier_model = AverageModel()
        if not local:
            classifier_model.load_state_dict(torch.load('/app/barge_in_model/resources/average_model.pth'))
        else:
            classifier_model.load_state_dict(torch.load(model_weights_path))
        classifier_model.eval()
    else:
        classifier_model = PatternModel()
        if not local:
            classifier_model.load_state_dict(torch.load('/app/barge_in_model/resources/pattern_model.pth'))
        else:
            classifier_model.load_state_dict(torch.load(model_weights_path))
        classifier_model.eval()

    return hubert_model, wav2vec2_processor, classifier_model