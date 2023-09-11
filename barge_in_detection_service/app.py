# Run app with: hypercorn app:app --keep-alive 10000 --bind 0.0.0.0:8080

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from barge_in_model.barge_in_inference_pipeline import check_barge_in, BargeInClassificationError
from barge_in_model.barge_in_model_loader import load_barge_in_models
import base64
from pydub import AudioSegment
import io
import os

app = FastAPI()
audio_cache = {}

# load in preprocessors and models as global variables, always in memory
AVERAGE = False # whether to use the AverageModel or the PatternModel
THRESHOLD = 0.5 # threshold for which we must exceed to deem a Barge-in
LOCAL = False # whether we are running the model in Docker or locally
LOCAL_AVERAGE_MODEL_WEIGHTS_PATH = '/Users/daniel/Desktop/DO voicebot/barge_in_detection_service/barge_in_model/resources/average_model.pth' # edit path here if running locally
LOCAL_PATTERN_MODEL_WEIGHTS_PATH = '/Users/daniel/Desktop/DO voicebot/barge_in_detection_service/barge_in_model/resources/pattern_model.pth' # edit path here if running locally

if AVERAGE:
    LOCAL_MODEL_WEIGHTS_PATH = LOCAL_AVERAGE_MODEL_WEIGHTS_PATH
else:
    LOCAL_MODEL_WEIGHTS_PATH = LOCAL_PATTERN_MODEL_WEIGHTS_PATH

embedding_model, embedding_processor, classifier_model = load_barge_in_models(average_model=AVERAGE, local=LOCAL, model_weights_path = LOCAL_MODEL_WEIGHTS_PATH)

@app.post("/query-interrupt")
def handle_query_interrupt(data: dict):
    """
    Handles the query to check for interruptions in the provided audio data.

    This endpoint expects the incoming data to contain the base64 encoded audio data and its associated ID.
    First audio data is processed and then we check for interruptions using the relevant barge-in models, and responds accordingly.
    The audio data associated with the given ID is cached to allow for continuous streaming and analysis.
    Once an interruption is detected, the cache is cleared for that ID.

    :param data: Dictionary containing the 'audio' (with 'data' as its base64 encoded audio content) and 'id'.
    :returns: JSONResponse indicating whether an interruption is detected or not, along with the provided 'id'.
    :raises HTTPException: If the provided data format is not as expected (status code 400), or there is an error in the barge-in classification (status code 500).
    """
    
    if 'audio' not in data or 'id' not in data or 'data' not in data['audio']:
        raise HTTPException(status_code=400, detail="Invalid data format.")

    audio = data['audio']['data']

    if audio is None or len(audio) == 0:
        raise HTTPException(status_code=400, detail="Audio data cannot be None or of length 0.")

    try:
        base64.b64decode(audio, validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data.")

    print('Audio snippet received: ', audio[:30], ' with ID: ', data['id'])
    wav_data = base64.b64decode(audio)
    id = data['id']

    audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))

    if id not in audio_cache:
        audio_cache.clear() 
        audio_cache[id] = audio_segment
    else:
        audio_cache[id] += audio_segment

    # Testing Only
    # output_path = "audio_cache/"
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # file_name = f"{output_path}{id}.wav"
    # audio_cache[id].export(file_name, format="wav")

    try:
        is_barge_in = check_barge_in(audio_cache[id], embedding_model, embedding_processor, classifier_model, average_embeddings=AVERAGE, threshold=THRESHOLD)
        
        if is_barge_in:
            del audio_cache[id]
            print('SENDING RESPONSE DATA')
            return JSONResponse(content={'isInterrupt': True, 'id': id})

        return JSONResponse(content={'isInterrupt': False, 'id': id})
    except BargeInClassificationError as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model processing failed. {str(e)}")

if __name__ == '__main__':
    import hypercorn.asyncio
    import hypercorn.config

    config = hypercorn.config.Config()
    config.bind = ["0.0.0.0:8080"]  # binds the app to all available network interfaces on port 8080
    config.http = "h2"  # set protocol to HTTP 2

    hypercorn.asyncio.serve(app, config)