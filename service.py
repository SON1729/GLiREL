"""Optional GLiREL backend for a machine with enough RAM."""
import hmac
import os
import threading
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException
from core import validate_example

app = FastAPI(title='GLiREL trace service', docs_url=None, redoc_url=None)
BUSY = threading.Lock()
CHECKPOINT = os.environ.get('GLIREL_CHECKPOINT', 'jackboyla/glirel-large-v0')

@lru_cache(maxsize=1)
def get_model():
    from inference import load_model
    return load_model(CHECKPOINT, os.environ.get('GLIREL_DEVICE', 'cpu'))

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': bool(get_model.cache_info().currsize)}

@app.post('/predict')
def predict_request(payload: dict, authorization: str = Header(default='')):
    secret = os.environ.get('GLIREL_API_KEY', '')
    if secret and not hmac.compare_digest(authorization, 'Bearer ' + secret):
        raise HTTPException(401, 'API key không đúng.')
    if not BUSY.acquire(blocking=False):
        raise HTTPException(429, 'Model đang xử lý câu khác. Thử lại sau ít giây.')
    try:
        example = validate_example(payload['example'])
        if len(example['tokens']) > 256 or len(example['entities']) > 10 or len(example['labels']) > 12:
            raise ValueError('Giới hạn: 256 token, 10 thực thể, 12 nhãn.')
        from inference import predict, model_config
        model = get_model()
        return dict(checkpoint=CHECKPOINT, config=model_config(model), cases=[predict(model, example)])
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        BUSY.release()
