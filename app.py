"""Streamlit host for the sentence-to-trace algorithm workbench."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from core import validate_bundle, validate_example

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='GLiREL · Algorithm Workbench', page_icon='🔎', layout='wide')
st.markdown('''<style>
.stApp{background:#f5f6f8}.block-container{padding:0!important;max-width:none}
[data-testid="stHeader"]{display:none}footer{display:none}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]{gap:0}
</style>''', unsafe_allow_html=True)

def setting(name, fallback=''):
    try:
        return os.environ.get(name) or st.secrets.get(name, fallback)
    except st.errors.StreamlitSecretNotFoundError:
        return os.environ.get(name, fallback)

def source_bundle():
    return validate_bundle(json.loads((ROOT/'data/recorded.json').read_text(encoding='utf-8')))

@st.cache_resource(show_spinner=False)
def local_model(checkpoint, device):
    from inference import load_model
    return load_model(checkpoint, device)

local_enabled = os.environ.get('GLIREL_LOCAL_INFERENCE') == '1'
local = local_enabled and importlib.util.find_spec('glirel') is not None
backend = ('local' if local else 'local_setup') if local_enabled else 'recorded'
saved = source_bundle()
current = st.session_state.get('trace_bundle', saved)
version = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
viewer = components.declare_component('glirel_algorithm_workbench', path=str(ROOT/'frontend'))
request = viewer(bundle=current, library=saved, version=version, backend=backend,
                 response=st.session_state.get('reply'), key='workbench', default=None)

if isinstance(request, dict) and request.get('action') == 'predict':
    request_id = request.get('id')
    if isinstance(request_id, str) and request_id != st.session_state.get('processed_request'):
        st.session_state.processed_request = request_id
        try:
            example = validate_example(request['example'])
            if len(example['tokens']) > 256 or len(example['entities']) > 10 or len(example['labels']) > 12:
                raise ValueError('Mỗi lượt nhận tối đa 256 token, 10 thực thể và 12 nhãn.')
            if backend == 'local':
                from inference import predict, model_config
                checkpoint = setting('GLIREL_CHECKPOINT', 'jackboyla/glirel-large-v0')
                model = local_model(checkpoint, setting('GLIREL_DEVICE', 'cpu'))
                result = dict(checkpoint=checkpoint, config=model_config(model), cases=[predict(model, example)])
            else:
                raise ValueError('Chạy câu mới bằng bản máy cá nhân: cài requirements-model.txt rồi mở start_local_model.bat. Bản web chỉ xem các trace, không chạy model online.')
            result = validate_bundle(result)
            returned = result['cases'][0]['example']
            if len(result['cases']) != 1 or any(returned.get(k) != example[k] for k in ('tokens', 'entities', 'labels')):
                raise ValueError('Trace trả về không khớp đầu vào. Không hiển thị điểm của câu khác.')
            st.session_state.trace_bundle = result
            st.session_state.reply = dict(id=request_id, status='ok')
        except Exception as exc:
            st.session_state.reply = dict(id=request_id, status='error', message=str(exc))
        st.rerun()
