"""Streamlit host + bundled animated component. No model needed for recorded mode."""
from pathlib import Path
import hashlib
import json
import streamlit as st
import streamlit.components.v1 as components
from core import validate_bundle, validate_example

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='GLiREL · Visual Lab',page_icon='🔎',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:#f7f9fa} .block-container{padding:0 0 1rem;max-width:1400px}
[data-testid="stHeader"]{background:transparent;height:2rem}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]{gap:0}
footer{visibility:hidden} [data-testid="stExpander"]{margin:0 28px 12px;border-color:#dce4e6}
</style>''',unsafe_allow_html=True)

bundle=validate_bundle(json.loads((ROOT/'data/recorded.json').read_text(encoding='utf-8')))
builtin=True

with st.expander('⚙ Chạy câu mới bằng checkpoint / nhập trace / hướng dẫn chia sẻ',expanded=False):
    mode=st.radio('Nguồn dữ liệu',['Bản ghi có sẵn','Nhập JSON từ Kaggle','Chạy checkpoint'],horizontal=True)
    if mode=='Nhập JSON từ Kaggle':
        upload=st.file_uploader('demo_predictions.json hoặc trace được xuất từ giao diện',type=['json'])
        if upload:
            try:
                if upload.size>8_000_000:raise ValueError('Giới hạn 8 MB.')
                bundle=validate_bundle(json.loads(upload.getvalue()))
                builtin=False
            except (ValueError,KeyError,TypeError,IndexError) as exc:st.error('Trace không hợp lệ: '+str(exc));st.stop()
    elif mode=='Chạy checkpoint':
        st.caption('Chỉ nút Chạy mới nạp model; giao diện ghi sẵn không cần Torch. Xem README để cài requirements-model.txt. Các file phải ở trên máy/server đang chạy Streamlit.')
        checkpoint=st.text_input('Thư mục checkpoint hoặc ID chính chủ','jackboyla/glirel-large-v0')
        device=st.selectbox('Thiết bị',['cpu','cuda'])
        i=st.selectbox('Bắt đầu từ ví dụ',range(len(bundle['cases'])),format_func=lambda i:bundle['cases'][i]['example']['name'])
        e=bundle['cases'][i]['example']
        with st.form('new-inference'):
            sentence=st.text_area('Câu tiếng Anh đã tách token bằng khoảng trắng',' '.join(e['tokens']))
            labels=st.text_area('Mỗi nhãn một dòng','\n'.join(e['labels']))
            entities=st.text_area('Thực thể: [[start, end, "type"], ...] — end được bao gồm',json.dumps(e['entities']))
            gold=st.text_area('Gold tùy chọn: [[head_index, "relation", tail_index], ...]',json.dumps(e.get('expected',[])))
            run=st.form_submit_button('Chạy GLiREL và mở trace mới',type='primary')
        if run:
            try:
                example=validate_example({'name':'Câu do bạn nhập','tokens':sentence.split(),
                    'entities':json.loads(entities),'labels':[s.strip() for s in labels.splitlines() if s.strip()],
                    'expected':json.loads(gold)})
                from inference import load_model,predict
                cached=st.cache_resource(show_spinner=False)(load_model)
                with st.spinner('Đang nạp mô hình / chạy inference...'):
                    model=cached(checkpoint,device)
                    result=predict(model,example)
                cfg={k:getattr(model.config,k,None) for k in ['model_name','hidden_size','subtoken_pooling','scorer','label_embed_strategy','span_marker_mode','rel_mode','refine_prompt','refine_relation']}
                st.session_state.live_bundle={'checkpoint':checkpoint,'config':cfg,'cases':[result]}
                st.success('Đã chạy. Giao diện phía dưới đang dùng trace mới.')
            except Exception as exc:st.error(f'{type(exc).__name__}: {exc}')
        if st.session_state.get('live_bundle'):
            bundle=validate_bundle(st.session_state.live_bundle);builtin=False
    st.markdown('**Chia sẻ:** đưa cả thư mục lên GitHub → Streamlit Community Cloud → Create app → chọn `app.py` → Deploy. Sau đó gửi đường link app cho người khác. Bản ghi mẫu nhẹ, không cần GPU. Máy khác không mở được địa chỉ localhost của bạn.')
    st.download_button('Tải trace đang dùng',json.dumps(bundle,ensure_ascii=False,indent=2),'glirel_trace.json','application/json')

viewer=components.declare_component('glirel_visual_story',path=str(ROOT/'frontend'))
version=hashlib.sha256(json.dumps(bundle,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
viewer(bundle=bundle,builtin=builtin,version=version,key='story')
