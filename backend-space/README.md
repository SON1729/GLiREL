---
title: GLiREL Trace API
sdk: docker
app_port: 7860
---

# GLiREL trace backend

Backend của ứng dụng Streamlit GLiREL Algorithm Workbench.
Mã GLiREL: https://github.com/jackboyla/GLiREL/tree/1f485a268c2144cdc0221bcf0b24152f93c0a155

Upload các file sau vào root của Docker Space:
`Dockerfile`, `service.py`, `inference.py`, `core.py`, `requirements-model.txt`,
`requirements.txt` và file này với tên **README.md**.

Đặt secret `GLIREL_API_KEY` trong Space Settings. Dùng cùng giá trị ở Streamlit.
Không cần HF token để tải model công khai. Model được tải khi có lượt đầu tiên.

`GET /health`: trạng thái server và cờ model_loaded.
`POST /predict`: JSON `{"example":{"name":"...","tokens":[...],"entities":[[0,0,"ENTITY"],[3,3,"ENTITY"]],"labels":["employer"],"expected":[]}}`.
Header `Authorization: Bearer <GLIREL_API_KEY>` nếu đã đặt secret.

Đây là API suy luận, không phải giao diện thuyết trình; phần giao diện vẫn chạy
trên Streamlit. Tạo Docker Space chạy compute hiện cần gói tài khoản HF phù hợp.
Tham khảo https://huggingface.co/docs/hub/spaces-overview trước khi triển khai.
