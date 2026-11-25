# GLiREL · Algorithm Workbench

Demo Streamlit bằng tiếng Việt, tập trung vào việc chạy tay pipeline GLiREL.
Giao diện bắt đầu ngay bằng câu, token, thực thể, nhãn và các bước xử lý; không có
hero/banner giới thiệu. Xanh dương = thao tác/bước hiện tại, màu thực thể nhất quán
qua các hình, tím = nhãn quan hệ; giữ/loại còn có chữ để không chỉ dựa vào màu.

## Mở và dùng

Bấm `start_windows.bat`, hoặc `python run_demo.py`, rồi mở http://127.0.0.1:8502.
Chỉ `requirements.txt` là đủ để xem các bản ghi; không tự tải trọng số.

1. Chọn ví dụ đã chạy, hoặc nhập câu tiếng Anh bất kỳ.
2. Bấm **Tách từ để chọn thực thể**. Bấm token đầu rồi token cuối, sau đó
   **Thêm thực thể**. Tên một từ: bấm cùng token hai lần. Chọn ít nhất hai thực thể.
3. Thêm/xóa nhãn bằng tên tiếng Anh, ví dụ `place of birth`, `employer`.
4. Bấm **Chạy câu này** để lấy trace mới. Với ví dụ sẵn có, **Phát bản ghi này**
   diễn lại các bước bằng điểm đã ghi, không gọi model.
5. Bấm bước 1–8, phát/tạm dừng, trước/tiếp và chọn tốc độ. Mỗi bước có dữ liệu
   minh họa ở giữa, đầu vào/xử lý/đầu ra và giải thích ở cạnh.
6. Bấm ô ma trận ở bước 7 để đọc cặp, nhãn, điểm, logit suy ngược và lý do giữ/loại.
   Đổi ngưỡng/top-k không chạy lại encoder.
7. **Kết quả & lỗi** cho gán đáp án và soi TP/FP/FN. Chưa gán đáp án thì không
   tự coi dự đoán là lỗi. **Lưu để so sánh** giữ tối đa 3 lần chạy/policy trong phiên.
8. **Thử phép tính** dùng vector 2D minh họa để hiểu dot product/sigmoid và N(N−1)M.

Câu mới giới hạn 256 token, 10 thực thể không chồng nhau và 12 nhãn. Tên nhãn
không rỗng, không lặp. Tokenization của giao diện tách dấu câu; không phải NER.
Sửa câu/nhãn/span sẽ loại toàn bộ điểm cũ khỏi view. Chưa chạy encoder thật thì
không tạo ID/vector/điểm giả. Với câu đang gõ, phải tách từ lại trước khi chạy.

## Chạy câu mới trên web: cần nối model

**Streamlit Cloud hiện chỉ cài bộ giao diện nhẹ. Có ô nhập câu không đồng nghĩa
model lớn đã được tải trên Cloud.** Cần một trong hai cách sau:

### A. Backend riêng (phù hợp với trang chia sẻ)

Repo có `service.py`, `transport.py`, `Dockerfile` để tách model khỏi giao diện.
Backend nhận câu, span, nhãn và trả về trace thật. Frontend giữ nguyên Streamlit.

- Chạy Dockerfile trên máy chủ đủ RAM; hoặc tạo Docker Space trên Hugging Face.
  Backend mặc định dùng `jackboyla/glirel-large-v0`, ghim revision
  `02e48d6f3fa5fe88953500abc4a37a5428b77af8`, code GLiREL commit `1f485a2`.
- Với Hugging Face, giải nén gói backend rồi upload nội dung vào Docker Space.
  README trong gói đã có metadata `sdk: docker`, `app_port: 7860`.
- Theo tài liệu HF hiện tại, tạo Docker Space chạy compute yêu cầu gói tài khoản
  phù hợp; không mặc định đây là dịch vụ miễn phí. Không có tài nguyên trả phí nào
  được tạo tự động bởi mã này. Có thể dùng máy chủ Docker sẵn có thay thế.
- Đặt secret `GLIREL_API_KEY` cho backend (một chuỗi do bạn tự chọn). API key của
  dịch vụ này khác HF token. Không đưa secret vào GitHub.
- Đợi `/health` trả JSON. `model_loaded: false` vẫn bình thường trước lượt đầu;
  lần đầu suy luận cần tải model. Nếu timeout, đợi backend nạp xong rồi thử lại.
- Vào Streamlit Cloud → app → Settings → Secrets, thêm:

```toml
GLIREL_API_URL = "https://YOUR-SPACE.hf.space"
GLIREL_API_KEY = "cung-chuoi-da-dat-o-backend"
```

URL là tên miền chạy thật của backend (lấy từ Space), không phải đường dẫn trang
`huggingface.co/spaces/...`. Không thêm `/predict` vì app tự nối. Reboot app nếu cần.
API key chỉ đi qua Python server, không truyền xuống component trình duyệt.
Backend xử lý một câu mỗi lần để tránh trộn trace hoặc hết RAM; lúc bận trả 429.

Có thể kiểm tra Docker trên máy riêng:

```bash
docker build -t glirel-api .
docker run --rm -p 7860:7860 -e GLIREL_API_KEY=your-key glirel-api
```

Nếu Streamlit cũng chạy trên máy đó, dùng `GLIREL_API_URL=http://127.0.0.1:7860`.
Một app Cloud không truy cập được localhost trên máy cá nhân của bạn.

### B. Chạy model ngay trên máy chạy Streamlit

Dùng môi trường riêng, khuyến nghị Python 3.12, có Git và đủ RAM/ổ đĩa:

```powershell
python -m pip install -r requirements-model.txt
python run_demo.py
```

App tự phát hiện `glirel`. Model chỉ nạp khi bạn bấm chạy câu mới. Có thể đặt:

```powershell
$env:GLIREL_CHECKPOINT = "D:\path\to\model_6600"
$env:GLIREL_DEVICE = "cpu"
python run_demo.py
```

Thư mục checkpoint cần `glirel_config.json` và `pytorch_model.bin`. Nếu không đặt,
app dùng model chính chủ. CPU có thể chậm. GPU dùng `cuda` khi đã cài Torch/CUDA đúng.
Mặc định không thêm bộ Torch/model lớn vào requirements Cloud vì có thể vượt tài
nguyên của nơi triển khai. Docker image/backend thực với trọng số chưa được chạy
end-to-end trên máy thực hiện thay đổi; các tests backend dùng fixture ghi rõ.

## Cập nhật web

Push lên nhánh `main` của GitHub repo đã kết nối. Streamlit Cloud nhận thay đổi.
Nếu CSS/JS còn bản cũ, tải lại trang; có thể Reboot app trong Cloud. URL giữ nguyên.
Các file frontend, data và `.streamlit/config.toml` phải giữ đúng cấu trúc.

## Nguồn dữ liệu và giới hạn diễn giải

`data/recorded.json` là trace `model_6600` do người dùng cung cấp từ thí nghiệm
trước, không phải checkpoint paper gốc hay các lượt synthetic mới. ID/subtoken,
điểm và tensor summary ở các ví dụ này là bản ghi thật. Mỗi tensor summary chỉ có
shape và vài tọa độ đầu của **toàn tensor**; không gán nó cho cặp đang được chọn.
Các hình khối và animation biểu diễn thứ tự/kiểu dữ liệu, không phải attention,
vector đầy đủ, thời gian thực đo hay giải thích nhân quả.

- Chỉ số span input bao gồm end; output API loại trừ end.
- Chọn `score > threshold`, rồi top-k riêng cho từng cặp có hướng.
- Logit ở ô điểm là `log(s/(1−s))` suy ngược từ điểm, không giả là tensor gốc đã ghi.
- Số cặp N(N−1) là tối đa trước giới hạn khoảng cách/mask của checkpoint.
- Gold là đáp án ví dụ/đáp án do người xem gán, không tự thành một benchmark.
- Muốn kiểm tra phủ định, giữ cùng checkpoint/span/nhãn/policy và chỉ sửa một yếu tố
  rồi chạy lại. Đổi tên nhãn cũng phải chạy encoder vì nhãn và câu được joint-encode.
- Cấu hình minh họa tương ứng nhánh dot/both/marker của các bản ghi đã cung cấp.
  Refinement và span mode tùy checkpoint; không tự dựng tensor của module vắng mặt.

Nhập/xuất JSON trong **Dữ liệu & kết nối**. Hỗ trợ schema `spans/gold` và
`entities/expected`. Lưu so sánh ở bộ nhớ phiên; reload trang có thể mất các bản lưu.
Không có cơ sở dữ liệu câu nhập được thêm trong app này.

## Kiểm tra

```powershell
python -m unittest discover -s tests -v
node tests/test_workbench.cjs
```

Bộ nhẹ chạy tests core/host/transport. Tests service cần FastAPI/httpx. Tests
kiểm tra khởi tạo không tải model, tokenization, chọn span, xóa điểm cũ khi sửa,
request/reply, lỗi backend, ngưỡng/top-k, phát đến bước cuối, so sánh, gold và escape.
Không coi fixture hợp đồng HTTP là lượt suy luận thực với trọng số.

## Tài liệu

- [GLiREL paper](https://arxiv.org/abs/2501.03172)
- [GLiREL code ghim commit](https://github.com/jackboyla/GLiREL/tree/1f485a268c2144cdc0221bcf0b24152f93c0a155)
- [Streamlit custom components](https://docs.streamlit.io/develop/concepts/custom-components)
- [HF Spaces overview và điều kiện tài khoản](https://huggingface.co/docs/hub/spaces-overview)
- [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
