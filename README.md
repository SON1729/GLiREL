# GLiREL · Algorithm Workbench

Demo Streamlit bằng tiếng Việt, tập trung vào việc chạy tay pipeline GLiREL.
Giao diện bắt đầu ngay bằng câu, token, thực thể, nhãn và các bước xử lý; không có
hero/banner giới thiệu. Canvas navy tương phản cao; cyan cho văn bản, tím cho
nhãn, amber cho cặp và xanh lá cho kết quả. Giữ/loại có chữ để không chỉ dựa vào màu.

## Mở và dùng

Bấm `start_windows.bat`, hoặc `python run_demo.py`, rồi mở http://127.0.0.1:8502.
Chỉ `requirements.txt` là đủ để xem các bản ghi; không tự tải trọng số.

1. Bản web chọn ví dụ đã chạy/nhập trace. Bản local có thêm nhập câu tiếng Anh mới.
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

## Hai cách dùng riêng

| Cách mở | Chức năng | Model chạy ở đâu? |
|---|---|---|
| Link Streamlit Cloud | Xem ví dụ/trace, pipeline, ngưỡng, ma trận, so sánh | Không nạp model |
| `start_local_model.bat` | Các chức năng trên + nhập câu mới và suy luận | Máy của người mở bản local |

Không cần backend online, Hugging Face Spaces hay tunnel. Mỗi người muốn chạy
câu mới tự cài bản local. Có thể xuất JSON từ local rồi nhập vào web để trình bày.
Mặc định chế độ web để câu/span/nhãn ở trạng thái chỉ đọc; vẫn chọn ví dụ khác,
nhập trace, điều chỉnh ngưỡng, xem từng bước và đối chiếu được.

### Chạy câu mới trên Windows

Yêu cầu Python (khuyến nghị 3.12), Git, RAM/ổ đĩa đủ cho model. Mở terminal trong
thư mục repo đã tải từ GitHub, thực hiện **một lần** trong môi trường riêng:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-model.txt
```

Sau đó bấm **start_local_model.bat** và mở **http://127.0.0.1:8503**.
Hoặc chạy trực tiếp:

```powershell
.\.venv\Scripts\python run_demo.py --local-model --port 8503
```

Lần đầu chạy câu mới sẽ tải model công khai `jackboyla/glirel-large-v0`, ghim
revision `02e48d6f3fa5fe88953500abc4a37a5428b77af8`. Không cần tài khoản HF để tải
model công khai này. Chỉ bấm nút chạy mới nạp model; xem trace không tải model.
CPU là mặc định. Nếu bạn có checkpoint riêng:

```powershell
$env:GLIREL_CHECKPOINT = "D:\path\to\model_6600"
.\.venv\Scripts\python run_demo.py --local-model --port 8503
```

Thư mục checkpoint cần `glirel_config.json` và `pytorch_model.bin`. Mỗi người dùng
trọng số của chính họ hoặc model công khai; model không được chứa trong ZIP/GitHub.
Terminal phải còn chạy để dùng bản local. Ctrl+C để dừng. Launcher chỉ bind
127.0.0.1, không công khai server máy cá nhân. Không có việc cài Torch hoặc tải
trọng số tự động trong lần cập nhật giao diện này; người dùng chủ động cài bộ model.

`start_windows.bat` vẫn mở chế độ xem trace nhẹ tại cổng 8502. Biến môi trường
`GLIREL_LOCAL_INFERENCE=1` chỉ được launcher local bật. Bản Cloud mặc định không
nhận yêu cầu suy luận kể cả thư viện model có mặt hoặc còn secrets backend cũ.
Các file Docker/service cũ trong repo không được app chính gọi.

## Cập nhật web

Push lên nhánh `main` của GitHub repo đã kết nối. Streamlit Cloud nhận thay đổi.
Nếu CSS/JS còn bản cũ, tải lại trang; có thể Reboot app trong Cloud. URL giữ nguyên.
Các file frontend, data và `.streamlit/config.toml` phải giữ đúng cấu trúc.

## Độ chính xác pipeline

Xem [bảng đối chiếu từng module](PIPELINE_AUDIT.md), gồm công thức, nguồn code,
kết quả kiểm tra 28 logit và cấu hình checkpoint còn thiếu. Không mặc định
span mode/refinement/scorer khi trace không chứa đủ thông tin.

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
