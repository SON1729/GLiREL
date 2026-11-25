# GLiREL · Visual Lab

Ứng dụng **Streamlit bằng tiếng Việt**, có giao diện HTML/CSS/JavaScript được
đóng gói thành một Streamlit component. Người xem dùng qua trình duyệt; không
cần cài model để khám phá các bản ghi có sẵn.

## Mở trên máy của bạn

Giải nén toàn bộ gói, giữ nguyên `frontend/`, `data/` và `.streamlit/`.
Trên Windows, bấm đúp **`start_windows.bat`**. Launcher tạo môi trường `.venv`
riêng, cài Streamlit khi cần rồi mở server. Sau đó vào:

**http://127.0.0.1:8502**

Giữ cửa sổ terminal mở; Ctrl+C để dừng. Nếu đã có Python 3.10–3.13, có thể chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_demo.py
```

Nếu cổng bận, dùng một cổng khác:

```powershell
.\.venv\Scripts\python -m streamlit run app.py --server.port 8503
```

## Chia sẻ qua link cho người khác cùng dùng

`localhost`/`127.0.0.1` chỉ mở được trên máy đang chạy app. Để gửi link mở từ
máy khác, triển khai bản ghi nhẹ lên **Streamlit Community Cloud**:

1. Tạo repository trên GitHub. Upload **nội dung bên trong thư mục này**:
   `app.py`, `core.py`, `inference.py`, `requirements.txt`, toàn bộ `frontend/`,
   `data/recorded.json` và `.streamlit/config.toml`. Có thể upload cả gói mã nguồn
   sau giải nén, trừ `.venv`, cache và trọng số model. Giữ nguyên cấu trúc thư mục.
2. Mở [Streamlit Community Cloud](https://share.streamlit.io/), đăng nhập/kết nối
   GitHub. Chọn **Create app**, repository, branch và **Main file path: `app.py`**.
   Nếu bạn đặt cả thư mục trong repository, chọn đường dẫn tương ứng tới `app.py`.
3. Bấm **Deploy**. Sau khi app chạy, sao chép URL có dạng
   `https://ten-app.streamlit.app` và gửi người khác. Kiểm tra quyền truy cập app;
   thử mở link bằng cửa sổ ẩn danh để chắc người nhận truy cập được.

Hướng dẫn chính thức:
[Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
Gói này **chưa được xuất bản lên tài khoản Cloud của bạn**; địa chỉ 8502 hiện là
bản chạy cục bộ. Nhiều người có thể mở app đã triển khai; thao tác xem/chỉnh
ngưỡng của từng người độc lập. App có thể ngủ khi không dùng và cần khởi động lại
khi mở, tùy dịch vụ hosting.

**Cách chia sẻ phù hợp cho thuyết trình:** triển khai giao diện và trace có sẵn.
Không cần GPU, Torch hay tải vài GB checkpoint. Người nhận vẫn tương tác được
đầy đủ với ví dụ, animation, ma trận điểm, ngưỡng và phần phân tích lỗi. Trace
nhập qua nút trong giao diện minh họa chỉ dùng ở phiên đó, không sửa bộ mẫu chung.

## Lộ trình khám phá

1. **Input → Output:** xem câu có chỉ số token, span thực thể, danh sách nhãn;
   quan sát bộ ba và đồ thị có hướng. Có 5 tình huống từ câu đơn đến phủ định,
   thiếu nhãn và nhiều câu có đại từ. Đồ thị hiển thị tối đa 4 bộ ba; danh sách
   kết quả hiển thị toàn bộ.
2. **Bên trong pipeline:** chọn từng module hoặc bấm **Phát luồng**, **Tạm dừng**,
   **Trước/Tiếp**, **Từ đầu**. Mỗi bước có hình minh họa, đầu vào, phép biến đổi,
   đầu ra và các số đã ghi. Mở chi tiết để xem subtoken/ID và tensor summary.
3. **Vì sao sai?:** đối chiếu TP/FP/FN với gold khai báo; bấm từng ô ma trận để
   xem cặp–nhãn. Kéo ngưỡng hoặc đổi top-k để phân biệt lỗi quyết định với giả
   thuyết về hiểu ngôn ngữ.
4. **Tự tay khám phá:** chỉnh vector nhãn 2D để xem dot product → sigmoid; chỉnh
   số thực thể và nhãn để xem số ô điểm N(N−1)M.
5. **Nguồn & chia sẻ:** cấu hình checkpoint, thuật ngữ, bài báo, mã nguồn.

Animation có hỗ trợ `prefers-reduced-motion`. Bố cục tự đổi trên màn hình hẹp;
đồ thị và bảng nhiều cột có thể cuộn ngang để đọc.

## Các số liệu đến từ đâu?

`data/recorded.json` là `demo_predictions.json` người dùng cung cấp trước đây,
checkpoint khai báo **`model_6600`**, backbone DeBERTa-v3-large. Đây không phải
checkpoint gốc của paper và không phải các lượt synthetic mới. Gold là đáp án
mong đợi của người viết ví dụ, không phải một benchmark được đánh giá độc lập.

- Điểm ứng viên, subtoken, ID, kích thước tensor và vài tọa độ đầu là bản ghi
  thực. Demo không thay điểm để làm đẹp kết quả.
- Animation, lưới màu và vector 2D trong phòng thí nghiệm là **minh họa**, không
  phải attention, embedding đầy đủ hay thời gian xử lý được đo.
- Đổi ngưỡng/top-k chỉ lọc lại điểm đã ghi. Điều kiện là **score > threshold**;
  top-k tính **riêng cho từng cặp có hướng**, không phải top-k toàn câu.
- Đổi câu/nhãn cần chạy checkpoint lấy trace mới. Nhãn cùng đi qua encoder với
  câu, nên không được giả định điểm giữ nguyên khi thay danh sách nhãn.
- Gold rỗng/chưa đầy đủ làm giới hạn ý nghĩa của TP/FP/FN. Trace chưa chứng minh
  nguyên nhân nhân quả của một lỗi hoặc chứng minh mô hình đã hiểu ngôn ngữ.

## Nhập trace từ Kaggle

Mở **⚙ Chạy câu mới bằng checkpoint / nhập trace / hướng dẫn chia sẻ** phía trên,
chọn **Nhập JSON từ Kaggle**. Hoặc bấm **Nhập trace JSON** ở chân giao diện.
Giới hạn 8 MB. Hỗ trợ schema `spans/gold` và `entities/expected`.
Span đầu vào `[start, end]` **bao gồm end**; span output `[start, end)` **loại
trừ end**. Demo nhận 2–20 thực thể không chồng nhau, tối đa 30 nhãn mỗi câu.

Để xuất trace mới trong môi trường GLiREL đã cài đúng trên Kaggle, đặt hai file
`export_trace.py`, `inference.py` và `core.py` cạnh nhau rồi chạy:

```bash
python export_trace.py /kaggle/input/.../model_2400 --output demo_predictions.json
```

Tải JSON về để nhập app. Không cần mang trọng số về máy trình bày. Các hooks cần
cấu trúc model tương ứng commit ghim. Không tự điền giả các tensor chưa ghi.

## Chạy checkpoint với câu mới (tùy chọn)

Chế độ xem bản ghi chỉ cần `requirements.txt`. Muốn chạy inference thật, cài thêm
trong môi trường riêng có Git, đủ RAM và ổ đĩa; Python 3.12 phù hợp với bộ gói này:

```powershell
python -m pip install -r requirements-model.txt
```

Trong phần ⚙ chọn **Chạy checkpoint**. Điền thư mục chứa `glirel_config.json`
và `pytorch_model.bin`, hoặc giữ ID `jackboyla/glirel-large-v0`. ID chính chủ ghim
revision `02e48d6f3fa5fe88953500abc4a37a5428b77af8`. Chọn CPU/CUDA, nhập câu đã
tách token bằng khoảng trắng, mỗi nhãn một dòng, span theo chỉ số token từ 0.
Ví dụ `Ada works for Acme .` → `[[0,0,"PERSON"],[3,3,"ORG"]]`;
gold tùy chọn `[[0,"employer",1]]`. Bấm **Chạy GLiREL và mở trace mới**.

Chỉ nút này mới nạp model. Hooks được gỡ trong `finally`; khóa inference tránh
trộn trace khi nhiều phiên dùng chung model. **Inference với trọng số chưa được
chạy end-to-end trên máy đóng gói**; chế độ bản ghi và thành phần giao diện đã
được kiểm tra. Không coi bản triển khai Cloud nhẹ là server GPU để chạy model lớn;
muốn phục vụ câu mới cho nhiều người cần máy đủ tài nguyên và cài bộ model.

## Kiến trúc và nguồn đối chiếu

Ghép `[REL] label ... [SEP] sentence` → encoder chung → pooling → hai nhánh:
văn bản qua BiLSTM và biểu diễn span/cặp; nhãn qua FFN → dot product + sigmoid →
threshold/top-k. Với cấu hình `both`, lấy trung bình vector marker và tên nhãn
trước FFN. Refinement tùy checkpoint, không tự bật trong minh họa.
Phép ghép biên span ở hình là ký hiệu rút gọn; `markerv1` còn có các phép chiếu
riêng cho biên đầu/cuối trước khi ghép. Không dùng pipeline RE-Matching để thay
cho GLiREL; mẫu người dùng cung cấp chỉ được tham khảo về cách trình bày.

- [Bài báo GLiREL](https://arxiv.org/abs/2501.03172)
- [Mã nguồn ghim commit](https://github.com/jackboyla/GLiREL/tree/1f485a268c2144cdc0221bcf0b24152f93c0a155)
- [Joint encoding và nhánh nhãn](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/model.py)
- [Biểu diễn span/cặp](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/rel_rep.py)

## Kiểm tra dành cho người phát triển

```powershell
python -m unittest discover -s tests -v
node tests/test_frontend.cjs
```

Tests kiểm tra span/chiều, ngưỡng nghiêm ngặt, top-k từng cặp, chẩn đoán thiếu
nhãn, dữ liệu lỗi, chế độ Streamlit không tự nạp trọng số, và luồng JavaScript.
Trình duyệt còn được kiểm tra thực tế: component tải được, animation đổi module,
bấm ô ma trận, trang soi lỗi, bố cục desktop/mobile. Đây là kiểm thử ứng dụng,
không phải đánh giá chất lượng model.
