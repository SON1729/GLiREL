# Đối chiếu pipeline minh họa GLiREL

Phạm vi: **một lượt suy luận phân loại quan hệ**, với thực thể và nhãn đã cho.
Không minh họa tối ưu loss, tạo ZeroRel hoặc toàn bộ quy trình huấn luyện.

Nguồn mã đối chiếu: commit
[`1f485a268c2144cdc0221bcf0b24152f93c0a155`](https://github.com/jackboyla/GLiREL/tree/1f485a268c2144cdc0221bcf0b24152f93c0a155).
Nguồn số liệu: `data/recorded.json`, checkpoint khai báo `model_6600`.
[Bài báo](https://arxiv.org/html/2501.03172v1) cung cấp mô tả phương pháp; các
chi tiết triển khai trên web bám theo mã và cấu hình của bản ghi, không tự coi mọi
biến thể trong mã là cùng một cấu hình của paper. Mô tả FFN span trong paper là
dạng rút gọn; hình trên web làm rõ các phép chiếu biên và BiLSTM có trong code.
Refinement cũng theo lời gọi và cờ read_only của commit, không tự thêm self-attention
chỉ vì sơ đồ tổng quát của paper có đề cập.

## Từng bước đang được minh họa

| Bước | Phép biến đổi theo code | Nguồn kiểm tra |
|---|---|---|
| 1. Đầu vào | Token câu + span `[start,end]` inclusive + tên nhãn. Entity type là metadata trong demo; không chạy NER và không tự ràng buộc loại head/tail. | [`base.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/base.py), `preprocess_spans`, `collate_fn` |
| 2. Prompt/tokenizer | Với M nhãn, thêm `[REL], label` cho từng nhãn, rồi `[SEP]` và câu. P=2M+1, U=P+N. Tên nhiều từ vẫn là một đơn vị trước tokenizer. Tokenizer dùng `is_split_into_words=True`; ID subtoken khác ID thực thể/vị trí token câu. | [`model.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/model.py), `compute_score`; [`transformer_embeddings.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/transformer_embeddings.py), `embed` |
| 3. Encoder/pooling/projection | DeBERTa tạo `[B,L,1024]`. Với `first`, giữ subtoken đầu của mỗi đơn vị theo word_ids, rồi Linear 1024→768. Hook `pooled_embeddings` thực tế ghi **sau phép chiếu**, không phải tensor 1024 chiều ngay sau pooling. | [`token_rep.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/token_rep.py), `TokenRepLayer.forward` |
| 4. Hai nhánh | Lấy câu `E[:,P:P+N,:]` qua BiLSTM 1 lớp hai chiều, hidden mỗi chiều 384, output 768. Nhãn lấy từ phần prompt; `both` trung bình vị trí 2t và 2t+1 **trước FFN nhãn**. Không chạy encoder lần hai. | `model.py`, `compute_score`; [`layers.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/layers.py), `LstmSeq2SeqEncoder` |
| 5. Thực thể/cặp | `markerv1`: FFN_start/FFN_end riêng → gather hai biên → nối → ReLU → FFN_span. Gather head/tail theo relations_idx → nối → FFN_pair. `markerv2` dùng mean trong span rồi chiếu. Không suy ra span mode chỉ từ shape. | [`rel_rep.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/rel_rep.py); [`span_rep.py`](https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/glirel/modules/span_rep.py), `create_projection_layer` |
| 6. FFN nhãn | Với `both`, `p=(marker+label)/2`. `prompt_rep_layer`: Linear 768→3072 → Dropout → ReLU → Linear 3072→768. Dropout tắt ở eval. FFN nhãn khác trọng số các FFN span/cặp. `label` và `ent_token` chọn nguồn vector khác; hình thay theo config. | `model.py`, `__init__`, `compute_score` |
| 7. Refine/scorer/sigmoid | Refine_relation (nếu có) đọc token sau RNN; refine_prompt đọc cặp sau bước refine_relation. Commit này khởi tạo read_only=True nên nhánh self-attention trong ReadProcess không chạy. Với scorer=dot, einsum `BKD,BCD→BKC`, sau đó sigmoid. Không có cosine hoặc softmax trong nhánh dot. | `model.py`, `compute_score`, `predict`; `layers.py`, `ReadProcess`, `ScorerLayer` |
| 8. Giải mã | `score > threshold` nghiêm ngặt; top-k riêng cho từng cặp có hướng. API trả end-exclusive `[start,end+1)`. Thay threshold/top-k chỉ lọc điểm cũ; đổi câu/span/nhãn phải suy luận lại. | `model.py`, `predict`, `batch_predict_relations` |

Nhiều bước trên là các phần của cùng một forward pass. Animation chia chúng ra
để học, không mô phỏng thời gian tính thực tế hoặc giả định nhãn/cặp được model
chạy tuần tự từng cái.

## Những điểm đã sửa để tránh hiểu sai

- Bước hai nhánh chỉ cho thấy tách tensor và BiLSTM/nguồn vector nhãn. FFN nhãn
  được trình bày ở bước 6, tránh tạo cảm giác nó chạy hai lần.
- Bước tạo span tách riêng FFN_start, FFN_end, ReLU sau concat, FFN_span và FFN_pair.
  Không dùng ký hiệu FFN duy nhất để ám chỉ các lớp này chia sẻ tham số.
- Dùng shape thật để phân biệt chiều encoder 1024 và chiều GLiREL 768.
- Refinement hiện rõ bật/tắt theo config, không thêm attention giả vào trace có
  cả hai cờ false. Các scorer khác dot cũng có công thức khác, không mặc định dot.
- Không lấy các tọa độ đầu của toàn tensor để gán cho cặp/nhãn đang chọn. Chúng là
  summary toàn tensor. Bản ghi mới từ chế độ local bổ sung axes và dữ liệu scorer
  để các lần kiểm tra sau có tọa độ cụ thể hơn.
- Logit lưu thật được dùng khi xác nhận được axes và đoạn prefix. Ngoài đoạn đó,
  chỉ hiển thị `log(s/(1-s))` với nhãn **suy ngược**, không giả đó là logit đã lưu.
- Điểm trong ma trận được làm tròn 2 chữ số. Ô `0.00` không có nghĩa score chính
  xác bằng 0; phần chi tiết hiển thị nhiều chữ số hơn.
- E(E−1) là số cặp tối đa khi không tự ghép; cặp thực tế phải theo relations_idx.
  Khi có giới hạn khoảng cách, không tự tái dựng bộ lọc đối xứng: hàm source còn
  OR với coref_mask giữ i<j. Trace mới lưu pair_axis để đối chiếu trực tiếp.

## Kiểm tra số liệu ghi sẵn

Chạy `python audit_pipeline.py` để tái tạo `data/pipeline_audit.json`.
Script kiểm tra thứ tự prompt, U=P+N, shape TokenRep/BiLSTM/nhãn/cặp/scorer,
bảng ứng viên đủ ô và đối chiếu sigmoid của đoạn logits đã lưu.

| Ví dụ | N token | E thực thể | M nhãn | U đơn vị | K cặp | Logit đã kiểm tra | Sai số sigmoid lớn nhất |
|---|---:|---:|---:|---:|---:|---:|---:|
| Câu đơn | 5 | 2 | 3 | 12 | 2 | 6 | 9,96×10⁻⁹ |
| Nhiều thực thể | 11 | 3 | 3 | 18 | 6 | 6 | 8,38×10⁻⁹ |
| Phủ định | 6 | 2 | 3 | 13 | 2 | 6 | 6 | 3,78×10⁻⁸ |
| Thiếu nhãn | 5 | 2 | 2 | 10 | 2 | 4 | 3,12×10⁻⁹ |
| Nhiều câu/đại từ | 12 | 3 | 3 | 19 | 6 | 6 | 2,61×10⁻⁸ |

Tổng **28 logit** đã lưu được kiểm tra. Đây không phải kiểm chứng tất cả logits:
trace cũ chỉ giữ prefix 6 tọa độ (ví dụ thiếu nhãn có 4). Các bảng output đã được
sắp theo điểm, nên không lấy thứ tự bảng output làm thứ tự hàng tensor. Script
đối chiếu theo thứ tự cặp của code và nhãn trong prompt; UI chỉ bind prefix khi
toàn đoạn đã lưu khớp sigmoid trong sai số 10⁻⁶.

Ví dụ: Ada → employer → Acme có `Z[0,0,0]=0.6871376633644104`;
`sigmoid(Z)≈0.665329884`, khớp điểm lưu trong sai số float32.

## Cấu hình chưa xác minh được

Trace `model_6600` thiếu `span_marker_mode`, `max_entity_pair_distance`,
`add_entity_markers`, `coref_classifier`. Không chỉnh sửa trace cũ để tự điền các
giá trị suy đoán. Riêng hình span đang có nhãn rõ: **minh họa markerv1 theo config
FewRel công khai, chưa xác nhận span mode của checkpoint này**.

Để chốt chính xác biến thể cần file **glirel_config.json của model_6600**. Config
của seed/checkpoint khác chưa đủ để chứng minh cấu hình của model này. Các flag
refinement, label strategy, scorer và hidden_size đã có trong trace được dùng trực tiếp.

Kiểm tra này không tái nạp trọng số để chạy inference, không tái lập các con số
benchmark của paper, và không xác nhận checkpoint này chính là checkpoint paper.
Ví dụ đại từ chỉ chấm quan hệ giữa các span được cung cấp, không chứng minh model
đã thực hiện đầy đủ giải quyết đồng tham chiếu. Gold vẫn là đáp án ví dụ/người dùng.
