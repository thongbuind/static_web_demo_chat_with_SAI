[🌐 Xem web demo](https://thongbuind.github.io/static_web_demo_chat_with_SAI/)

# SAI_375M Chat Demo

Trang web tĩnh để xem các đoạn chat mẫu của SAI_375M, không hỗ trợ chat trực tiếp.

Nếu ai muốn trải nghiệm thì ghé [Hugging Face](https://huggingface.co/collections/thongbuind/sai).

Kết quả được sinh từ checkpoint `tmp/sft1_375M.pt` với tokenizer của
[SAI_100M trên Hugging Face](https://huggingface.co/thongbuind/SAI_100M) và mã nguồn
[SAI_dev2](https://github.com/thongbuind/SAI_dev2), cấu hình 375M.
510 câu hỏi độc lập được dùng cho 680 lượt trả lời trong 43 nhóm chat; các câu hỏi
trùng nhau dùng chung kết quả. Mỗi câu chạy riêng, không thêm system prompt hay
lịch sử hội thoại. Sử dụng beam search 5 nhánh, tối đa 500 token, repetition penalty
1.2 và no-repeat 3-gram. Trọng số và KV cache chạy float16 trên Apple Metal,
điểm beam tính bằng float32 trên CPU. Chi tiết và SHA-256 của checkpoint/tokenizer
nằm trong `generation_375M.json`.

Để sinh lại kết quả, đặt checkpoint vào `tmp/sft1_375M.pt`, rồi chạy từ thư mục dự án:

```sh
python3 -m venv tmp/inference-venv
tmp/inference-venv/bin/python -m pip install -r scripts/requirements-inference.txt
tmp/inference-venv/bin/python scripts/regenerate_history.py --device mps --dtype float16
```

Lần chạy đầu cần Internet để tải tokenizer và mã nguồn ở các phiên bản cố định.
Máy không hỗ trợ Apple Metal có thể dùng `--device cpu --dtype float32`.
Tiến độ được lưu trong `tmp/results_375M_<dtype>.json`, dữ liệu trước khi thay nằm
trong `tmp/history_100M`. Chỉ cập nhật toàn bộ JSON của demo khi đã sinh đủ kết quả.
Thư mục `tmp/` chứa checkpoint và môi trường chạy, được bỏ qua bởi Git.
