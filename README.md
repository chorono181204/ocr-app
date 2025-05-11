# OCR App

Ứng dụng OCR (Optical Character Recognition) cho phép nhận dạng và trích xuất văn bản từ hình ảnh.

## Cấu trúc dự án

- `client/`: Frontend React application
- `server/`: Backend Python application

## Cài đặt

### Backend (Python)

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Frontend (React)

```bash
cd client
npm install
```

## Chạy ứng dụng

### Backend

```bash
cd server
uvicorn app:app --reload

```

### Frontend

```bash
cd client
npm start
```

## Tính năng

- Upload hình ảnh
- Nhận dạng văn bản từ hình ảnh
- Hiển thị kết quả OCR
- Hỗ trợ nhiều định dạng hình ảnh 