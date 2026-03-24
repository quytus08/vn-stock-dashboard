# 📈 VN Stock Dashboard

Dashboard phân tích chứng khoán Việt Nam — portfolio project cho ngành Data Analytics.

## Tech stack
- **vnstock** — dữ liệu giá cổ phiếu HOSE/HNX (miễn phí)
- **Pandas** — xử lý dữ liệu
- **Plotly** — biểu đồ candlestick tương tác
- **Scikit-learn** — mô hình Ridge Regression dự đoán xu hướng
- **Streamlit** — web app

## Cài đặt & chạy

```bash
# 1. Clone hoặc tải project
cd vn-stock-dashboard

# 2. Tạo virtual environment (khuyến nghị)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy app
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

## Tính năng
- Candlestick chart tương tác với Plotly
- MA20 / MA50 overlay
- Volume bar chart
- ML prediction với confidence band (Ridge Regression)
- Download dữ liệu CSV
- Offline demo mode (dữ liệu mô phỏng nếu không có internet)

## Deploy lên Streamlit Cloud (miễn phí)

1. Push code lên GitHub
2. Vào [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Deploy
4. Có link public để gắn vào portfolio/CV

## Cấu trúc project

```
vn-stock-dashboard/
├── app.py                 ← Main app
├── data/
│   ├── __init__.py
│   └── fetch.py           ← Lấy dữ liệu vnstock
├── models/
│   ├── __init__.py
│   └── predictor.py       ← ML model
├── requirements.txt
└── README.md
```

---
> ⚠️ Dự đoán ML chỉ mang tính học thuật / demo, không phải tư vấn tài chính.
