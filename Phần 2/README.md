# VinDatathon — EDA Phần 2

Exploratory data analysis for the Datathon 2026 dataset. The main notebook builds a clean data pipeline on the provided CSVs, then walks through three business insights ("Phần 2 — Trực quan hoá & Phân tích EDA").

## The three insights

1. **Lệch cơ cấu sản phẩm theo vùng làm giảm giá trị đơn hàng** — vùng Tây có AOV thấp hơn ~14% do cơ cấu danh mục (Outdoor 49% vs Streetwear 60% ở East/Central), không phải pricing.
2. **Khuyến mãi đang đổi lợi nhuận lấy doanh thu** — promo tạo 29.9% revenue nhưng lỗ gross ~₫677M; promo `fixed` có biên xấu nhất ~ −63%.
3. **Doanh thu mùa cao điểm bị chặn bởi năng lực tồn kho** — peak tháng 4–6 đi kèm `stockout_days` tăng (corr = 0.589) và `fill_rate` giảm (corr = −0.363). Chỉ báo dẫn `stockout_days_7d` ↔ `revenue_next_7d` corr = 0.51.

Each section follows: **Giả thuyết → Bằng chứng (charts) → Kết luận → Hàm ý kinh doanh → Hành động đề xuất**.

## Repo layout

```
VinDatathon/
├── part2_eda.ipynb       # main notebook: setup → cleaning → joining → 3 insights
├── data/
│   ├── baseline.ipynb    # provided baseline
│   ├── customers.csv
│   ├── geography.csv
│   ├── inventory.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── payments.csv
│   ├── products.csv
│   ├── promotions.csv
│   ├── returns.csv
│   ├── reviews.csv
│   ├── sales.csv
│   ├── shipments.csv
│   ├── web_traffic.csv
│   └── sample_submission.csv
├── requirements.txt
└── README.md
```

The notebook expects the working directory to be the repo root, so the `data/` path resolves correctly.

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
pandas
numpy
matplotlib
seaborn
jupyter
ipykernel
nbclient
nbformat
```

## Run

Open the notebook in your editor of choice and run all cells top-to-bottom:

```bash
jupyter notebook part2_eda.ipynb
```

Or in VS Code: open `part2_eda.ipynb` and click **Run All**.

The notebook is structured so that cells 0–18 build all shared dataframes (`fact_orders`, `daily_metrics`, `cust_geo`, `oi_full`, etc.); the three insight sections (2.1, 2.2, 2.3) then read from those tables.
