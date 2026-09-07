# US Sector Rotation Dashboard

Estimates where money is rotating across US GICS sectors and a few industry groups (semis, memory, software, and others) using prices, volume, and S&P 500 breadth. This is a pressure map, not true institutional fund-flow data.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501). Use the **Industries** tab for subgroups that XLK would otherwise hide.

Data comes from Yahoo Finance and the Wikipedia S&P 500 list, cached locally under `data_cache/` (not committed).
