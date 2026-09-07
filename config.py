"""Universe, lookbacks, and composite score weights for the sector-rotation app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"

BENCHMARK = "SPY"
STYLE_ETFS = ("QQQ", "IWM")

# GICS sector -> SPDR Select Sector ETF
SECTOR_ETFS: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

ETF_TO_SECTOR: dict[str, str] = {etf: sector for sector, etf in SECTOR_ETFS.items()}

SECTOR_LABELS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Discretionary",
    "XLP": "Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communications",
}

CYCLICAL_ETFS = ("XLK", "XLF", "XLE", "XLY", "XLI", "XLB", "XLC")
DEFENSIVE_ETFS = ("XLV", "XLP", "XLU", "XLRE")

ALL_ETFS: tuple[str, ...] = (BENCHMARK,) + STYLE_ETFS + tuple(SECTOR_ETFS.values())

# Memory names sit in GICS Semiconductors or storage hardware, so they need an explicit basket.
MEMORY_TICKERS: tuple[str, ...] = ("MU", "WDC", "STX", "SNDK")

# Equal-weight S&P 500 industry baskets. Keep this short so XLK-style blending does not hide the tape.
INDUSTRY_GROUPS: dict[str, dict] = {
    "SEM": {
        "label": "Semiconductors",
        "parent_etf": "XLK",
        "sub_industries": ("Semiconductors",),
        "exclude_tickers": MEMORY_TICKERS,
    },
    "MEM": {
        "label": "Memory & Storage",
        "parent_etf": "XLK",
        "tickers": MEMORY_TICKERS,
    },
    "EQS": {
        "label": "Semi Equipment",
        "parent_etf": "XLK",
        "sub_industries": ("Semiconductor Materials & Equipment",),
    },
    "SOFT": {
        "label": "Software",
        "parent_etf": "XLK",
        "sub_industries": ("Systems Software", "Application Software"),
    },
    "BANK": {
        "label": "Banks",
        "parent_etf": "XLF",
        "sub_industries": ("Diversified Banks", "Regional Banks"),
    },
    "BIOT": {
        "label": "Biotech",
        "parent_etf": "XLV",
        "sub_industries": ("Biotechnology",),
    },
    "EXPL": {
        "label": "Oil E&P",
        "parent_etf": "XLE",
        "sub_industries": ("Oil & Gas Exploration & Production",),
    },
    "RETL": {
        "label": "Retail",
        "parent_etf": "XLY",
        "sub_industries": (
            "Broadline Retail",
            "Apparel Retail",
            "Other Specialty Retail",
            "Homefurnishing Retail",
            "Automotive Retail",
        ),
    },
    "HOME": {
        "label": "Homebuilders",
        "parent_etf": "XLY",
        "sub_industries": ("Homebuilding",),
    },
    "NET": {
        "label": "Internet Platforms",
        "parent_etf": "XLC",
        "sub_industries": ("Interactive Media & Services",),
    },
}

INDUSTRY_LABELS: dict[str, str] = {key: spec["label"] for key, spec in INDUSTRY_GROUPS.items()}

# Trading-day lookbacks
HORIZONS: dict[str, int] = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
}

PRICE_HISTORY = "18mo"
RRG_WINDOW = 21
CMF_WINDOW = 20
MA_WINDOW = 50
BREADTH_HIGH_LOW_WINDOW = 20
RRG_TRAIL_STEPS = 5
RRG_TRAIL_STRIDE = 5  # plot a trail point every N trading days
DOWNLOAD_CHUNK_SIZE = 50
CACHE_MAX_AGE_HOURS = 18

# Percentile ranks of these components are blended into a 0-100 rotation score.
SCORE_WEIGHTS: dict[str, float] = {
    "rs_1m": 0.25,
    "rs_momentum": 0.20,
    "volume_expansion": 0.20,
    "cmf": 0.20,
    "breadth": 0.15,
}

INFLOW_THRESHOLD = 60.0
OUTFLOW_THRESHOLD = 40.0

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_USER_AGENT = "SectorRotationDashboard/1.0 (research; local dashboard)"
