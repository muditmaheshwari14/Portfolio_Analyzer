# backend/app/catalog.py
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session

# Reuse your existing Base/engine/SessionLocal from main.py
# We'll import these in main.py, not here. This file only defines the model & helpers.

from .main import Base  # main.py must define Base = declarative_base()

class AssetDB(Base):
    __tablename__ = "assets"
    id          = Column(Integer, primary_key=True)
    asset_class = Column(String(32), index=True, nullable=False)  # Equity | ETF | MutualFund | Crypto | Commodity
    region      = Column(String(64), index=True, nullable=True)   # USA, India, UK, Japan, Australia, Hong Kong, Canada, China (NULL for Crypto/Commodity)
    symbol      = Column(String(64), index=True, nullable=False, unique=True)
    name        = Column(String(256), nullable=False)
    exchange    = Column(String(128), nullable=True)
    currency    = Column(String(8),  nullable=True)
    sector      = Column(String(64), nullable=True)         # ← NEW
    market_cap  = Column(String(32), nullable=True) 

SEED: List[Dict[str, Any]] = [
    # ---- USA ----
    {"asset_class":"Equity","region":"USA","symbol":"AAPL","name":"Apple Inc.","exchange":"NASDAQ","currency":"USD","sector":"Information Technology"},
    {"asset_class":"Equity","region":"USA","symbol":"MSFT","name":"Microsoft Corporation","exchange":"NASDAQ","currency":"USD","sector":"Information Technology"},
    {"asset_class":"ETF","region":"USA","symbol":"SPY","name":"SPDR S&P 500 ETF Trust","exchange":"NYSEARCA","currency":"USD","sector":"Multi-Asset"},
    {"asset_class":"MutualFund","region":"USA","symbol":"VTSAX","name":"Vanguard Total Stock Mkt Adm","exchange":"USMF","currency":"USD","sector":"Multi-Asset"},

    # ---- India ----
    {"asset_class":"Equity","region":"India","symbol":"RELIANCE.NS","name":"Reliance Industries Ltd","exchange":"NSE","currency":"INR","sector":"Energy"},
    {"asset_class":"ETF","region":"India","symbol":"NIFTYBEES.NS","name":"Nippon India ETF Nifty BeES","exchange":"NSE","currency":"INR","sector":"Multi-Asset"},
    {"asset_class":"MutualFund","region":"India","symbol":"0P0000XQW5.BO","name":"SBI Small Cap Fund - Reg Gr","exchange":"BSEMF","currency":"INR","sector":"Multi-Asset"},

    # ---- UK ----
    {"asset_class":"Equity","region":"UK","symbol":"HSBA.L","name":"HSBC Holdings plc","exchange":"LSE","currency":"GBP","sector":"Financials"},
    {"asset_class":"ETF","region":"UK","symbol":"VUKE.L","name":"Vanguard FTSE 100 UCITS ETF","exchange":"LSE","currency":"GBP","sector":"Multi-Asset"},

    # ---- Japan ----
    {"asset_class":"Equity","region":"Japan","symbol":"7203.T","name":"Toyota Motor Corporation","exchange":"TSE","currency":"JPY","sector":"Consumer Discretionary"},
    {"asset_class":"ETF","region":"Japan","symbol":"1306.T","name":"TOPIX ETF","exchange":"TSE","currency":"JPY","sector":"Multi-Asset"},

    # ---- Australia ----
    {"asset_class":"Equity","region":"Australia","symbol":"CBA.AX","name":"Commonwealth Bank of Australia","exchange":"ASX","currency":"AUD","sector":"Financials"},

    # ---- Hong Kong ----
    {"asset_class":"Equity","region":"Hong Kong","symbol":"0700.HK","name":"Tencent Holdings Ltd","exchange":"HKEX","currency":"HKD","sector":"Communication Services"},

    # ---- Canada ----
    {"asset_class":"Equity","region":"Canada","symbol":"RY.TO","name":"Royal Bank of Canada","exchange":"TSX","currency":"CAD","sector":"Financials"},

    # ---- China ----
    {"asset_class":"Equity","region":"China","symbol":"600519.SS","name":"Kweichow Moutai Co., Ltd.","exchange":"SSE","currency":"CNY","sector":"Consumer Staples"},

    # ---- Global (no region) ----
    {"asset_class":"Crypto","region":None,"symbol":"BTC-USD","name":"Bitcoin","exchange":"Crypto","currency":"USD","sector":"Crypto"},
    {"asset_class":"Crypto","region":None,"symbol":"ETH-USD","name":"Ethereum","exchange":"Crypto","currency":"USD","sector":"Crypto"},
    {"asset_class":"Commodity","region":None,"symbol":"GC=F","name":"Gold Futures","exchange":"COMEX","currency":"USD","sector":"Commodities"},
    {"asset_class":"Commodity","region":None,"symbol":"CL=F","name":"Crude Oil WTI","exchange":"NYMEX","currency":"USD","sector":"Commodities"},
]

def seed_assets_once(db: Session):
    """Seed the catalog if empty."""
    from sqlalchemy import func
    count = db.query(func.count(AssetDB.id)).scalar()
    if count == 0:
        db.bulk_insert_mappings(AssetDB, SEED)
        db.commit()
