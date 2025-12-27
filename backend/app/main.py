# backend/app/main.py
import logging, traceback, warnings, json, os, time, math, statistics
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any,Tuple
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]  # SAAS_PROJECT/
load_dotenv(ROOT / ".env") 
import numpy as np
import pandas as pd
import yfinance as yf
import httpx
from fastapi import Query
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from fastapi.responses import PlainTextResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy import or_, func 
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.exc import IntegrityError

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from .catalog import AssetDB, seed_assets_once
# =========================
# Config
# =========================



warnings.filterwarnings("default")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio-api")

SECRET_KEY = os.environ.get("APP_SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# =========================
# DB Models
# =========================

try:
    from openai import OpenAI, RateLimitError
    _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception:
    _openai_client = None   # no key or package => raise nicely below

class ReportIn(BaseModel):
    request: Dict[str, Any]
    analytics: Dict[str, Any]
    risk: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    portfolios = relationship("PortfolioDB", back_populates="user", cascade="all, delete-orphan")

class PortfolioDB(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    name = Column(String(255), nullable=False)
    holdings_json = Column(Text, nullable=False)  # list[Holding] as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("UserDB", back_populates="portfolios")

Base.metadata.create_all(bind=engine)
with SessionLocal() as _db:
    seed_assets_once(_db)

# =========================
# FastAPI app & CORS
# =========================
app = FastAPI(title="Portfolio Analytics API")

@app.exception_handler(Exception)
async def _debug_any_exc(request: Request, exc: Exception):
    if os.environ.get("REPORT_DEBUG","0") == "1":
        return PlainTextResponse(traceback.format_exc(), status_code=500)
    raise exc




app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# =========================
# Schemas
# =========================




class Holding(BaseModel):
    asset_class: str                      # "Equity" | "ETF" | "Commodity" | "Crypto" | "Bond" | "Cash"
    country: Optional[str] = None
    name: Optional[str] = None            # friendly name from UI
    ticker: str                           # canonical Yahoo symbol (e.g., AAPL, RELIANCE.NS, GC=F, BTC-USD)
    currency: str                         # local trading currency code (USD, INR, GBP, EUR, etc.)
    quantity: float
    buy_price: float 



class Portfolio(BaseModel):
    name: str
    base_currency: str = "USD"            # NEW
    holdings: List[Holding]

class Shock(BaseModel):
    ticker: str            # "AAPL"
    pct: float             # -0.50 = -50%, +0.20 = +20%
    mode: str              # "day1" | "random" | "first_k" | "last_k"
    k_days: Optional[int] = None

class StressScenario(BaseModel):
    name: str = "User Scenario"
    shocks: List[Shock] = []                 # <-- keep this
    day1_shock_pct: Dict[str, float] = {}    # internal map for simulator
    mu_scale: float = 1.0
    vol_scale: float = 1.0
    corr_toward: Optional[float] = None
    corr_alpha: float = 0.0
    horizon_days: int = 126
    n_sims: int = 200

class StressRequest(BaseModel):
    name: str
    base_currency: Optional[str] = "USD"     # <-- add this if you want
    holdings: List[Holding]
    scenario: StressScenario

# Auth schemas
class RegisterBody(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MeOut(BaseModel):
    id: int
    email: EmailStr

class PortfolioOut(BaseModel):
    id: int
    name: str
    created_at: datetime

class PortfolioDetailOut(BaseModel):
    id: int
    name: str
    holdings: List[Holding]
    created_at: datetime

# =========================
# Auth helpers
# =========================


def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> UserDB:
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
        user_id = int(sub)
    except JWTError:
        raise credentials_exception
    user = db.get(UserDB, user_id)
    if not user:
        raise credentials_exception
    return user

# =========================
# Data fetching & analytics
# =========================
def fetch_price(ticker: str, start: str, end: str = None):
    if end is None:
        end = datetime.utcnow().date().isoformat()
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data is None or data.empty:
        raise ValueError(f"No price data for {ticker}")
    series = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
    series = series.ffill().dropna()
    series.name = ticker
    return series

def compute_returns(price_series: pd.Series):
    return price_series.pct_change().dropna()

def annualized_sharpe(returns, risk_free_rate=0.0):
    mean_daily = np.nanmean(returns)
    vol_daily = np.nanstd(returns, ddof=0)
    if vol_daily == 0:
        return np.nan
    mean_annual = mean_daily * 252
    vol_annual = vol_daily * np.sqrt(252)
    return (mean_annual - risk_free_rate) / vol_annual

def sortino_ratio(returns, required_return=0.0):
    downside = returns[returns < required_return]
    if downside.size == 0:
        return np.nan
    downside_dev_daily = np.sqrt(np.nanmean(downside ** 2))
    mean_annual = np.nanmean(returns) * 252
    downside_annual = downside_dev_daily * np.sqrt(252)
    return (mean_annual - required_return) / downside_annual

def max_drawdown(price_series: pd.Series):
    roll_max = price_series.cummax()
    drawdown = (price_series - roll_max) / roll_max
    return float(drawdown.min())

def monte_carlo_price_sim(series: pd.Series, n_sims=200, n_days=252):
    returns = series.pct_change().dropna()
    mu, sigma = np.nanmean(returns), np.nanstd(returns)
    last = float(series.iloc[-1])
    sims = []
    for _ in range(n_sims):
        prices = [last]
        for _ in range(n_days):
            prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
        sims.append(prices)
    return sims

# ---- Forward Monte Carlo (bootstrap from historical daily returns) ----
def mc_forward_summary(
    port_daily_rets: pd.Series,
    start_amount: float,
    years: int = 15,
    n_sims: int = 3000,
    fees_bps: float = 0.0,
    inflation: float = 0.0,
    block_len: int = 0,
):
    if port_daily_rets is None:
        return None
    if isinstance(port_daily_rets, pd.Series):
        ret = port_daily_rets.dropna().astype(float).values
    else:
        ret = pd.Series(port_daily_rets, dtype="float64").dropna().values
    if ret.size == 0 or not np.isfinite(start_amount) or start_amount <= 0:
        return None

    days = int(252 * years)
    rng = np.random.default_rng()

    # per-day drags
    fee_d = (fees_bps / 10000.0) / 252.0
    infl_d = inflation / 252.0

    # --- draw daily sequences ---
    if block_len and block_len > 1:
        n_blocks = int(np.ceil(days / block_len))
        max_start = max(0, len(ret) - block_len)
        values = np.empty((n_sims, days), dtype=float)
        for s in range(n_sims):
            seq = []
            for _ in range(n_blocks):
                i0 = rng.integers(0, max_start + 1)
                chunk = ret[i0:i0+block_len]
                seq.append(chunk)
            seq = np.concatenate(seq, axis=0)[:days]
            steps = (1.0 + seq) * (1.0 - fee_d)
            growth = np.cumprod(steps, axis=0)
            values[s, :] = start_amount * growth
    else:
        draws = rng.choice(ret, size=(n_sims, days), replace=True)
        steps = (1.0 + draws) * (1.0 - fee_d)
        growth = np.cumprod(steps, axis=1)
        values = start_amount * growth

    # "real" values (deflate by inflation)
    infl_factor = (1.0 + infl_d) ** np.arange(1, days + 1)
    values_real = values / infl_factor[None, :]

    # --- year grid (include 0) ---
    yr_points = [0] + [min(days, int(round(252 * y))) for y in range(1, years + 1)]

    def pctile_path(valmat, q):
        out = []
        for i in yr_points:
            v = start_amount if i == 0 else np.percentile(valmat[:, i - 1], q)
            out.append(float(v))
        return out

    percentile_paths = {
        "years": list(range(0, years + 1)),
        "p10": pctile_path(values, 10),
        "p25": pctile_path(values, 25),
        "p50": pctile_path(values, 50),
        "p75": pctile_path(values, 75),
        "p90": pctile_path(values, 90),
    }
    percentile_paths_real = {
        "years": list(range(0, years + 1)),
        "p10": pctile_path(values_real, 10),
        "p25": pctile_path(values_real, 25),
        "p50": pctile_path(values_real, 50),
        "p75": pctile_path(values_real, 75),
        "p90": pctile_path(values_real, 90),
    }

    # ---------- helper metrics from a single daily price series ----------
    def percentile_daily(valmat, q):
        v = np.percentile(valmat, q, axis=0)           # (days,)
        return np.concatenate([[start_amount], v])     # include t=0

    def metrics_from_values(vs: np.ndarray):
        rets = vs[1:] / vs[:-1] - 1.0
        yrs = len(rets) / 252.0
        cagr = (vs[-1] / vs[0]) ** (1.0 / yrs) - 1.0 if vs[0] > 0 and yrs > 0 else None
        if rets.size:
            mu_d = np.nanmean(rets)
            sd_d = np.nanstd(rets, ddof=0)
            vol_ann = sd_d * np.sqrt(252)
            downside = rets[rets < 0]
            sortino = (mu_d * 252) / (np.sqrt(np.nanmean(downside**2)) * np.sqrt(252)) if downside.size else None
            sharpe = (mu_d * 252) / vol_ann if vol_ann and vol_ann > 0 else None
        else:
            vol_ann = sharpe = sortino = None
        peak = np.maximum.accumulate(vs)
        mdd = float(np.min(vs / peak - 1.0))
        return cagr, vol_ann, sharpe, sortino, mdd

    pcts = [10, 25, 50, 75, 90]

    # ---------- NOMINAL tables ----------
    def _report_horizons(total_years: int) -> list[int]:
        # include a nice set but only those ≤ total_years; always include total_years
        base = [1, 3, 5, 10, 15, total_years]
        hs = sorted({h for h in base if h <= total_years})
        if total_years not in hs:
            hs.append(total_years)
        return hs

    horizons = _report_horizons(years)

    # ---------- NOMINAL tables ----------
    perf = {"percentiles": [10, 25, 50, 75, 90], "twrr": [], "vol_annual": [], "sharpe": [], "sortino": [], "max_drawdown": [], "end_balance": []}
    for q in perf["percentiles"]:
        v_daily = percentile_daily(values, q)
        twrr, vol_a, sh, so, mdd = metrics_from_values(v_daily)
        perf["twrr"].append(twrr); perf["vol_annual"].append(vol_a); perf["sharpe"].append(sh)
        perf["sortino"].append(so); perf["max_drawdown"].append(mdd); perf["end_balance"].append(float(v_daily[-1]))

    # expected annual return percentiles (nominal)
    exp_ann = {"horizons": horizons, "p10": [], "p25": [], "p50": [], "p75": [], "p90": []}
    for H in horizons:
        idx = min(days - 1, int(round(252 * H)) - 1)   # clamp to the end of the series
        end_vals = values[:, idx]
        ann = (end_vals / start_amount) ** (1.0 / H) - 1.0
        for qq, key in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]:
            exp_ann[key].append(float(np.percentile(ann, qq)))

    # annual return probabilities (nominal)
    thresholds = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125]
    # split daily steps into `years` approximately-equal chunks; each chunk ≈ 1 year
    blocks = np.array_split((values[:, 1:] / values[:, :-1]), years, axis=1)
    ann_rets = [np.prod(b, axis=1) - 1.0 for b in blocks]  # yearly returns per sim
    probs_matrix = []
    for thr in thresholds:
        row = []
        for H in horizons:
            avg_ann = np.vstack(ann_rets[:H]).mean(axis=0)  # average of the first H yearly returns
            row.append(float(np.mean(avg_ann >= thr)))
        probs_matrix.append(row)

    # ---------- REAL tables (same logic on deflated paths) ----------
    perf_real = {"percentiles": [10, 25, 50, 75, 90], "twrr": [], "vol_annual": [], "sharpe": [], "sortino": [], "max_drawdown": [], "end_balance": []}
    for q in perf_real["percentiles"]:
        v_daily_r = percentile_daily(values_real, q)
        twrr, vol_a, sh, so, mdd = metrics_from_values(v_daily_r)
        perf_real["twrr"].append(twrr); perf_real["vol_annual"].append(vol_a); perf_real["sharpe"].append(sh)
        perf_real["sortino"].append(so); perf_real["max_drawdown"].append(mdd); perf_real["end_balance"].append(float(v_daily_r[-1]))

    exp_ann_real = {"horizons": horizons, "p10": [], "p25": [], "p50": [], "p75": [], "p90": []}
    for H in horizons:
        idx = min(days - 1, int(round(252 * H)) - 1)
        end_vals_r = values_real[:, idx]
        ann_r = (end_vals_r / start_amount) ** (1.0 / H) - 1.0
        for qq, key in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]:
            exp_ann_real[key].append(float(np.percentile(ann_r, qq)))

    blocks_r = np.array_split((values_real[:, 1:] / values_real[:, :-1]), years, axis=1)
    ann_rets_r = [np.prod(b, axis=1) - 1.0 for b in blocks_r]
    probs_matrix_r = []
    for thr in thresholds:
        row = []
        for H in horizons:
            avg_ann_r = np.vstack(ann_rets_r[:H]).mean(axis=0)
            row.append(float(np.mean(avg_ann_r >= thr)))
        probs_matrix_r.append(row)

    return {
        "horizon_years": years,
        "start_amount": float(start_amount),
        "assumptions": {
            "fees_bps": float(fees_bps),
            "inflation": float(inflation),
            "block_len": int(block_len) if block_len else 0,
            "n_sims": int(n_sims),
        },
        "percentile_paths": percentile_paths,
        "percentile_paths_real": percentile_paths_real,
        "performance_summary": perf,
        "expected_annual_return": exp_ann,
        "annual_return_probabilities": {
            "thresholds": [t * 100 for t in thresholds],
            "horizons": horizons,
            "matrix": probs_matrix,
        },
        # NEW: real (inflation-adjusted) tables
        "performance_summary_real": perf_real,
        "expected_annual_return_real": exp_ann_real,
        "annual_return_probabilities_real": {
            "thresholds": [t * 100 for t in thresholds],
            "horizons": horizons,
            "matrix": probs_matrix_r,
        },
    }





def portfolio_metrics(returns_df: pd.DataFrame, weights: np.ndarray):
    port_ret = returns_df.dot(weights)
    price_index = (1 + port_ret).cumprod()
    metrics = {
        "sharpe": annualized_sharpe(port_ret),
        "sortino": sortino_ratio(port_ret),
        "max_drawdown": max_drawdown(price_index)
    }
    return metrics, price_index

def _corr_from_cov(cov):
    d = np.sqrt(np.diag(cov))
    outer = np.outer(d, d)
    return cov / outer

def _cov_from_corr(corr, std):
    return corr * np.outer(std, std)

def simulate_stress(df_prices: pd.DataFrame,
    weights: np.ndarray,
    scenario,
    random_shocks: List[tuple] = None,   # (ticker, pct)
    firstk: List[tuple] = None,          # (ticker, pct, k_days)
    lastk: List[tuple] = None            # (ticker, pct, k_days)
):
    rets = df_prices.pct_change().dropna()
    tickers = list(df_prices.columns)
    mu_daily = rets.mean().values
    cov_daily = rets.cov().values

    # scale mu/vol
    mu_daily = mu_daily * scenario.mu_scale
    cov_daily = cov_daily * (scenario.vol_scale ** 2)

    if (scenario.corr_toward is not None
        and 0.0 <= scenario.corr_toward <= 1.0
        and scenario.corr_alpha > 0):
        std = np.sqrt(np.diag(cov_daily))
        # guard against zero std to avoid divide-by-zero
        eps = 1e-12
        denom = np.outer(np.where(std==0, eps, std), np.where(std==0, eps, std))
        corr = cov_daily / denom

        k = float(scenario.corr_alpha)             # 0..1 blend weight
        target = float(scenario.corr_toward)       # target correlation level (e.g., 1.0)
        corr_stressed = (1.0 - k) * corr + k * target
        np.fill_diagonal(corr_stressed, 1.0)

        cov_daily = corr_stressed * np.outer(std, std)

    # correlation stress (unchanged) ...
    
    last_prices = df_prices.iloc[-1].values
    port_index0 = 1.0

    # ===== Day-1 (gap) shocks =====
    shocked_prices = last_prices.copy()
    if scenario.day1_shock_pct:
        for i, t in enumerate(tickers):
            shocked_prices[i] *= (1.0 + scenario.day1_shock_pct.get(t, 0.0))

    w = weights
    day1_ret = np.sum(w * (shocked_prices / last_prices - 1.0))
    start_index = port_index0 * (1.0 + day1_ret)

    # Build lookup indices
    t2i = {t: i for i, t in enumerate(tickers)}
    random_shocks = random_shocks or []
    firstk = firstk or []
    lastk = lastk or []

    sims = []
    rng = np.random.default_rng()
    L = np.linalg.cholesky(cov_daily + 1e-12 * np.eye(cov_daily.shape[0]))
    for _ in range(scenario.n_sims):
        idx = [start_index]

        # Choose random shock day for each "random" shock
        rand_days = []
        for t, pct in random_shocks:
            d = int(rng.integers(1, max(2, scenario.horizon_days // 2)))
            rand_days.append((t2i.get(t, -1), pct, d))  # apply once on that day

        # precompute which days have first_k/last_k active for each ticker
        firstk_map = [(t2i.get(t, -1), pct, k) for (t, pct, k) in firstk]
        lastk_map = [(t2i.get(t, -1), pct, k) for (t, pct, k) in lastk]

        for day in range(1, scenario.horizon_days + 1):
            z = rng.standard_normal(len(tickers))
            daily_ret_vec = mu_daily + (L @ z)

            # ===== Apply per-day shocks to the daily return vector =====
            # random (single day)
            for j, pct, d in rand_days:
                if j >= 0 and day == d:
                    daily_ret_vec[j] = (1.0 + daily_ret_vec[j]) * (1.0 + pct) - 1.0

            # first k days
            for j, pct, k in firstk_map:
                if j >= 0 and day <= k:
                    daily_ret_vec[j] = (1.0 + daily_ret_vec[j]) * (1.0 + pct) - 1.0

            # last k days
            for j, pct, k in lastk_map:
                if j >= 0 and day > max(0, scenario.horizon_days - k):
                    daily_ret_vec[j] = (1.0 + daily_ret_vec[j]) * (1.0 + pct) - 1.0
            # =====================================
            daily_ret_vec = np.clip(daily_ret_vec, -0.99, None)
            port_ret = float(np.dot(w, daily_ret_vec))
            idx.append(idx[-1] * (1.0 + port_ret))
        sims.append(idx)

    def series_mdd_from_prices(prices):
        prices = np.asarray(prices, float)
        roll_max = np.maximum.accumulate(prices)
        dd = (prices - roll_max) / roll_max
        return float(dd.min())

    final_returns = [(s[-1] / s[0] - 1.0) for s in sims]
    mdds = [series_mdd_from_prices(s) for s in sims]

    def pctile(a, p): return float(np.percentile(a, p)) if len(a) else None
    summary = {
        "final_return": {"median": pctile(final_returns, 50), "p05": pctile(final_returns, 5), "p95": pctile(final_returns, 95)},
        "max_drawdown": {"median": pctile(mdds, 50), "p05": pctile(mdds, 5), "p95": pctile(mdds, 95)},
        "prob_breach": {
            "lt_-10": float(np.mean(np.array(final_returns) < -0.10)),
            "lt_-20": float(np.mean(np.array(final_returns) < -0.20))
        },
        "horizon_days": scenario.horizon_days,
        "n_sims": scenario.n_sims,
    }
    return sims, summary

def optimize_portfolio(returns_df: pd.DataFrame, n_trials=10000):
    n_assets = returns_df.shape[1]
    all_returns, all_vols, all_sharpes = [], [], []
    best_sharpe, best_w = -np.inf, None
    for _ in range(n_trials):
        w = np.random.random(n_assets); w /= np.sum(w)
        port_ret = np.dot(returns_df.mean() * 252, w)
        port_vol = np.sqrt(np.dot(w.T, np.dot(returns_df.cov() * 252, w)))
        sharpe = (port_ret) / port_vol if port_vol > 0 else np.nan
        all_returns.append(port_ret); all_vols.append(port_vol); all_sharpes.append(sharpe)
        if not np.isnan(sharpe) and sharpe > best_sharpe:
            best_sharpe, best_w = sharpe, w
    sorted_pairs = sorted(zip(all_vols, all_returns), key=lambda x: x[0])
    sorted_vols, sorted_returns = zip(*sorted_pairs)
    frontier_curve = {"volatilities": list(sorted_vols), "returns": list(sorted_returns)}
    return best_w, best_sharpe, {
        "returns": all_returns, "volatilities": all_vols, "sharpes": all_sharpes, "curve": frontier_curve
    }

# --- FX helpers ---
def yahoo_fx_symbol(from_ccy: str, to_ccy: str) -> str:
    """
    Yahoo quotes pairs like USDINR=X (INR per 1 USD).
    For conversion we want price_in_to = price_in_from / fx if pair is USDINR=X and from=INR,to=USD.
    We'll always request USDxxx=X or xxxUSD=X depending what's available.
    Strategy: try direct <FROM><TO>=X else try <TO><FROM>=X and invert.
    """
    return f"{from_ccy}{to_ccy}=X"

def fetch_fx_series(from_ccy: str, to_ccy: str, start: str, end: str):
    """
    Returns a pd.Series of the conversion rate to turn 'from_ccy' amounts into 'to_ccy'.
    """
    if from_ccy == to_ccy:
        # flat 1 series
        idx = pd.date_range(start=start, end=end, freq="B")
        s = pd.Series(1.0, index=idx, name=f"{from_ccy}->{to_ccy}")
        return s

    direct = yahoo_fx_symbol(from_ccy, to_ccy)   # e.g., INRUSD=X (rare), USDINR=X (common)
    inverse = yahoo_fx_symbol(to_ccy, from_ccy)

    # Try direct first
    df = yf.download(direct, start=start, end=end, progress=False, auto_adjust=False)
    if df is not None and not df.empty:
        s = (df["Adj Close"] if "Adj Close" in df else df["Close"]).ffill().dropna()
        s.name = f"{from_ccy}->{to_ccy}"
        return s

    # Try inverse and invert
    df2 = yf.download(inverse, start=start, end=end, progress=False, auto_adjust=False)
    if df2 is None or df2.empty:
        raise ValueError(f"No FX pair found for {from_ccy}/{to_ccy}")
    s2 = (df2["Adj Close"] if "Adj Close" in df2 else df2["Close"]).ffill().dropna()
    s = 1.0 / s2
    s.name = f"{from_ccy}->{to_ccy}"
    return s

# =========================
# AUTH ROUTES
# =========================


REGION_REQUIRED_FOR = {"Equity", "ETF", "MutualFund"}
REGION_SYNONYMS = {
    "united states": "USA",
    "united kingdom": "UK",
    "u.s.": "USA",
    "u.k.": "UK",
    "us": "USA",
    "america": "USA",
    "global": "Any",
    "any": "Any",
    "europe": "Any",
}
CLASS_SYNONYMS = {
    "fund": "MutualFund",
    "mutual fund": "MutualFund",
    "mutualfund": "MutualFund",
}

def _norm_region(r: Optional[str]) -> Optional[str]:
    if not r: return r
    return REGION_SYNONYMS.get(r.strip().lower(), r)

def _norm_class(c: Optional[str]) -> Optional[str]:
    if not c: return c
    return CLASS_SYNONYMS.get(c.strip().lower(), c)

def fetch_ticker_currency(sym: str) -> Optional[str]:
    try:
        info = yf.Ticker(sym).get_info() or {}
        return info.get("currency")
    except Exception:
        return None
    

# --- SAFE helpers for JSON ---
def _sf(x):
    """safe float: NaN/Inf -> None"""
    try:
        xx = float(x)
        if not np.isfinite(xx):
            return None
        return xx
    except Exception:
        return None

def _safe_list(iterable):
    return [ _sf(v) for v in iterable ]


# -------- helpers to keep token count small --------
def _round(x, nd=4):
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), nd)
    except Exception:
        pass
    return x

def _ds(seq: List[Any], max_points: int = 60) -> List[Any]:
    if not isinstance(seq, list) or len(seq) <= max_points:
        return seq
    n = len(seq)
    if n <= 2:
        return seq
    step = max(1, n // (max_points - 1))
    out = [seq[0]]
    i = step
    while i < n - 1 and len(out) < max_points - 1:
        out.append(seq[i])
        i += step
    if seq[-1] != out[-1]:
        out.append(seq[-1])
    return out

def _topn_weights(weights_map: Dict[str, float], n=12) -> Dict[str, Any]:
    if not isinstance(weights_map, dict):
        return {}
    items = sorted([(k, float(v)) for k, v in weights_map.items()], key=lambda kv: abs(kv[1]), reverse=True)
    top = items[:n]
    other = items[n:]
    other_sum = sum(v for _, v in other)
    res = {k: _round(v, 6) for k, v in top}
    if other:
        res["__other__"] = _round(other_sum, 6)
    res["__count__"] = len(items)
    return res

def _arr_to_map(arr: List[float], tickers: List[str]) -> Dict[str, float]:
    try:
        m = {}
        for i in range(min(len(arr), len(tickers))):
            m[tickers[i]] = float(arr[i])
        return m
    except Exception:
        return {}

def _aggregate_exposure(per_asset: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    expo: Dict[str, float] = {}
    for r in per_asset or []:
        k = r.get(key) or "Unknown"
        w = r.get("weight")
        try:
            w = float(w) if w is not None else 0.0
        except Exception:
            w = 0.0
        expo[k] = expo.get(k, 0.0) + w
    # round and sort
    expo = {k: _round(v, 6) for k, v in sorted(expo.items(), key=lambda kv: abs(kv[1]), reverse=True)}
    # add total count of unique categories for context
    expo["__count__"] = len([k for k in expo.keys() if not k.startswith("__")])
    return expo

def _annual_returns_from_index(index: List[float], dates: List[str]) -> Dict[str, float]:
    """
    Compute calendar-year returns from a normalized index (1.0=start).
    Returns dict: { "YYYY": pct_return }
    """
    if not isinstance(index, list) or not isinstance(dates, list) or len(index) != len(dates) or len(index) < 2:
        return {}
    # build year -> (first_index, last_index)
    first: Dict[int, float] = {}
    last: Dict[int, float] = {}
    for v, ds in zip(index, dates):
        try:
            y = int(ds[:4])
        except Exception:
            continue
        if y not in first:
            first[y] = float(v)
        last[y] = float(v)
    out = {}
    for y in sorted(first.keys()):
        fi, la = first[y], last.get(y, first[y])
        if fi and fi != 0:
            out[str(y)] = _round((la / fi) - 1.0, 6)
    return out

def _estimate_tokens(s: str) -> int:
    # very rough: 1 token ≈ 4 chars
    return max(1, len(s) // 4)

def _numbers_only(d: Dict[str, Any], keys: List[str], nd=6) -> Dict[str, Any]:
    out = {}
    for k in keys:
        if k in d:
            out[k] = _round(d[k], nd)
    return out

def _compact_analytics(a: Dict[str, Any], max_points=60, topn=12) -> Dict[str, Any]:
    a = a or {}
    out: Dict[str, Any] = {}
    out["portfolio"] = a.get("portfolio")
    out["base_currency"] = a.get("base_currency")

    # Monte Carlo summary
    mc = a.get("mc_summary") or {}
    perf = mc.get("performance_summary") or {}
    exp_ann = mc.get("expected_annual_return") or {}
    probs = mc.get("annual_return_probabilities") or {}
    try:
        # perf["percentiles"] == [10,25,50,75,90]
        pct_list = perf.get("percentiles") or []
        eb = perf.get("end_balance") or []   # same order as pct_list
        tw = perf.get("twrr") or []          # CAGR over full horizon, same order

        # map -> {p10: {...}, p25: {...}, ...}
        pct_map = {}
        for i, q in enumerate(pct_list):
            key = f"p{int(q)}"
            pct_map[key] = {
                "end_balance": _round(eb[i]) if i < len(eb) else None,
                "cagr": _round(tw[i]) if i < len(tw) else None,
            }

        # expected CAGR (median over final horizon)
        # exp_ann = { horizons: [...], p10:[...], p25:[...], p50:[...], ... }
        horizons = (exp_ann.get("horizons") or [])
        exp_cagr = None
        if horizons and exp_ann.get("p50"):
            exp_cagr = _round(exp_ann["p50"][-1])

        # probability of negative year (H=1)
        # probs = { thresholds:[...], horizons:[...], matrix:[[...], ...] }
        prob_neg_year = None
        try:
            thr_idx = (probs.get("thresholds") or []).index(0.0)
            Hs = probs.get("horizons") or []
            if 1 in Hs:
                h_idx = Hs.index(1)
                prob_ge_0 = float((probs.get("matrix") or [])[thr_idx][h_idx])
                prob_neg_year = _round(1.0 - prob_ge_0)
        except Exception:
            pass

        out["mc_summary"] = {
            "assumptions": {
                "horizon_years": mc.get("horizon_years"),
                "n_sims": (mc.get("assumptions") or {}).get("n_sims"),
                "fees_bps": (mc.get("assumptions") or {}).get("fees_bps"),
                "inflation": (mc.get("assumptions") or {}).get("inflation"),
                "block_len": (mc.get("assumptions") or {}).get("block_len"),
            },
            "percentiles": pct_map,              # ← has end_balance + cagr for p10..p90
            "expected_cagr": exp_cagr,           # ← median expected CAGR at final horizon
            "prob_neg_year": prob_neg_year,      # ← 1 - P(annual >= 0) at H=1
        }
    except Exception:
        # fallback with minimal fields so the prompt never breaks
        out["mc_summary"] = {
            "assumptions": mc.get("assumptions"),
            "percentiles": None,
            "expected_cagr": None,
            "prob_neg_year": None,
        }
    
    # Efficient frontier → pick max Sharpe & min Vol from arrays
    ef = a.get("efficient_frontier") or {}
    vols = ef.get("volatilities") or []
    rets = ef.get("returns") or []
    sharpes = ef.get("sharpes") or []
    def _pick_max_idx(arr):
        try: return max(range(len(arr)), key=lambda k: arr[k])
        except Exception: return None
    def _pick_min_idx(arr):
        try: return min(range(len(arr)), key=lambda k: arr[k])
        except Exception: return None
    ms_i = _pick_max_idx(sharpes)
    mv_i = _pick_min_idx(vols)
    out["efficient_frontier"] = {
        "max_sharpe": ({"vol": _round(vols[ms_i]), "er": _round(rets[ms_i]), "sharpe": _round(sharpes[ms_i])} if ms_i is not None else None),
        "min_vol": ({"vol": _round(vols[mv_i]), "er": _round(rets[mv_i]), "sharpe": _round(sharpes[mv_i])} if mv_i is not None else None),
    }

    # Weights: map arrays -> {ticker: weight}
    tickers = a.get("tickers") or []
    ws = a.get("weights") or {}
    out["weights"] = {
        "user": _topn_weights(_arr_to_map(ws.get("user") or [], tickers), n=topn),
        "equal": _topn_weights(_arr_to_map(ws.get("equal") or [], tickers), n=topn),
        "optimized": _topn_weights(_arr_to_map(ws.get("optimized") or [], tickers), n=topn),
    }

    # Metrics: what your analytics actually exposes
    m = a.get("metrics") or {}
    uw = m.get("user_weighted") or {}
    out["metrics"] = {
        "cagr": _round(m.get("cagr")),
        "sharpe_user": _round(uw.get("sharpe")),
        "sortino_user": _round(uw.get("sortino")),
    }

    # Valuation & PnL table (top-N)
    val = a.get("valuation") or {}
    per_asset = val.get("per_asset") or []
    try:
        per_asset_sorted = sorted(per_asset, key=lambda r: abs(float(r.get("weight") or 0.0)), reverse=True)
    except Exception:
        per_asset_sorted = per_asset
    out["valuation"] = {
        "total_value_now": _round(val.get("total_value_now")),
        "total_pl_abs": _round(val.get("total_pl_abs")),
        "total_pl_pct": _round(val.get("total_pl_pct")),
        "per_asset_top": [
            {
                "symbol": r.get("symbol") or r.get("ticker") or r.get("name"),
                "name": r.get("name"),
                "asset_class": r.get("asset_class"),
                "sector": r.get("sector"),
                "weight": _round(r.get("weight")),
                "pl_abs": _round(r.get("pl_abs")),
                "pl_pct": _round(r.get("pl_pct")),
            } for r in per_asset_sorted[:topn]
        ],
        "count_assets": len(per_asset),
        # Aggregated exposures for narrative
        "exposure_by_sector": _aggregate_exposure(per_asset, "sector"),
        "exposure_by_asset_class": _aggregate_exposure(per_asset, "asset_class"),
    }

    # Time series (downsampled) + compute annual returns & portfolio MDD
    user_index = a.get("user_index") or []
    user_dates = a.get("user_index_dates") or []
    if isinstance(user_index, list) and isinstance(user_dates, list) and len(user_index) == len(user_dates):
        out["user_index"] = _ds(user_index, max_points=max_points)
        out["user_index_dates"] = _ds(user_dates, max_points=max_points)
        # annual returns from full arrays (no downsample here to keep accuracy)
        out["annual_returns_portfolio"] = _annual_returns_from_index(user_index, user_dates)
        # portfolio max drawdown (quick)
        try:
            peak, mdd = -1e18, 0.0
            for v in user_index:
                peak = max(peak, float(v))
                dd = (float(v) / peak) - 1.0 if peak else 0.0
                mdd = min(mdd, dd)
            out["portfolio_max_drawdown"] = _round(mdd)
        except Exception:
            out["portfolio_max_drawdown"] = None

    # Date window mirrors what Metrics tab uses
    out["date_window"] = a.get("date_window")
    return out

def _compact_risk(r: Dict[str, Any], max_points=60) -> Dict[str, Any]:
    r = r or {}
    out = {
        "var_95": _round(r.get("var_95")),
        "cvar_95": _round(r.get("cvar_95")),
        "max_drawdown": _round(r.get("max_drawdown")),
    }
    for key in ["drawdown", "rolling_vol", "rolling_sharpe", "dates"]:
        v = r.get(key)
        if isinstance(v, list):
            out[key] = _ds(v, max_points=max_points)
    return out

def _compact_benchmark(b: Dict[str, Any], max_points=60) -> Optional[Dict[str, Any]]:
    if not b: return None
    out = {
        "symbol": b.get("symbol"),
        "base": b.get("base"),
        "summary": b.get("summary"),  # { cagr, stdev_annual, max_drawdown, best_year, worst_year }
        "final_value": _round(b.get("final_value")),
        "final_profit_abs": _round(b.get("final_profit_abs")),
        "final_profit_pct": _round(b.get("final_profit_pct")),
        "years": _ds(b.get("years") or [], max_points=max_points),
        "returns": _ds(b.get("returns") or [], max_points=max_points),
    }
    return out






@app.post("/api/v1/portfolio/risk")
async def portfolio_risk(portfolio: Portfolio, years: int = 5):
    """
    Computes historical risk metrics (NaN-safe JSON):
      - rolling drawdown (aligned to returns dates)
      - daily returns (for histogram)
      - VaR/CVaR (95%)
      - rolling volatility (annualized)
      - rolling Sharpe
    """
    try:
        if not portfolio.holdings:
            raise HTTPException(status_code=400, detail="No holdings provided")

        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=365 * max(1, years))
        start_iso, end_iso = start.isoformat(), today.isoformat()

        # 1) Fetch local prices per ticker
        series = []
        qty = []
        tickers = []
        for h in portfolio.holdings:
            s = fetch_price(h.ticker, start_iso, end_iso)  # local currency
            if s is None or s.empty:
                continue
            series.append(s.ffill().dropna())
            qty.append(float(h.quantity or 0.0))
            tickers.append(h.ticker)

        if not series:
            raise HTTPException(status_code=400, detail="No valid price data for any holdings")

        df_loc = pd.concat(series, axis=1)
        df_loc.columns = tickers

        # 2) Convert each series -> base currency (align daily)
        base = (portfolio.base_currency or "USD").upper()
        conv_cols = []
        for sym in df_loc.columns:
            # currency from input (fallback to base)
            from_ccy = next((h.currency for h in portfolio.holdings if h.ticker == sym), None) or base
            fx = fetch_fx_series(from_ccy.upper(), base, start_iso, end_iso)  # factor local->base
            aligned = pd.concat([df_loc[sym], fx], axis=1).dropna()
            price_base = aligned.iloc[:, 0] * aligned.iloc[:, 1]
            price_base.name = sym
            conv_cols.append(price_base)

        if not conv_cols:
            raise HTTPException(status_code=400, detail="No FX-aligned series available")

        # Strict intersection to avoid gaps
        df = pd.concat(conv_cols, axis=1).dropna(how="any")
        if df.shape[0] < 3:
            raise HTTPException(status_code=400, detail="Insufficient aligned history")

        # 3) Build weights from latest values
        quantities = np.array([
            next((h.quantity for h in portfolio.holdings if h.ticker == t), 0.0)
            for t in df.columns
        ], dtype=float)
        last_vals = df.iloc[-1].values * quantities
        tv = float(np.sum(last_vals))
        if tv > 0 and np.isfinite(tv):
            w = last_vals / tv
        else:
            # fallback: equal weight
            w = np.ones(df.shape[1], dtype=float) / df.shape[1]

        # 4) Returns/index
        rets_df = df.pct_change().dropna()
        # Weighted daily returns
        port_ret = (rets_df.values @ w).astype(float)
        port_ret = pd.Series(port_ret, index=rets_df.index, name="ret")
        # Index for drawdown
        port_index = (1.0 + port_ret).cumprod()

        # 5) Drawdown aligned to returns dates
        peak = port_index.cummax()
        dd = (port_index / peak) - 1.0
        # align to returns index (already aligned)
        dd_aligned = dd.reindex(port_ret.index).fillna(0.0)

        # 6) VaR & CVaR (95%) on daily returns
        alpha = 0.05
        if len(port_ret) == 0:
            var_95 = None
            cvar_95 = None
        else:
            var_95 = float(np.percentile(port_ret.values, 100 * alpha))
            tail = port_ret.values[port_ret.values <= var_95]
            cvar_95 = float(np.mean(tail)) if tail.size else None

        # 7) Rolling stats (NaN-safe)
        win = 63  # ~ 3 months
        roll_std = port_ret.rolling(win).std(ddof=0)
        roll_vol_ann = roll_std * np.sqrt(252)

        roll_mean = port_ret.rolling(win).mean()
        # Sharpe = mean/std * sqrt(252); guard division-by-zero
        roll_sharpe = np.where(
            (roll_std.values > 0) & np.isfinite(roll_std.values),
            (roll_mean.values / roll_std.values) * np.sqrt(252),
            np.nan
        )
        roll_sharpe = pd.Series(roll_sharpe, index=port_ret.index)

        # 8) Build NaN-safe JSON
        dates = [d.strftime("%Y-%m-%d") for d in port_ret.index]
        return {
            "dates": dates,
            "returns": _safe_list(port_ret.values),
            "drawdown": _safe_list(dd_aligned.values),
            "rolling_vol": _safe_list(roll_vol_ann.values),
            "rolling_sharpe": _safe_list(roll_sharpe.values),
            "var_95": _sf(var_95),
            "cvar_95": _sf(cvar_95),
        }

    except HTTPException:
        raise
    except Exception as e:
        # Optional: show traceback only when REPORT_DEBUG=1
        if os.environ.get("REPORT_DEBUG","0") == "1":
            logger.error("portfolio_risk error:\n%s", traceback.format_exc())
            return PlainTextResponse(traceback.format_exc(), status_code=500)
        raise HTTPException(status_code=500, detail="Risk metrics failed")

@app.post("/api/v1/report")
def generate_report(body: ReportIn):
    if _openai_client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not configured on server")

    portfolio_name = (body.request or {}).get("name") or (body.analytics or {}).get("portfolio") or "Portfolio"
    base_ccy = (body.request or {}).get("base_currency") or (body.analytics or {}).get("base_currency") or "USD"

    # compact context (and auto-shrink if needed)
    max_points, topn = 60, 12
    ca = _compact_analytics(body.analytics, max_points=max_points, topn=topn)
    cr = _compact_risk(body.risk, max_points=max_points) if body.risk else None
    cb = _compact_benchmark(body.benchmark, max_points=max_points)

    def make_prompt(ca, cr, cb) -> str:
        return f"""
Write a detailed and Professional grade, neutral three-page Markdown report (~1,800 words) explaining the **actual portfolio results** in depth.  
Use the **real numbers** in the JSON (do not invent).  
Tone: clear, explanatory, data-driven — e.g., “Your portfolio’s Sharpe ratio is 1.36, which means that for each unit of risk you earn 1.36 units of return.”  
Avoid investment advice; explain implications only.

Structure the report in **three main sections** with short headings.

---

## Analytics — Monte Carlo and Allocations

1. **Monte Carlo Simulation:**  
   • Use the numeric percentiles (`p10 p25 p50 p75 p90`), expected CAGR, and probability of negative year from `mc_summary`.  
   • Explain what each percentile means (“there is a 10 % chance returns fall below X …”) and interpret how wide the range is (volatility of outcomes).  
   • Discuss what a higher/lower expected CAGR or probability of negative year implies for long-term growth and downside.  

2. **Efficient Frontier:**  
   • Compare the **max Sharpe** point (`er`, `vol`, `sharpe`) and **min vol** point.  
   • Explain what these values reveal about the risk/return efficiency of this portfolio (e.g., “Your max Sharpe 1.53 indicates strong risk-adjusted efficiency compared to 1.0 as a balanced benchmark”).  

3. **Weights and Exposures:**  
   • Describe **user**, **equal**, and **optimized** allocations with percentages (top N assets).  
   • Explain how the differences change exposure or diversification.  
   • Use `valuation.exposure_by_sector` and `valuation.exposure_by_asset_class` to narrate the sectoral and asset-class tilt — e.g., “53 % commodities and 45 % technology means results will be influenced by metal prices and tech earnings cycles.”  
   • Conclude this section with a paragraph linking concentration to potential cyclicality or rate-sensitivity (neutral, no advice).

---

## Risk — Downside and Stability

Use `var_95`, `cvar_95`, `max_drawdown`, and rolling series.

• Describe **maximum drawdown** numerically and explain what that loss depth means (“a –17 % drawdown implies that at the worst point the portfolio value dropped 17 % before recovery”).  
• Explain **VaR/CVaR (95 %)** in simple terms — the loss threshold and average loss in tail scenarios.  
• Discuss **rolling volatility** and **rolling Sharpe** trends: identify peaks/troughs, what high volatility means for short-term fluctuations, and what changing Sharpe indicates about consistency.  
• Conclude with 2–3 sentences about **stability vs. variability** — e.g., “Periods of higher volatility coincided with larger drawdowns, meaning the portfolio’s short-term path can deviate significantly from its long-term growth line.”

---

## Metrics — Performance vs Benchmark and Year-by-Year Results

1. **Portfolio Summary:**  
   • Report **CAGR, Sharpe, Sortino, and Max Drawdown** from analytics metrics and explain each in context:  
     “A CAGR of 22.4 % means the portfolio compounded 22 % annually on average; combined with a Sharpe 1.36, it shows good reward per unit of risk.”  
     Describe what the Sortino ratio adds (focus on downside).  

2. **Benchmark Comparison:**  
   • Use benchmark summary (`cagr`, `stdev_annual`, `max_drawdown`, `best_year`, `worst_year`).  
   • Compare each metric with the portfolio: higher/lower CAGR, volatility, and drawdown.  
   • Explain the implications neutrally — e.g., “While your portfolio’s CAGR 22 % exceeds the benchmark’s 10 %, its drawdown –17 % is larger, showing greater growth but higher swings.”  

3. **Annual Returns:**  
   • Use `annual_returns_portfolio` and benchmark annual arrays to describe how returns changed each year, highlighting best/worst years.  
   • Explain what drove the variation (market cycles, sector exposures) and what it reveals about consistency.  

4. **P&L Table and Contributors:**  
   • Use `valuation.per_asset_top` to list top contributors and detractors (symbol, weight, pl % and abs).  
   • Explain in plain language how each major holding affected total P&L (“Gold +12 % added $1.2 k to returns; Apple –5 % subtracted $600 ”) and how concentration of profits/losses affects overall stability.  

Conclude with a short paragraph summarizing what these results collectively indicate — the balance between growth, risk, and diversification — **without** suggesting any action.

---

### Context JSON (ready for interpretation)
ANALYTICS_COMPACT:
{json.dumps(ca, ensure_ascii=False)}

RISK_COMPACT:
{json.dumps(cr, ensure_ascii=False) if cr else "null"}

BENCHMARK_COMPACT:
{json.dumps(cb, ensure_ascii=False) if cb else "null"}

USER_NOTES:
{body.notes or ""}
""".strip()

    prompt = make_prompt(ca, cr, cb)

    # guardrail for prompt size
    if _estimate_tokens(prompt) > 12000:
        max_points, topn = 30, 8
        ca = _compact_analytics(body.analytics, max_points=max_points, topn=topn)
        cr = _compact_risk(body.risk, max_points=max_points) if body.risk else None
        cb = _compact_benchmark(body.benchmark, max_points=max_points)
        prompt = make_prompt(ca, cr, cb)

    model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")
    max_output_tokens = int(os.environ.get("REPORT_MAX_TOKENS", "2200"))

    last_err = None
    for attempt in range(3):
        try:
            resp = _openai_client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=max_output_tokens,
                messages=[
                    {"role": "system", "content": "You write neutral, precise portfolio reports. No advice; explain metrics clearly."},
                    {"role": "user", "content": prompt},
                ],
            )
            md = resp.choices[0].message.content or ""
            return {
                "ok": True,
                "portfolio": portfolio_name,
                "base_currency": base_ccy,
                "markdown": md,
                "meta": {"model": model, "length": len(md)}
            }
        except RateLimitError as e:
            last_err = e
            # further shrink once, then retry with small backoff
            if attempt == 0:
                max_points, topn = 20, 6
                ca = _compact_analytics(body.analytics, max_points=max_points, topn=topn)
                cr = _compact_risk(body.risk, max_points=max_points) if body.risk else None
                cb = _compact_benchmark(body.benchmark, max_points=max_points)
                prompt = make_prompt(ca, cr, cb)
            time.sleep(0.6 + 0.2 * attempt)
        except Exception as e:
            last_err = e
            break

    detail = getattr(last_err, "message", None) or str(last_err) or "Report generation failed"
    raise HTTPException(status_code=500, detail=f"Report generation failed: {detail}")


@app.get("/api/v1/benchmark")
def benchmark(
    symbol: str,
    base: str = Query("USD"),
    start: Optional[str] = None,
    end: Optional[str] = None,
    amount: Optional[float] = None,
    years: int = Query(10, ge=1, le=30),  # ← default 10-year window
):
    """
    Returns:
      {
        dates: [...],                # ISO dates aligned to price series
        index: [...],                # normalized to 1 at start (base-currency)
        value_path: [...],           # amount * index  (if 'amount' provided)
        final_value: float,          # if amount provided
        final_profit_abs: float,
        final_profit_pct: float,
        symbol: str,
        base: str,

        # For 10-year charts & tables:
        annual: {
          years: [YYYY, ...],
          benchmark: [r1, r2, ...],  # calendar-year returns as decimals
        },
        summary: {
          cagr: float,               # annualized return
          stdev_annual: float,       # annualized stdev (daily→annual)
          max_drawdown: float,       # min drawdown as negative decimal
          best_year: float,          # best calendar-year return
          worst_year: float          # worst calendar-year return
        }
      }
    """
    try:
        # --- 1) Resolve date window (years if start/end not given) ---
        today = datetime.now(timezone.utc).date()
        end_iso = end or today.isoformat()
        if start:
            start_iso = start
        else:
            start_iso = (today - timedelta(days=365 * years)).isoformat()

        # --- 2) Fetch local prices and FX → convert to base currency ---
        s_local = fetch_price(symbol, start_iso, end_iso)   # local currency series
        if s_local is None or len(s_local) == 0:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        from_ccy = (fetch_ticker_currency(symbol) or "USD").upper()
        fx = fetch_fx_series(from_ccy, base.upper(), start_iso, end_iso)  # factor local->base

        aligned = pd.concat([s_local, fx], axis=1).dropna()
        if aligned.empty:
            raise HTTPException(status_code=400, detail="No overlapping price/FX data")

        price_base = aligned.iloc[:, 0] * aligned.iloc[:, 1]  # in 'base' currency
        idx = (price_base / price_base.iloc[0]).astype(float) # normalized index (start=1)

        # --- 3) Optional value path from a fixed amount ---
        if amount is None:
            amount = 10_000.0  # sensible default for compare charts
        value_path = (idx * float(amount))

        final_value = float(value_path.iloc[-1])
        final_profit_abs = float(final_value - float(amount))
        final_profit_pct = float(final_value / float(amount) - 1.0)

        # --- 4) Annual returns & summary for 10Y visuals ---
        # Calendar-year returns using last trading day of each year
        price_year_end = price_base.resample("A").last()
        annual_rets = price_year_end.pct_change().dropna()

        # Daily returns for stdev & drawdown
        daily_rets = price_base.pct_change().dropna()

        # CAGR based on actual time span (not 252-count)
        dt_years = (price_base.index[-1] - price_base.index[0]).days / 365.25
        cagr = None
        if dt_years > 0 and price_base.iloc[0] > 0:
            cagr = float((price_base.iloc[-1] / price_base.iloc[0]) ** (1.0 / dt_years) - 1.0)

        # Annualized volatility (daily → annual)
        stdev_annual = float(daily_rets.std() * (252 ** 0.5)) if len(daily_rets) else None

        # Max drawdown
        cum_max = price_base.cummax()
        drawdown = (price_base / cum_max) - 1.0
        max_drawdown = float(drawdown.min()) if len(drawdown) else None

        best_year = float(annual_rets.max()) if len(annual_rets) else None
        worst_year = float(annual_rets.min()) if len(annual_rets) else None

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in idx.index],
            "index": idx.tolist(),
            "value_path": value_path.tolist(),
            "final_value": final_value,
            "final_profit_abs": final_profit_abs,
            "final_profit_pct": final_profit_pct,
            "symbol": symbol,
            "base": base.upper(),
            "annual": {
                "years": [int(d.year) for d in annual_rets.index],
                "benchmark": annual_rets.values.tolist(),
            },
            "summary": {
                "cagr": cagr,
                "stdev_annual": stdev_annual,
                "max_drawdown": max_drawdown,
                "best_year": best_year,
                "worst_year": worst_year,
            },
        }

    except HTTPException:
        raise
    except Exception:
        logger.error("benchmark error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Benchmark fetch failed")

@app.get("/api/v1/assets/search")
def assets_search(
    q: str,
    asset_class: Optional[str] = None,
    region: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    q = (q or "").strip()
    cls = _norm_class(asset_class or "")
    reg = _norm_region(region or "")

    qry = db.query(AssetDB)

    if cls:
        qry = qry.filter(AssetDB.asset_class == cls)

    if reg and reg.lower() != "any":
        qry = qry.filter(func.lower(AssetDB.region) == reg.lower())

    if q:
        like = f"%{q.lower()}%"
        qry = qry.filter(or_(func.lower(AssetDB.symbol).like(like),
                             func.lower(AssetDB.name).like(like)))

    rows = (qry.order_by(AssetDB.symbol.asc())
               .limit(min(max(limit, 1), 300))
               .all())

    return {"results": [
        dict(symbol=r.symbol, name=r.name, exchange=r.exchange,
             region=r.region, currency=r.currency, asset_class=r.asset_class)
        for r in rows
    ]}






@app.get("/api/v1/assets/resolve")
def assets_resolve(q: str):
    """
    Verifies a symbol exists and returns minimal metadata.
    """
    try:
        t = yf.Ticker(q)
        hist = t.history(period="5d")
        if hist is None or hist.empty:
            return {"ok": False}
        info = t.get_info() or {}
        return {
            "ok": True,
            "symbol": q.upper(),
            "name": info.get("shortName") or info.get("longName") or q.upper(),
            "exchange": info.get("exchange") or info.get("fullExchangeName"),
            "currency": info.get("currency"),
        }
    except Exception:
        return {"ok": False}

@app.post("/api/v1/auth/register", status_code=201)
def register(body: RegisterBody, db: Session = Depends(get_db)):
    try:
        if db.query(UserDB).filter(UserDB.email == body.email.lower()).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        user = UserDB(email=body.email.lower(), password_hash=hash_password(body.password))
        db.add(user); db.commit(); db.refresh(user)
        return {"id": user.id, "email": user.email}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")


@app.post("/api/v1/auth/login", response_model=TokenOut)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == form_data.username.lower()).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/v1/me", response_model=MeOut)
def me(user: UserDB = Depends(get_current_user)):
    return MeOut(id=user.id, email=user.email)

# =========================
# PORTFOLIO CRUD
# =========================
@app.post("/api/v1/portfolios", response_model=PortfolioOut)
def create_portfolio(p: Portfolio, user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = PortfolioDB(user_id=user.id, name=p.name, holdings_json=json.dumps([h.dict() for h in p.holdings]))
    db.add(rec); db.commit(); db.refresh(rec)
    return PortfolioOut(id=rec.id, name=rec.name, created_at=rec.created_at)

@app.get("/api/v1/portfolios", response_model=List[PortfolioOut])
def list_portfolios(user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(PortfolioDB).filter(PortfolioDB.user_id == user.id).order_by(PortfolioDB.created_at.desc()).all()
    return [PortfolioOut(id=r.id, name=r.name, created_at=r.created_at) for r in rows]

@app.get("/api/v1/portfolios/{pid}", response_model=PortfolioDetailOut)
def get_portfolio(pid: int, user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.get(PortfolioDB, pid)
    if not rec or rec.user_id != user.id:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    holdings = [Holding(**h) for h in json.loads(rec.holdings_json)]
    return PortfolioDetailOut(id=rec.id, name=rec.name, holdings=holdings, created_at=rec.created_at)

@app.put("/api/v1/portfolios/{pid}", response_model=PortfolioOut)
def update_portfolio(pid: int, p: Portfolio, user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.get(PortfolioDB, pid)
    if not rec or rec.user_id != user.id:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    rec.name = p.name
    rec.holdings_json = json.dumps([h.dict() for h in p.holdings])
    rec.updated_at = datetime.utcnow()
    db.commit(); db.refresh(rec)
    return PortfolioOut(id=rec.id, name=rec.name, created_at=rec.created_at)

@app.delete("/api/v1/portfolios/{pid}", status_code=204)
def delete_portfolio(pid: int, user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.get(PortfolioDB, pid)
    if not rec or rec.user_id != user.id:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    db.delete(rec); db.commit()
    return

# =========================
# EXISTING ANALYTICS & STRESS
# =========================
# --- replace the whole stress_test endpoint in main.py ---
@app.post("/api/v1/portfolio/stress")
async def stress_test(req: StressRequest):
    try:
        today = datetime.utcnow().date()
        start_dt = today - timedelta(days=365)
        start_iso, end_iso = start_dt.isoformat(), today.isoformat()

        # base currency (default USD if the caller didn't pass it)
        base = "USD"
        # try to infer from holdings payload shape (Portfolio object), else USD
        # If you want it mandatory, add base_currency into StressRequest and use it.
        # For now we check if any holding has currency and convert to USD; later safe to make paramized.
        # We'll still FX convert every leg to 'base'.
        # -- If you already send base_currency at caller side, replace this with that value.
        base = (getattr(req, "base_currency", None) or "USD").upper()

        # 1) Fetch local price series per holding
        series_map = {}
        ccys = {}
        for h in req.holdings:
            s = fetch_price(h.ticker, start_iso, end_iso)  # local currency
            if s is None or s.empty:
                continue
            series_map[h.ticker] = s.ffill().dropna()
            ccys[h.ticker] = (h.currency or base).upper()

        if not series_map:
            raise HTTPException(status_code=400, detail="No valid price data for any holdings.")

        # 2) FX convert each series -> base currency
        converted = {}
        for sym, s in series_map.items():
            from_ccy = ccys.get(sym, base).upper()
            fx = fetch_fx_series(from_ccy, base, start_iso, end_iso)  # factor local->base
            aligned = pd.concat([s, fx], axis=1).dropna()
            if aligned.empty:
                continue
            conv = aligned.iloc[:, 0] * aligned.iloc[:, 1]
            conv.name = sym
            converted[sym] = conv

        if not converted:
            raise HTTPException(status_code=400, detail="No FX-aligned series available.")

        # Strict intersection for simulation
        df_base = pd.concat(converted.values(), axis=1).dropna(how="any")
        if df_base.shape[1] == 0 or df_base.shape[0] < 3:
            raise HTTPException(status_code=400, detail="Insufficient aligned history.")

        tickers = list(df_base.columns)

        # 3) Build weights from latest base prices
        last_prices = df_base.iloc[-1].values
        quantities = np.array([next((h.quantity for h in req.holdings if h.ticker == t), 0.0) for t in tickers], dtype=float)
        values_now = last_prices * quantities
        total_value_now_base = float(np.sum(values_now))
        user_weights = (
            values_now / total_value_now_base if total_value_now_base > 0
            else np.repeat(1/len(values_now), len(values_now))
        )

        # Map UI shocks into simulator knobs
        day1 = {}
        random_shocks = []   # (ticker, pct, 1-day)
        firstk = []          # (ticker, pct, k_days)
        lastk = []           # (ticker, pct, k_days)

        for s in (req.scenario.shocks or []):
            pct = float(s.pct)
            # clamp falls so we don't go below zero price
            if pct < -0.99:
                pct = -0.99

            if s.mode == "day1":
                day1[s.ticker] = pct
            elif s.mode == "random":
                random_shocks.append((s.ticker, pct))
            elif s.mode == "first_k" and s.k_days and s.k_days > 0:
                firstk.append((s.ticker, pct, int(s.k_days)))
            elif s.mode == "last_k" and s.k_days and s.k_days > 0:
                lastk.append((s.ticker, pct, int(s.k_days)))

        # store day-1 shocks where the simulator already looks
        req.scenario.day1_shock_pct = day1

        # 4) Simulate on base-currency series (same engine)
        sims_index, summary = simulate_stress(df_base,
                                                user_weights,
                                                req.scenario,
                                                random_shocks=random_shocks,
                                                firstk=firstk,
                                                lastk=lastk,
                                                )

        # 5) Return BOTH index and value paths (value = index * starting value)
        paths_value = [[float(x) * total_value_now_base for x in path] for path in sims_index]

        return {
            "portfolio": req.name,
            "base_currency": base,
            "scenario": req.scenario.dict(),
            "summary": summary,
            "paths_index": sims_index[:50],   # keep subset for chart
            "paths_value": paths_value[:50],  # balances in base ccy
            "start_value": total_value_now_base,
            "tickers": tickers,
        }
    except Exception:
        tb = traceback.format_exc()
        logger.error("Stress test error: %s", tb)
        raise HTTPException(status_code=500, detail="Internal error")

@app.post("/api/v1/portfolio/analytics")
async def analytics(portfolio: Portfolio,
    years: int = Query(10, ge=1, le=30),          # lookback for history
    mc_years: int = Query(15, ge=1, le=50),       # forward horizon
    n_sims: int = Query(3000, ge=200, le=20000),  # simulation count
    fees_bps: float = Query(0.0, ge=0.0, le=500.0),
    inflation: float = Query(0.0, ge=0.0, le=0.20),
    block_len: int = Query(0, ge=0, le=252),      # 0 = off, else 5/21/42/63
):
    """
    Compute portfolio analytics in the requested base currency.

    Response includes:
      weights, metrics (user/equal/optimized + cagr), efficient_frontier,
      monte_carlo_paths, mc_summary, valuation (per-asset P/L),
      user_index (+dates), and 'failed' tickers.
    """
    try:
        # ---------- 0) Date window ----------
        today = datetime.now(timezone.utc).date()
        start_dt = today - timedelta(days=365 * years)
        start_iso, end_iso = start_dt.isoformat(), today.isoformat()

        base = (portfolio.base_currency or "USD").upper().strip()
        logger.info("Fetching data for %s (base=%s, years=%s)", portfolio.name, base, years)

        # ---------- 1) Fetch local price series per holding; SKIP failures ----------
        series_map: dict[str, pd.Series] = {}
        ccys: dict[str, str] = {}
        failed: list[dict[str, str]] = []

        for h in (portfolio.holdings or []):
            try:
                s = fetch_price(h.ticker, start_iso, end_iso)  # local currency
                if s is None or len(s) == 0:
                    raise ValueError("empty price series")
                series_map[h.ticker] = s.ffill().dropna()
                ccys[h.ticker] = (h.currency or base).upper()
            except Exception as e:
                failed.append({"ticker": h.ticker, "reason": str(e)})

        if not series_map:
            raise HTTPException(
                status_code=400,
                detail={"message": "No valid price data for any holdings.", "failed": failed},
            )

        # ---------- 2) FX convert each series -> base ----------
        converted: dict[str, pd.Series] = {}
        for sym, s in series_map.items():
            from_ccy = ccys.get(sym, base).upper()
            fx = fetch_fx_series(from_ccy, base, start_iso, end_iso)  # factor local->base
            aligned = pd.concat([s, fx], axis=1).dropna()
            if aligned.empty:
                failed.append({"ticker": sym, "reason": f"No FX alignment {from_ccy}->{base}"})
                continue
            conv = aligned.iloc[:, 0] * aligned.iloc[:, 1]
            conv.name = sym
            converted[sym] = conv

        if not converted:
            raise HTTPException(
                status_code=400,
                detail={"message": "All holdings failed after FX alignment.", "failed": failed},
            )

        # ---------- 3A) Strict intersection (for metrics/optimizer/stress) ----------
        df_base = pd.concat(converted.values(), axis=1).dropna(how="any")
        if df_base.shape[1] == 0 or df_base.shape[0] < 3:
            raise HTTPException(
                status_code=400,
                detail={"message": "Insufficient aligned history.", "failed": failed},
            )
        tickers = list(df_base.columns)

        # ---------- 3B) Outer-join (for dynamic MC lookback) ----------
        all_prices = pd.concat(converted.values(), axis=1, sort=True)
        all_prices.columns = list(converted.keys())
        all_prices = all_prices.sort_index()
        all_returns = all_prices.pct_change()  # will contain NaNs where price missing

        # ---------- 4) Returns & last prices (strict set) ----------
        returns_df = df_base.pct_change().dropna()
        last_prices_base = df_base.iloc[-1].values

        # ---------- 5) Compute user / equal weights on strict set ----------
        quantities = np.array([
            next((h.quantity for h in portfolio.holdings if h.ticker == t), 0.0) for t in tickers
        ], dtype=float)
        buy_prices_local = np.array([
            next((h.buy_price for h in portfolio.holdings if h.ticker == t), 0.0) for t in tickers
        ], dtype=float)

        # Convert buy prices to base using last available FX
        buy_fx = []
        for t in tickers:
            from_ccy = ccys.get(t, base).upper()
            fx_series = fetch_fx_series(from_ccy, base, start_iso, end_iso)
            buy_fx.append(fx_series.iloc[-1].item())
        buy_prices_base = buy_prices_local * np.array(buy_fx, dtype=float)

        values_now = last_prices_base * quantities
        total_value_now = float(np.sum(values_now))
        user_weights = (values_now / total_value_now) if total_value_now > 0 else np.repeat(1/len(tickers), len(tickers))
        equal_weights = np.repeat(1/len(tickers), len(tickers))

        # ---------- 6) Optimization & metrics (strict set) ----------
        opt_weights, opt_sharpe, frontier_data = optimize_portfolio(returns_df)
        user_metrics, user_index = portfolio_metrics(returns_df, user_weights)   # index ~ starts at 1
        equal_metrics, _ = portfolio_metrics(returns_df, equal_weights)
        opt_metrics, _ = portfolio_metrics(returns_df, opt_weights)

        # ---------- 7) P/L in base (per-asset + total) ----------
        invested_base = buy_prices_base * quantities
        total_invested = float(np.sum(invested_base))
        total_pl_abs = float(total_value_now - total_invested)
        total_pl_pct = float(total_value_now / total_invested - 1.0) if total_invested > 0 else None

        # Enrich with sector/region
        symbol_to_sector = {}
        with SessionLocal() as _db:
            rows = _db.query(AssetDB).filter(AssetDB.symbol.in_(tickers)).all()
            for r in rows:
                symbol_to_sector[r.symbol] = r.sector or None

        per_asset = []
        for i, sym in enumerate(tickers):
            inv = float(invested_base[i]); cur = float(values_now[i])
            pl_abs = float(cur - inv)
            pl_pct = float(cur / inv - 1.0) if inv > 0 else None
            per_asset.append({
                "symbol": sym,
                "quantity": float(quantities[i]),
                "buy_price_local": float(buy_prices_local[i]),
                "buy_fx_to_base": float(buy_fx[i]),
                "buy_price_base": float(buy_prices_base[i]),
                "last_price_base": float(last_prices_base[i]),
                "invested_base": inv,
                "current_value_base": cur,
                "pl_abs": pl_abs,
                "pl_pct": pl_pct,
                "weight": float(user_weights[i]),
                "region": next((h.country for h in portfolio.holdings if h.ticker == sym), None),
                "sector": symbol_to_sector.get(sym),
            })

        # ---------- 8) Dynamic daily portfolio returns (outer-join; long lookback) ----------
        # Build base weights from latest available prices across ALL tickers
        tickers_all = list(all_prices.columns)
        last_prices_all = all_prices.ffill().iloc[-1].values
        quantities_all = np.array([
            next((h.quantity for h in portfolio.holdings if h.ticker == t), 0.0)
            for t in tickers_all
        ], dtype=float)
        values_now_all = last_prices_all * quantities_all
        total_value_now_all = float(np.sum(values_now_all))
        base_weights_all = (
            values_now_all / total_value_now_all
            if total_value_now_all > 0 and np.isfinite(total_value_now_all)
            else (np.ones(len(tickers_all)) / max(len(tickers_all), 1))
        )

        R = all_returns.values                      # (T,N) with NaNs
        mask = np.isfinite(R)                       # True where we have a return
        W_raw = mask * base_weights_all             # broadcast (T,N)
        row_sums = W_raw.sum(axis=1, keepdims=True) # (T,1)
        W = np.divide(W_raw, row_sums, out=np.zeros_like(W_raw), where=row_sums > 0)
        R_filled = np.nan_to_num(R, nan=0.0)
        port_daily_dyn = (R_filled * W).sum(axis=1)
        port_daily_dyn = pd.Series(port_daily_dyn, index=all_returns.index).dropna()

        # ---------- 9) Monte Carlo (15y) from CURRENT value; clamp outputs for log-scale ----------
        start_amount = (
        total_value_now_all if total_value_now_all > 0
        else (total_value_now if total_value_now > 0 else 10_000.0)
    )

        MC = mc_forward_summary(
            port_daily_rets=port_daily_dyn,
            start_amount=start_amount,
            years=mc_years,
            n_sims=n_sims,
            fees_bps=fees_bps,
            inflation=inflation,
            block_len=block_len,
        )

        # Clamp for log-scale safety (unchanged)
        if MC and "percentile_paths" in MC:
            def _clamp_pos(a): return [max(1e-9, float(x)) for x in a]
            for k in ("p10","p25","p50","p75","p90"):
                MC["percentile_paths"][k] = _clamp_pos(MC["percentile_paths"][k])
                # also clamp real paths if present
                if "percentile_paths_real" in MC:
                    MC["percentile_paths_real"][k] = _clamp_pos(MC["percentile_paths_real"][k])
        

        # ---------- 10) CAGR from index (strict set; ~252 d/yr) ----------
        def compute_cagr(index_series: pd.Series):
            if index_series is None or len(index_series) < 2: return None
            first, last = float(index_series.iloc[0]), float(index_series.iloc[-1])
            if first <= 0: return None
            years_float = len(index_series) / 252.0
            if years_float <= 0: return None
            return (last / first) ** (1.0 / years_float) - 1.0

        cagr = compute_cagr(user_index)

        # ---------- 11) Build response ----------
        result = {
            "portfolio": portfolio.name,
            "base_currency": base,
            "weights": {
                "user": user_weights.tolist(),
                "equal": equal_weights.tolist(),
                "optimized": opt_weights.tolist(),
            },
            "metrics": {
                "user_weighted": user_metrics,
                "equal_weighted": equal_metrics,
                "optimized": opt_metrics,
                "cagr": cagr,
            },
            "valuation": {
                "total_value_now": total_value_now,
                "total_invested": total_invested,
                "total_pl_abs": total_pl_abs,
                "total_pl_pct": total_pl_pct,
                "per_asset": per_asset,
            },
            "tickers": tickers,
            "date_window": {"start": start_iso, "end": end_iso},
            "efficient_frontier": frontier_data,
            "monte_carlo_paths": (MC["percentile_paths"] if MC else None),  # keep compat
            "mc_summary": MC,
            "user_index": list(map(float, user_index.values)),
            "user_index_dates": [d.strftime("%Y-%m-%d") for d in user_index.index],
            "failed": failed,
        }
        return result

    except HTTPException:
        raise
    except Exception:
        logger.error("Error in portfolio analytics:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal error")




