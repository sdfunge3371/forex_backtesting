from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from ConBacktester import ConBacktester
from MeanRevBacktester import BollingerBacktester
from MoBacktester import MoBacktester

# --- CORS 設定 ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5500",
    "*"
]

plt.style.use("seaborn-v0_8")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 共用類別 ---
# class BacktestRequest(BaseModel):
#     symbol: str = "EURUSD=X"
#     sma_s: int = 50
#     sma_l: int = 200
#     start_date: str = "2004-01-01"
#     end_date: str = "2020-06-30"

# class OptimizeRequest(BacktestRequest):
#     sma_s_range_start: int = 10
#     sma_s_range_end: int = 100
#     sma_s_range_step: int = 10
#     sma_l_range_start: int = 150
#     sma_l_range_end: int = 250
#     sma_l_range_step: int = 10

class ConBacktestRequest(BaseModel):
    symbol: str
    start: str
    end: str
    tc: float
    window: int

class BollingerBacktestRequest(BaseModel):
    symbol: str
    SMA: int
    dev: float
    start: str
    end: str
    tc: float

class MoBacktestRequest(BaseModel):
    symbol: str
    start: str
    end: str
    tc: float
    window: int

# --- API路徑 ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Backtesting API. Available endpoints: /backtest/contrarian, /backtest/meanreversion"}


@app.post("/backtest/contrarian")
async def run_contrarian_backtest(request: ConBacktestRequest):
    try:
        tester = ConBacktester(request.symbol, request.start, request.end, request.tc)
        perf, outperf = tester.test_strategy(window=request.window)
        df = tester.results.reset_index()
        df["time"] = df["time"].astype(str)

        data = tester.results.copy()
        rf = 0.04
        n_per_day = 4  # 每天4筆
        rf_daily = rf / (252 * n_per_day)
        N = 252 * n_per_day
        excess_returns = data["strategy"] - rf_daily
        sharpe = np.sqrt(N) * excess_returns.mean() / excess_returns.std()
        win_rate = (data["strategy"] > 0).sum() / len(data)
        trades = (data["position"].diff().abs() > 0).sum()

        rolling_max = data["cstrategy"].cummax()
        drawdown = (data["cstrategy"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        max_single_loss = data["strategy"].min()
        losses = data["strategy"] < 0
        consec_losses = losses.astype(int).groupby((~losses).cumsum()).sum()
        max_consec_losses = consec_losses.max()

        return {
            "symbol": request.symbol,
            "start": request.start,
            "end": request.end,
            "tc": request.tc,
            "window": request.window,
            "performance": perf,
            "outperformance": outperf,
            "results_data": df.to_dict(orient="records"),
            "sharpe": round(sharpe, 4),
            "win_rate": round(win_rate * 100, 2),
            "trades": int(trades),
            "max_drawdown": round(max_drawdown * 100, 2),
            "max_single_loss": round(max_single_loss, 4),
            "max_consec_losses": int(max_consec_losses),
        }
    except Exception as e:
        print("錯誤發生：", str(e))
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    

@app.post("/backtest/momentum")
async def run_momentum_backtest(request: MoBacktestRequest):
    try:
        print("收到請求內容：", request.dict())
        tester = MoBacktester(request.symbol, request.start, request.end, request.tc)
        perf, outperf = tester.test_strategy(window=request.window)
        df = tester.results.reset_index()
        df["time"] = df["time"].astype(str)

        data = tester.results.copy()
        rf = 0.04
        n_per_day = 4  # 每天4筆
        rf_daily = rf / (252 * n_per_day)
        N = 252 * n_per_day
        excess_returns = data["strategy"] - rf_daily
        sharpe = np.sqrt(N) * excess_returns.mean() / excess_returns.std()
        win_rate = (data["strategy"] > 0).sum() / len(data)
        trades = (data["position"].diff().abs() > 0).sum()

        rolling_max = data["cstrategy"].cummax()
        drawdown = (data["cstrategy"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        max_single_loss = data["strategy"].min()
        losses = data["strategy"] < 0
        consec_losses = losses.astype(int).groupby((~losses).cumsum()).sum()
        max_consec_losses = consec_losses.max()

        return {
            "symbol": request.symbol,
            "start": request.start,
            "end": request.end,
            "tc": request.tc,
            "window": request.window,
            "performance": perf,
            "outperformance": outperf,
            "results_data": df.to_dict(orient="records"),
            "sharpe": round(sharpe, 4),
            "win_rate": round(win_rate * 100, 2),
            "trades": int(trades),
            "max_drawdown": round(max_drawdown * 100, 2),
            "max_single_loss": round(max_single_loss, 4),
            "max_consec_losses": int(max_consec_losses),
        }
    except Exception as e:
        print("錯誤發生：", str(e))
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    
@app.post("/backtest/bollinger")
async def run_meanrev_backtest(request: BollingerBacktestRequest):
    try:
        tester = BollingerBacktester(request.symbol, request.SMA, request.dev, request.start, request.end, request.tc)
        perf, outperf = tester.test_strategy()
        df = tester.results.reset_index()
        df["time"] = df["time"].astype(str)

        data = tester.results.copy()
        rf = 0.04
        n_per_day = 4  # 每天4筆
        rf_daily = rf / (252 * n_per_day)
        N = 252 * n_per_day
        excess_returns = data["strategy"] - rf_daily
        sharpe = np.sqrt(N) * excess_returns.mean() / excess_returns.std()
        win_rate = (data["strategy"] > 0).sum() / len(data)
        trades = (data["position"].diff().abs() > 0).sum()

        rolling_max = data["cstrategy"].cummax()
        drawdown = (data["cstrategy"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        max_single_loss = data["strategy"].min()
        losses = data["strategy"] < 0
        consec_losses = losses.astype(int).groupby((~losses).cumsum()).sum()
        max_consec_losses = consec_losses.max()

        return {
            "symbol": request.symbol,
            "SMA": request.SMA,
            "dev": request.dev,
            "tc": request.tc,
            "start": request.start,
            "end": request.end,
            "performance": perf,
            "outperformance": outperf,
            "results_data": df.to_dict(orient="records"),
            "sharpe": round(sharpe, 4),
            "win_rate": round(win_rate * 100, 2),
            "trades": int(trades),
            "max_drawdown": round(max_drawdown * 100, 2),
            "max_single_loss": round(max_single_loss, 4),
            "max_consec_losses": int(max_consec_losses),
        }
    except Exception as e:
        print("錯誤發生：", str(e))
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
