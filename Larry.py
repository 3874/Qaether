import pandas as pd
import numpy as np
from datetime import datetime
from pykrx import stock

def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Williams %R for given OHLC dataframe.
    df must have columns ['고가', '저가', '종가'].
    """
    highest_high = df['고가'].rolling(window=period).max()
    lowest_low = df['저가'].rolling(window=period).min()
    williams_r = (highest_high - df['종가']) / (highest_high - lowest_low) * -100
    return williams_r

def compute_ema(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Exponential Moving Average (EMA) for given OHLC dataframe.
    df must have column ['종가'].
    """
    return df['종가'].ewm(span=period, adjust=False).mean()

def fetch_all_krx_tickers(date: str) -> list:
    """
    Fetch all KRX tickers (KOSPI & KOSDAQ combined) for the given date (YYYYMMDD).
    """
    return stock.get_market_ticker_list(date)

def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker between start and end dates (YYYYMMDD).
    """
    return stock.get_market_ohlcv_by_date(start, end, ticker)

def main():
    # 설정: 기간과 날짜 계산
    period = 14
    today = datetime.now().strftime("%Y%m%d")
    start_date = (pd.to_datetime(today) - pd.Timedelta(days=period*2)).strftime("%Y%m%d")
    
    tickers = fetch_all_krx_tickers(today)
    results = []

    for ticker in tickers:
        df = fetch_ohlcv(ticker, start_date, today)
        if len(df) < period:
            continue
        willr = compute_williams_r(df, period)
        ema = compute_ema(df, period)
        latest_r = willr.iloc[-1]
        latest_ema = ema.iloc[-1]
        status = (
            "Overbought" if latest_r > -20
            else "Oversold" if latest_r < -80
            else "Neutral"
        )
        results.append({
            "Ticker": ticker,
            "Name": stock.get_market_ticker_name(ticker),
            "Williams %R": round(latest_r, 2),
            "EMA": round(latest_ema, 2),
            "Status": status
        })

    result_df = pd.DataFrame(results)
    # 결과 출력
    print(result_df)  # display 대신 print 사용 (Jupyter가 아닐 때)
    # CSV로 저장
    result_df.to_csv("williams_r_results.csv", index=False, encoding="utf-8-sig")
    print("\nResults saved to williams_r_results.csv")

if __name__ == "__main__":
    main()
