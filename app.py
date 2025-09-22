# ======================================================================
# DSSMM Streamlit App — Static by default (LIVE block commented below)
# ======================================================================

import os
import re
import uuid
import datetime as dt
from datetime import date
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Iterable

# --- Load .env so users can edit credentials there (optional for LIVE mode) ---
from dotenv import load_dotenv
load_dotenv()  # reads .env into process env

import pandas as pd
import streamlit as st
import pandas_gbq
from pandas_gbq import to_gbq

import yfinance as yf
from dateutil import parser as dateparser
from google.cloud import bigquery

# -------------- Streamlit page config --------------
st.set_page_config(page_title="DSSMM", layout="wide", page_icon=":material/finance_mode:")
st.title(":violet[D]etecting :green[Statistically Significant] :red[M]arket :red[M]oves with :blue[BigQuery AI]")

# -------------- Top disclaimer --------------
st.warning(
    "Disclaimer: The prototype currently shows pre-calculated values (from a "
    "[Kaggle Notebook](https://www.kaggle.com/code/kushtrivedi14728/noise-to-decision-with-bigquery-ai)) for demonstration purposes. "
    "To view live results, clone the [repository](https://github.com/Kush-Trivedi/DSSMM) and follow the configuration steps in the README."
)

# -------------- Format helpers --------------
def _pct(x, digits=2):
    if x is None or pd.isna(x): return "—"
    return f"{x*100:.{digits}f}%"

def _sigma(x, digits=2):
    if x is None or pd.isna(x): return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}σ"

def _num(x, digits=3):
    if x is None or pd.isna(x): return "—"
    return f"{x:.{digits}f}"

def _conf_delta(conf):
    if conf is None or pd.isna(conf): return "—"
    return f"confidence {_pct(conf, 0)}"

# ============================================================
# LIVE RUN — uncomment to enable (uses your GCP credits)
# Steps:
#  1) Edit your `.env` with GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID,
#     GCP_LOCATION, BQ_DATASET, CONNECTION_ID.
#  2) Uncomment below to enable live run.
# ============================================================

# # --- LIVE RUN (uncomment to enable) ---

# try:
#     from IPython.display import display  # noqa: F401
#     _HAS_DISPLAY = True
# except Exception:
#     _HAS_DISPLAY = False


# # =========================
# # BigQuery Config & Client
# # =========================
# @dataclass
# class BQConfig:
#     project_id: str = os.getenv("GCP_PROJECT_ID")
#     location:   str = os.getenv("GCP_LOCATION")
#     dataset_id: str = os.getenv("BQ_DATASET")
#     connection_id: str = os.getenv("CONNECTION_ID")


# # =========================
# # MarketSignalPipeline
# # =========================
# class MarketSignalPipeline:
#     """
#     Parameterized wrapper around the end-to-end pipeline.
#     """

#     def __init__(
#         self,
#         cfg: Optional[BQConfig] = None,
#         news_lookback_days: int = 120,
#         cosine_min: float = 0.60,
#         window_days: int = 2,
#         news_feeds: Optional[Dict[str, str]] = None
#     ):
#         self.cfg = cfg or BQConfig()
#         self.PROJECT_ID = self.cfg.project_id
#         self.LOCATION   = self.cfg.location
#         self.DATASET_ID = self.cfg.dataset_id
#         self.BQDFT      = f"`{self.PROJECT_ID}.{self.DATASET_ID}`"

#         self.client = bigquery.Client(project=self.PROJECT_ID, location=self.LOCATION)
#         self.client.create_dataset(
#             bigquery.Dataset(f"{self.PROJECT_ID}.{self.DATASET_ID}"),
#             exists_ok=True
#         )
#         print("✅ BigQuery ready:", self.PROJECT_ID, self.DATASET_ID, "in", self.LOCATION)

#         # Connection candidates
#         region_lower = (self.LOCATION or "").lower()
#         self.CANDIDATES = [f"{region_lower}.llm-connection"] if region_lower else []

#         # Connection id from external connection of BigQuery (can be preset via env)
#         self.CONNECTION_ID = self.cfg.connection_id

#         # GEN AI Models 
#         self.EMB_MODEL: Optional[str] = None
#         self.GEN_MODEL: Optional[str] = None

#         # Tunables
#         self.NEWS_LOOKBACK_DAYS = news_lookback_days
#         self.COSINE_MIN         = cosine_min
#         self.WINDOW_DAYS        = window_days

#         # News feeds (public) — can be swapped to premium in prod
#         self.NEWS_FEEDS: Dict[str, str] = news_feeds or {
#             "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
#             "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
#             "Yahoo Finance": "https://finance.yahoo.com/rss/topstories",
#         }

#         # Keywords (set per-run via set_keywords / run_all)
#         self.TICKER_KEYWORDS: Dict[str, List[str]] = {}

#     # ------------------------- #
#     #      Setup / Models       #
#     # ------------------------- #
#     def try_create_models(self, conn_id: str) -> bool:
#         try:
#             ddl = f"""
#             CREATE SCHEMA IF NOT EXISTS `{self.PROJECT_ID}.{self.DATASET_ID}` OPTIONS(location='{self.LOCATION}');
    
#             CREATE OR REPLACE MODEL `{self.PROJECT_ID}.{self.DATASET_ID}.ms_gen`
#               REMOTE WITH CONNECTION `{conn_id}`
#               OPTIONS (endpoint = 'gemini-2.5-flash');
    
#             CREATE OR REPLACE MODEL `{self.PROJECT_ID}.{self.DATASET_ID}.ms_embed`
#               REMOTE WITH CONNECTION `{conn_id}`
#               OPTIONS (endpoint = 'text-embedding-004');
#             """
#             self.client.query(ddl).result()
#             print("✅ Remote models created via connection:", conn_id)
#             return True
#         except Exception as e:
#             print(f"⚠️ Could not create models with {conn_id}: {e}")
#             return False

#     def setup_models(self) -> None:
#         # If a specific connection id is provided, try it first
#         tried_any = False
#         if self.CONNECTION_ID:
#             tried_any = True
#             if self.try_create_models(self.CONNECTION_ID):
#                 pass
#             else:
#                 self.CONNECTION_ID = None

#         # Else try candidates derived from region
#         if not self.CONNECTION_ID:
#             for c in self.CANDIDATES:
#                 tried_any = True
#                 if self.try_create_models(c):
#                     self.CONNECTION_ID = c
#                     break

#         if self.CONNECTION_ID is None or not tried_any:
#             raise RuntimeError("No working external connection found. Set CONNECTION_ID env or create <region>.llm-connection with Vertex AI permissions.")

#         self.EMB_MODEL = f"{self.PROJECT_ID}.{self.DATASET_ID}.ms_embed"
#         self.GEN_MODEL = f"{self.PROJECT_ID}.{self.DATASET_ID}.ms_gen"
#         print("EMB_MODEL =", self.EMB_MODEL)
#         print("GEN_MODEL =", self.GEN_MODEL)

#     # ------------------------- #
#     #  Price loading & returns  #
#     # ------------------------- #
#     def _prices_table_exists(self) -> bool:
#         try:
#             self.client.get_table(f"{self.PROJECT_ID}.{self.DATASET_ID}.prices")
#             return True
#         except Exception:
#             return False

#     def load_prices_to_bq(
#         self,
#         ticker: str,
#         benchmark: str = "^GSPC",
#         start: str = "2015-01-01",
#         end: Optional[str] = None,
#         if_exists_strategy: Optional[str] = None,
#     ) -> str:
#         end = end or dt.date.today().isoformat()
#         df = yf.download([ticker, benchmark], start=start, end=end, auto_adjust=True, progress=False)
#         if df.empty:
#             raise ValueError(f"No price data returned for {ticker}. Check symbols or dates.")
#         df = df["Close"].reset_index().rename(columns={ticker: "close", benchmark: "bench_close"})
#         df["ticker"] = ticker
#         df["benchmark"] = benchmark
#         df.rename(columns={"Date": "date"}, inplace=True)

#         if_exists = if_exists_strategy or ("append" if self._prices_table_exists() else "replace")
#         pandas_gbq.to_gbq(df, f"{self.DATASET_ID}.prices", project_id=self.PROJECT_ID, if_exists=if_exists)
#         table = f"{self.PROJECT_ID}.{self.DATASET_ID}.prices"
#         print(f"✅ Loaded prices to: {table}  (mode={if_exists})")
#         return table

#     def load_prices_many(
#         self,
#         tickers: Iterable[str],
#         benchmark: str = "^GSPC",
#         start: str = "2018-01-01",
#         end: Optional[str] = None,
#     ) -> None:
#         tickers = list(tickers)
#         if not tickers:
#             raise ValueError("No tickers provided.")
#         for i, t in enumerate(tickers):
#             self.load_prices_to_bq(
#                 ticker=t,
#                 benchmark=benchmark,
#                 start=start,
#                 end=end,
#                 if_exists_strategy=("replace" if i == 0 and not self._prices_table_exists() else "append"),
#             )

#     def build_daily_returns(self) -> None:
#         sql_returns = f"""
#         CREATE OR REPLACE TABLE {self.BQDFT}.daily_returns AS
#         WITH base AS (
#           SELECT
#             ticker,
#             DATE(date) AS date,
#             close,
#             bench_close,
#             SAFE_DIVIDE(close - LAG(close) OVER (PARTITION BY ticker ORDER BY date),
#                         LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS r,
#             SAFE_DIVIDE(bench_close - LAG(bench_close) OVER (PARTITION BY ticker ORDER BY date),
#                         LAG(bench_close) OVER (PARTITION BY ticker ORDER BY date)) AS r_mkt
#           FROM {self.BQDFT}.prices
#         )
#         SELECT
#           *,
#           STDDEV_SAMP(r)     OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS vol20,
#           STDDEV_SAMP(r_mkt) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) AS vol_mkt120,
#           SAFE_DIVIDE(r, NULLIF(STDDEV_SAMP(r) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),0)) AS z_day
#         FROM base;
#         """
#         self.client.query(sql_returns).result()
#         print("✅ daily_returns created")

#     def build_linear_model(self) -> None:
#         create_model_sql = f"""
#         CREATE OR REPLACE MODEL {self.BQDFT}.mm_linear_regression
#         OPTIONS(model_type='linear_reg', input_label_cols=['r']) AS
#         SELECT r, r_mkt
#         FROM {self.BQDFT}.daily_returns
#         WHERE r IS NOT NULL AND r_mkt IS NOT NULL;
#         """
#         self.client.query(create_model_sql).result()
#         print("✅ Model created")

#     def build_abnormal_returns(self) -> None:
#         sql_ar = f"""
#         CREATE OR REPLACE TABLE {self.BQDFT}.abnormal_returns AS
#         WITH pred AS (
#           SELECT
#             ticker, date, r, r_mkt,
#             predicted_r AS r_hat
#           FROM ML.PREDICT(
#             MODEL {self.BQDFT}.mm_linear_regression,
#             (SELECT ticker, date, r, r_mkt
#              FROM {self.BQDFT}.daily_returns
#              WHERE r IS NOT NULL AND r_mkt IS NOT NULL)
#           )
#         ),
#         stats AS (
#           SELECT
#             ticker,
#             date,
#             r,
#             r_mkt,
#             r_hat,
#             (r - r_hat) AS ar,
#             AVG(r) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) AS mean_r_120,
#             STDDEV_SAMP(r) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) AS sd_r_120,
#             STDDEV_SAMP(r - r_hat) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 120 PRECEDING AND 1 PRECEDING) AS sd_ar_120
#           FROM pred
#         )
#         SELECT
#           ticker,
#           date,
#           r,
#           r_mkt,
#           r_hat,
#           ar,
#           SAFE_DIVIDE(ar, sd_ar_120) AS z_ar,
#           SAFE_DIVIDE(r - mean_r_120, sd_r_120) AS z_day
#         FROM stats;
#         """
#         self.client.query(sql_ar).result()
#         print("✅ abnormal_returns (with z_day) created")

#     def build_trend_10d(self) -> None:
#         sql_trend = f"""
#         CREATE OR REPLACE TABLE {self.BQDFT}.trend_10d AS
#         WITH base AS (
#           SELECT
#             ticker,
#             date,
#             close,
#             SAFE.LOG(close) AS ln_p,
#             ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS t
#           FROM {self.BQDFT}.prices
#           WHERE close IS NOT NULL AND close > 0
#         ),
#         roll AS (
#           SELECT
#             ticker,
#             date,
#             COUNT(*)        OVER w AS n,
#             SUM(t)          OVER w AS sum_t,
#             SUM(ln_p)       OVER w AS sum_y,
#             SUM(t*t)        OVER w AS sum_tt,
#             SUM(ln_p*t)     OVER w AS sum_ty,
#             SUM(ln_p*ln_p)  OVER w AS sum_yy
#           FROM base
#           WINDOW w AS (
#             PARTITION BY ticker
#             ORDER BY date
#             ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
#           )
#         ),
#         fit AS (
#           SELECT
#             ticker,
#             date,
#             n,
#             (sum_tt - (sum_t*sum_t)/CAST(n AS FLOAT64)) AS sxx,
#             (sum_ty - (sum_t*sum_y)/CAST(n AS FLOAT64)) AS sxy,
#             (sum_yy - (sum_y*sum_y)/CAST(n AS FLOAT64)) AS syy
#           FROM roll
#         ),
#         stats AS (
#           SELECT
#             ticker,
#             date,
#             n,
#             sxx,
#             sxy,
#             syy,
#             SAFE_DIVIDE(sxy, sxx) AS slope,
#             CASE
#               WHEN n >= 3 AND sxx > 0 THEN
#                 SAFE_DIVIDE(syy - SAFE_DIVIDE(sxy*sxy, sxx), n - 2)
#             END AS s2
#           FROM fit
#         )
#         SELECT
#           ticker,
#           date,
#           n,
#           slope,
#           s2,
#           CASE
#             WHEN n >= 3 AND sxx > 0 AND s2 IS NOT NULL
#               THEN SAFE_DIVIDE(slope, SQRT(SAFE_DIVIDE(s2, sxx)))
#             ELSE NULL
#           END AS t_stat
#         FROM stats;
#         """
#         self.client.query(sql_trend).result()
#         print("✅ trend_10d created")

#     # -------------------------
#     #       News ingest
#     # -------------------------
#     @staticmethod
#     def _parse_ts(x: Any) -> dt.datetime:
#         try:
#             ts = dateparser.parse(x)
#             return ts if ts and ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
#         except Exception:
#             return dt.datetime.now(dt.timezone.utc)

#     @staticmethod
#     def _ensure_feedparser_installed() -> None:
#         try:
#             import feedparser  # noqa: F401
#         except Exception:
#             import subprocess, sys
#             subprocess.check_call([sys.executable, "-m", "pip", "install", "feedparser", "-q"])

#     def load_news_raw(self) -> None:
#         self._ensure_feedparser_installed()
#         import feedparser  # type: ignore

#         items: List[Dict[str, Any]] = []
#         for src, url in self.NEWS_FEEDS.items():
#             feed = feedparser.parse(url)
#             if not feed or not getattr(feed, "entries", None):
#                 continue
#             for e in feed.entries[:50]:
#                 title = (e.get("title","") or "").strip()
#                 summary = re.sub("<[^>]+>", " ", e.get("summary","") or "").strip()
#                 ts = self._parse_ts(e.get("published", e.get("updated", "")))
#                 items.append({
#                     "id": str(uuid.uuid4()),
#                     "date": ts,
#                     "ticker": None,
#                     "title": title[:500],
#                     "content": summary[:4000],
#                     "source": src,
#                     "url": e.get("link",""),
#                 })

#         df_news = pd.DataFrame(items)
#         pandas_gbq.to_gbq(df_news, f"{self.DATASET_ID}.news_raw", project_id=self.PROJECT_ID, if_exists="replace")
#         print("✅ news_raw loaded:", len(df_news))

#     # -------------------------
#     #  Embeddings & similarity
#     # -------------------------
#     def build_news_embeddings(self) -> None:
#         assert self.EMB_MODEL, "Call setup_models() first."
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.news_embeddings` AS
#         SELECT
#           t.id,
#           t.date,
#           t.ticker,
#           t.title,
#           t.ml_generate_embedding_result AS embedding
#         FROM ML.GENERATE_EMBEDDING(
#           MODEL `{self.EMB_MODEL}`,
#           (
#             SELECT
#               id,
#               date,
#               ticker,
#               title,
#               CONCAT(IFNULL(title,''), '\\n', IFNULL(content,'')) AS content
#             FROM `{self.PROJECT_ID}.{self.DATASET_ID}.news_raw`
#             WHERE (title IS NOT NULL OR content IS NOT NULL)
#               AND DATE(date) >= DATE_SUB(CURRENT_DATE(), INTERVAL {self.NEWS_LOOKBACK_DAYS} DAY)
#           )
#         ) AS t;
#         """).result()
#         print("✅ news_embeddings rebuilt")

#     def build_ticker_embeddings(self) -> None:
#         assert self.EMB_MODEL, "Call setup_models() first."
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_blobs` AS
#         WITH base AS (
#           SELECT
#             ticker,
#             date,
#             PERCENTILE_CONT(z_day, 0.5) OVER (PARTITION BY ticker) AS med_z_day,
#             PERCENTILE_CONT(z_ar,  0.5) OVER (PARTITION BY ticker) AS med_z_ar,
#             ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.abnormal_returns`
#         )
#         SELECT
#           ticker,
#           CONCAT(
#             'Ticker ', ticker, ' recent context. ',
#             'Median z_day=', FORMAT('%+.2f', med_z_day),
#             ', median z_ar=', FORMAT('%+.2f', med_z_ar)
#           ) AS blob
#         FROM base
#         WHERE rn = 1;
#         """).result()

#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_embeddings_raw` AS
#         SELECT *
#         FROM ML.GENERATE_EMBEDDING(
#           MODEL `{self.EMB_MODEL}`,
#           (
#             SELECT
#               ticker,
#               CAST(blob AS STRING) AS content
#             FROM `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_blobs`
#             WHERE blob IS NOT NULL
#           )
#         );
#         """).result()

#         rows = list(self.client.query(f"""
#         SELECT column_name, data_type
#         FROM `{self.PROJECT_ID}.{self.DATASET_ID}.INFORMATION_SCHEMA.COLUMNS`
#         WHERE table_name = 'ticker_embeddings_raw'
#         ORDER BY ordinal_position
#         """).result())
#         tick_emb_col = None
#         for r in rows:
#             if "embedding" in (r["column_name"] or "").lower() and (r["data_type"] or "").startswith(("ARRAY<","VECTOR<")):
#                 tick_emb_col = r["column_name"]; break
#         if tick_emb_col is None:
#             raise ValueError("No embedding column found in ticker_embeddings_raw.")

#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_embeddings` AS
#         SELECT ticker, `{tick_emb_col}` AS embedding
#         FROM `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_embeddings_raw`;
#         """).result()
#         print("✅ ticker_embeddings ready")

#     def _detect_news_embedding_col(self) -> str:
#         rows = list(self.client.query(f"""
#         SELECT column_name, data_type
#         FROM `{self.PROJECT_ID}.{self.DATASET_ID}.INFORMATION_SCHEMA.COLUMNS`
#         WHERE table_name = 'news_embeddings'
#         ORDER BY ordinal_position
#         """).result())
#         news_emb_col = None
#         for r in rows:
#             if "embedding" in (r["column_name"] or "").lower() and (r["data_type"] or "").startswith(("ARRAY<","VECTOR<")):
#                 news_emb_col = r["column_name"]; break
#         if news_emb_col is None:
#             raise ValueError("No embedding column found in news_embeddings.")
#         return news_emb_col

#     def rebuild_similarity(self) -> None:
#         news_emb_col = self._detect_news_embedding_col()
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_similarity` AS
#         WITH src AS (
#           SELECT
#             n.id AS news_id,
#             t.ticker AS ticker,
#             CAST(n.{news_emb_col} AS ARRAY<FLOAT64>) AS n_emb,
#             CAST(t.embedding     AS ARRAY<FLOAT64>) AS t_emb
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.news_embeddings` n
#           CROSS JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_embeddings` t
#         ),
#         scored AS (
#           SELECT
#             news_id,
#             ticker,
#             SAFE_DIVIDE(
#               (SELECT SUM(ne * te)
#                  FROM UNNEST(n_emb) AS ne WITH OFFSET i
#                  JOIN UNNEST(t_emb) AS te WITH OFFSET j
#                    ON i = j),
#               (
#                 SQRT((SELECT SUM(ne * ne) FROM UNNEST(n_emb) AS ne)) *
#                 SQRT((SELECT SUM(te * te) FROM UNNEST(t_emb) AS te))
#               )
#             ) AS cosine_sim
#           FROM src
#         )
#         SELECT news_id, ticker, cosine_sim
#         FROM scored
#         WHERE cosine_sim IS NOT NULL AND cosine_sim >= {self.COSINE_MIN};
#         """).result()
#         print("✅ news_ticker_similarity rebuilt")

#     # -------------------------
#     # Keywords (parameterized)
#     # -------------------------
#     def set_keywords(self, ticker_keywords: Dict[str, Iterable[str]]) -> None:
#         self.TICKER_KEYWORDS = {t.upper(): [str(k).lower() for k in ks] for t, ks in ticker_keywords.items()}

#     def keyword_fallback(self) -> None:
#         if not self.TICKER_KEYWORDS:
#             print("⚠️ No TICKER_KEYWORDS provided; skipping keyword fallback.")
#             return

#         rows = [{"ticker": t, "keyword": k} for t, ks in self.TICKER_KEYWORDS.items() for k in ks]
#         to_gbq(pd.DataFrame(rows), f"{self.DATASET_ID}.ticker_keywords",
#                project_id=self.PROJECT_ID, if_exists="replace")

#         self.client.query(fr"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_keyword` AS
#         WITH n AS (
#           SELECT id, LOWER(CONCAT(' ', title, ' ', content, ' ')) AS blob
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.news_raw`
#           WHERE date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {self.NEWS_LOOKBACK_DAYS} DAY)
#         ),
#         k AS (SELECT ticker, LOWER(keyword) AS keyword FROM `{self.PROJECT_ID}.{self.DATASET_ID}.ticker_keywords`)
#         SELECT
#           n.id AS news_id, k.ticker, {self.COSINE_MIN} AS cosine_sim
#         FROM n JOIN k
#         ON REGEXP_CONTAINS(
#              n.blob,
#              r'(?:^|\W)' ||
#              REGEXP_REPLACE(k.keyword, r'([\.^$|?*+()\[\]{{}}])', r'\\\1') ||
#              r'(?:\W|$)'
#            );
#         """).result()
#         print("✅ news_ticker_keyword rebuilt")

#     def link_news_events(self) -> None:
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_links` AS
#         SELECT news_id, ticker, MAX(cosine_sim) AS best_sim
#         FROM (
#           SELECT * FROM `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_similarity`
#           UNION ALL
#           SELECT * FROM `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_keyword`
#         )
#         GROUP BY news_id, ticker;
        
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.event_news_links` AS
#         SELECT
#           m.ticker, m.date, m.z_day, m.z_ar,
#           nr.id AS news_id, nr.title, nr.content, nr.source, nr.url, nr.date AS news_time
#         FROM `{self.PROJECT_ID}.{self.DATASET_ID}.abnormal_returns` m
#         JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_links` l USING (ticker)
#         JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.news_raw` nr ON nr.id = l.news_id
#         WHERE (ABS(m.z_day) >= 1.96 OR ABS(m.z_ar) >= 1.96)
#           AND DATE(nr.date) BETWEEN DATE_SUB(m.date, INTERVAL {self.WINDOW_DAYS} DAY)
#                                 AND DATE_ADD(m.date,  INTERVAL {self.WINDOW_DAYS} DAY);
#         """).result()
#         print("✅ event_news_links rebuilt")

#     # -------------------------
#     # AI summaries, labels, backtest
#     # -------------------------
#     def build_event_ai_summary(self) -> None:
#         assert self.GEN_MODEL, "Call setup_models() first."
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_summary` AS
#         WITH grp AS (
#           SELECT
#             ticker,
#             date,
#             ANY_VALUE(z_day) AS z_day,
#             ANY_VALUE(z_ar)  AS z_ar,
#             COUNT(*) AS news_count,
#             STRING_AGG(
#               CONCAT('[', source, '] ', title, ' — ', SUBSTR(content, 1, 160)),
#               ' || '
#               ORDER BY news_time DESC
#               LIMIT 20
#             ) AS headlines_blob
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_news_links`
#           GROUP BY ticker, date
#         )
#         SELECT
#           g.ticker,
#           g.date,
#           g.z_day,
#           g.z_ar,
#           g.news_count,
#           (
#             SELECT ml_generate_text_llm_result
#             FROM ML.GENERATE_TEXT(
#               MODEL `{self.GEN_MODEL}`,
#               (
#                 SELECT CONCAT(
#                   'Explain in plain investor language the likely drivers behind ', g.ticker,
#                   ' on ', CAST(g.date AS STRING),
#                   '. Use light terms like beat/miss and explain briefly. ',
#                   'Keep to 2–4 sentences. Headlines: ',
#                   g.headlines_blob
#                 ) AS prompt
#               ),
#               STRUCT(
#                 256 AS max_output_tokens,
#                 0.2 AS temperature,
#                 TRUE AS flatten_json_output
#               )
#             )
#           ) AS summary
#         FROM grp AS g;
#         """).result()

#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_tone` AS
#         WITH grp AS (
#           SELECT
#             ticker,
#             date,
#             STRING_AGG(
#               CONCAT('[', source, '] ', title, ' — ', SUBSTR(content, 1, 160)),
#               ' || '
#               ORDER BY news_time DESC
#               LIMIT 20
#             ) AS headlines_blob
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_news_links`
#           GROUP BY ticker, date
#         )
#         SELECT
#           g.ticker,
#           g.date,
#           CASE
#             WHEN LOWER(REGEXP_REPLACE((
#               SELECT ml_generate_text_llm_result
#               FROM ML.GENERATE_TEXT(
#                 MODEL `{self.GEN_MODEL}`,
#                 (
#                   SELECT CONCAT(
#                     'Are the headlines net positive for equity holders? Answer True or False only. ',
#                     g.headlines_blob
#                   ) AS prompt
#                 ),
#                 STRUCT(
#                   16 AS max_output_tokens,
#                   0.0 AS temperature,
#                   TRUE AS flatten_json_output
#                 )
#               )
#             ), r'[^a-z]', '')) = 'true'
#             THEN TRUE
#             ELSE FALSE
#           END AS is_positive
#         FROM grp AS g;
#         """).result()

#         print("✅ event_ai_summary + event_ai_tone rebuilt")

#     def build_event_labels(self) -> None:
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.event_label` AS
#         WITH base AS (
#           SELECT
#             a.ticker, a.date, a.z_day, a.z_ar,
#             COALESCE(s.news_count,0) AS news_count,
#             CASE
#               WHEN t.is_positive IS TRUE  THEN  1
#               WHEN t.is_positive IS FALSE THEN -1
#               ELSE 0
#             END AS ai_tilt
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.abnormal_returns` a
#           LEFT JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_summary` s USING (ticker, date)
#           LEFT JOIN `{self.PROJECT_ID}.{self.PROJECT_ID}.event_ai_tone`   t USING (ticker, date)
#           WHERE (ABS(a.z_day) >= 1.96 OR ABS(a.z_ar) >= 1.96)
#         ),
#         score AS (
#           SELECT
#             *,
#             LEAST(GREATEST(ABS(COALESCE(z_day,0)), ABS(COALESCE(z_ar,0))), 4)/4 * 0.4
#             + LEAST(news_count, 5)/5 * 0.4
#             + ((ai_tilt + 1)/2) * 0.2 AS signal_score
#           FROM base
#         )
#         SELECT
#           *,
#           CASE
#             WHEN signal_score >= 0.65 THEN 'Signal'
#             WHEN signal_score >= 0.40 THEN 'Weak'
#             ELSE 'Noise'
#           END AS label,
#           CASE
#             WHEN signal_score >= 0.65 THEN 0.75
#             WHEN signal_score >= 0.40 THEN 0.55
#             ELSE 0.35
#           END AS confidence
#         FROM score;
#         """).result()
#         print("✅ event_label rebuilt")

#     def build_event_ai_summary_filled(self) -> None:
#         assert self.GEN_MODEL, "Call setup_models() first."
#         self.client.query(f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_summary_filled` AS
#         WITH have AS (
#           SELECT ticker, date, z_day, z_ar, news_count, summary
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_summary`
#         ),
#         need AS (
#           SELECT e.ticker, e.date, a.z_day, a.z_ar
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_label` e
#           JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.abnormal_returns` a USING(ticker, date)
#           LEFT JOIN have h USING(ticker, date)
#           WHERE h.ticker IS NULL
#         )
#         SELECT * FROM have
#         UNION ALL
#         SELECT
#           n.ticker,
#           n.date,
#           n.z_day,
#           n.z_ar,
#           0 AS news_count,
#           (
#             SELECT ml_generate_text_llm_result
#             FROM ML.GENERATE_TEXT(
#               MODEL `{self.GEN_MODEL}`,
#               (SELECT CONCAT(
#                 'Explain in plain investor language a statistically significant move when no clear headlines are present. ',
#                 'Use the data provided and avoid speculation. ',
#                 'Say briefly whether it looks like technical flow vs. idiosyncratic news. ',
#                 'z_day=', FORMAT('%+.2f', n.z_day), ', z_ar=', FORMAT('%+.2f', n.z_ar), '.'
#               ) AS prompt),
#               STRUCT(192 AS max_output_tokens, 0.2 AS temperature, TRUE AS flatten_json_output)
#             )
#           ) AS summary
#         FROM need n;
#         """).result()
#         print("✅ event_ai_summary_filled ready")

#     def build_backtest(self) -> None:
#         sql_bt = f"""
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.forward_5d` AS
#         WITH r AS (
#           SELECT
#             ticker, date, close,
#             LEAD(close, 5) OVER (PARTITION BY ticker ORDER BY date) AS close_5d
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.prices`
#         )
#         SELECT
#             ticker, date,
#             SAFE_DIVIDE(close_5d - close, close) AS fwd_5d
#         FROM r;
        
#         CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.backtest_5d` AS
#         WITH j AS (
#           SELECT e.ticker, e.date, e.label, e.confidence, e.z_day, e.z_ar, f.fwd_5d
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_label` e
#           JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.forward_5d` f USING (ticker, date)
#           WHERE f.fwd_5d IS NOT NULL
#         )
#         SELECT
#           label,
#           CASE WHEN COALESCE(ABS(z_day),0) >= ABS(COALESCE(z_ar,0)) THEN SIGN(z_day) ELSE SIGN(z_ar) END AS shock_dir,
#           COUNT(*) AS n_events,
#           AVG(fwd_5d) AS avg_fwd5d,
#           AVG(CASE WHEN fwd_5d > 0 THEN 1 ELSE 0 END) AS hit_rate
#         FROM j
#         GROUP BY label, shock_dir
#         ORDER BY label, shock_dir;
#         """
#         self.client.query(sql_bt).result()
#         print("✅ backtest_5d created")

#     # -------------------------
#     # Readouts
#     # -------------------------
#     def verdict_panel(self, ticker: str, date: str | dt.date):
#         as_of = dt.date.fromisoformat(date) if isinstance(date, str) else date
#         q = f"""
#         WITH d AS (
#           SELECT a.ticker, a.date, a.r AS day_ret, a.z_day, a.z_ar
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.abnormal_returns` a
#           WHERE a.ticker = @t AND a.date = @d
#         ),
#         tr AS (
#           SELECT t_stat FROM `{self.PROJECT_ID}.{self.DATASET_ID}.trend_10d`
#           WHERE ticker = @t AND date = @d
#         ),
#         lab AS (
#           SELECT label, confidence, signal_score
#           FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_label`
#           WHERE ticker = @t AND date = @d
#         ),
#         sum AS (
#           SELECT summary FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_summary_filled`
#           WHERE ticker = @t AND date = @d
#         )
#         SELECT d.*, tr.t_stat, lab.label, lab.confidence, lab.signal_score, sum.summary
#         FROM d
#         LEFT JOIN tr ON TRUE
#         LEFT JOIN lab ON TRUE
#         LEFT JOIN sum ON TRUE;
#         """
#         job = self.client.query(q, job_config=bigquery.QueryJobConfig(
#             query_parameters=[
#                 bigquery.ScalarQueryParameter("t","STRING",ticker),
#                 bigquery.ScalarQueryParameter("d","DATE",as_of),
#             ]
#         ))
#         df = job.result().to_dataframe()
#         if df.empty:
#             print(f"No data for {ticker} on {as_of}")
#         else:
#             if _HAS_DISPLAY:
#                 display(df)
#             else:
#                 print(df.to_string(index=False))

#     def recent_signals_and_backtest(self, limit: int = 200) -> None:
#         signals = self.client.query(f"""
#         SELECT s.*, e.summary
#         FROM `{self.PROJECT_ID}.{self.DATASET_ID}.event_label` s
#         LEFT JOIN `{self.PROJECT_ID}.{self.DATASET_ID}.event_ai_summary` e USING (ticker, date)
#         ORDER BY date DESC, signal_score DESC
#         LIMIT {int(limit)}
#         """).result().to_dataframe()

#         bt = self.client.query(f"""
#         SELECT * FROM `{self.PROJECT_ID}.{self.DATASET_ID}.backtest_5d` ORDER BY label, shock_dir
#         """).result().to_dataframe()

#         print("=== Recent Signals ===")
#         for _, r in signals.iterrows():
#             zd, za = r.get("z_day") or 0, r.get("z_ar") or 0
#             direction = "UpShock" if (abs(zd) >= abs(za) and zd >= 0) or (abs(za) > abs(zd) and za >= 0) else "DownShock"
#             conf = int((r.get("confidence") or 0)*100)
#             z_max = max(abs(zd), abs(za))
#             news = int(r.get("news_count") or 0)
#             print(f"{r['date']}  {r['ticker']}  {direction}  z≈{z_max:.2f}σ  |  {r['label']}  (conf {conf}%)  news={news}")
#             if isinstance(r.get("summary"), str) and r["summary"].strip():
#                 print("  ↳", r["summary"].strip()[:350])
#             print("-")

#         print("\n=== Backtest Summary (5d fwd) ===")
#         print(bt.to_string(index=False))

#     # -------------------------
#     # Orchestrator
#     # -------------------------
#     def run_all(
#         self,
#         tickers: Iterable[str],
#         ticker_keywords: Optional[Dict[str, Iterable[str]]] = None,
#         benchmark: str = "^GSPC",
#         start: str = "2018-01-01",
#         end: Optional[str] = None,
#     ) -> None:
#         """End-to-end run for any set of tickers + keywords."""
#         self.setup_models()
#         self.load_prices_many(tickers=tickers, benchmark=benchmark, start=start, end=end)
#         self.build_daily_returns()
#         self.build_linear_model()
#         self.build_abnormal_returns()
#         self.build_trend_10d()
#         self.load_news_raw()
#         self.build_news_embeddings()
#         self.build_ticker_embeddings()
#         self.rebuild_similarity()
#         if ticker_keywords:
#             self.set_keywords(ticker_keywords)
#             self.keyword_fallback()
#         else:
#             self.client.query(f"""
#             CREATE OR REPLACE TABLE `{self.PROJECT_ID}.{self.DATASET_ID}.news_ticker_keyword` AS
#             SELECT CAST(NULL AS STRING) AS news_id, CAST(NULL AS STRING) AS ticker, CAST(NULL AS FLOAT64) AS cosine_sim
#             WHERE FALSE;
#             """).result()
#         self.link_news_events()
#         self.build_event_ai_summary()
#         self.build_event_labels()
#         self.build_event_ai_summary_filled()
#         self.build_backtest()


# tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
# keywords = {
#     "AAPL": ["apple", "iphone", "ipad", "macbook", "tim cook"],
#     "MSFT": ["microsoft", "windows", "azure", "copilot", "satya nadella"],
#     "NVDA": ["nvidia", "gpu", "geforce", "cuda", "h100", "blackwell", "jensen huang"],
#     "GOOGL": ["google", "alphabet", "youtube", "sundar pichai"],
#     "AMZN":  ["amazon", "aws", "prime", "jeff bezos"],
# }

# try:
#     with st.spinner("Running live pipeline in BigQuery…"):
#         pipeline = MarketSignalPipeline(
#             news_lookback_days=120,
#             cosine_min=0.60,
#             window_days=2
#         )
#         pipeline.run_all(
#             tickers=tickers,
#             ticker_keywords=keywords,
#             benchmark="^GSPC",
#             start="2018-01-01"
#         )

#     # Pull one fresh event row to render
#     client = pipeline.client
#     row_df = client.query(f"""
#       SELECT e.ticker, e.date, a.r AS day_ret, e.z_day, e.z_ar, t.t_stat,
#              e.label, e.confidence, e.signal_score, s.summary
#       FROM {pipeline.BQDFT}.event_label e
#       JOIN {pipeline.BQDFT}.abnormal_returns a USING (ticker, date)
#       LEFT JOIN {pipeline.BQDFT}.trend_10d t USING (ticker, date)
#       LEFT JOIN {pipeline.BQDFT}.event_ai_summary_filled s USING (ticker, date)
#       ORDER BY e.date DESC, e.signal_score DESC
#       LIMIT 1
#     """).result().to_dataframe()
# except Exception as ex:
#     st.error(f"Live run failed: {ex}")



# ============================================================
# STATIC SNAPSHOT (Obtained from Kaggle notebook run, no credits)
# ============================================================
row = {
    "ticker": "AAPL",
    "date": "2025-08-08",
    "day_ret": 0.042358,
    "z_day": 1.691366,
    "z_ar": 2.351215,
    "t_stat": 1.378697,
    "label": "Noise",
    "confidence": 0.35,
    "signal_score": 0.335121,
    "summary": (
        "Okay, let’s stick to the data: AAPL rose ~4.2%, with a market-adjusted abnormal return ≈ 2.35σ, "
        "which is statistically notable. With limited headline evidence around the event window, the move looks "
        "more like technical/flow-driven strength than a clear idiosyncratic catalyst. The recent 10-day trend "
        "is only mildly positive (t≈1.38), so we treat this as noise rather than a firm signal."
    ),
}
cols = ["ticker","date","day_ret","z_day","z_ar","t_stat","label","confidence","signal_score","summary"]
row_df = pd.DataFrame([row], columns=cols)
row_df["date"] = pd.to_datetime(row_df["date"])

# =========================
# Simple Controls (static only)
# =========================
co1, co2 = st.columns(2)
with co1:
    st.success("Ticker Context Keywords: apple, iphone, ipad, macbook, tim cook")
#     # st.selectbox("Ticker", ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"], index=0)
#     # st.success("Ticker:  AAPL")
with co2:
    with st.expander("How to read this dashboard", expanded=False):
        st.markdown("""
            ### Inputs
            - **Ticker** — The stock you’re analyzing.  
            - **Keywords** — Optional terms (e.g., product/CEO) used to find relevant headlines near the event date.  
            - **Date** — Event day for the snapshot.
                            
            ### What the units mean
            - **Day Return** — **percent (%)** move for the day.
            - **z_day** — **standard deviations (σ)** from the stock’s recent normal (~last 20 trading days).  
            *Plain English:* “How unusual was today for **this** stock?”
            - **z_ar (abnormal)** — **standard deviations (σ)** after removing the market effect (~120-day market model).  
            *Plain English:* “How company-specific was the move?”
            - **10-day trend (t-stat)** — **unitless score**; bigger magnitude = stronger short-term trend.
            - **Signal Score** — **0–1 scale** (shown as %); higher = more conviction that today mattered.
            - **Direction** — **UpShock / DownShock** from the sign of the bigger of z_day or z_ar.
            - **Label (confidence)** — quick bucket (Signal / Weak / Noise) with a confidence nudge.
    
            ### Rule of thumb
            - **Unusual size:**  
            • |z| ≈ **2σ** → worth a look.  
            • |z| ≈ **3σ+** → rare; dig deeper.
            - **Company vs market:**  
            • **High |z_ar|** → likely a company story (earnings, product, guidance).  
            • **High |z_day|, low |z_ar|** → may be market/sector flow.
            - **Trend context:**  
            • **Trend > +2** and **UpShock** → momentum supports the pop.  
            • **Trend < −2** and **DownShock** → momentum supports the drop.  
            • Shock **against** trend → treat as a potential one-off; wait for follow-through.
            - **Signal Score guide:**  
            • **≥ 0.65** → strong signal: read the headlines; consider if it fits your plan.  
            • **0.40–0.64** → mixed: note it, look for confirmation (next day/volume/news).  
            • **< 0.40** → likely routine noise for now.        
    
            ### Signal score formula
        """)
    
        st.latex(r"""
        \mathbf{signal\_score}
        = 0.4 \cdot \frac{\min(\max(|z_{day}|,|z_{ar}|),\,4)}{4}
        + 0.4 \cdot \frac{\min(\text{news\_count},\,5)}{5}
        + 0.2 \cdot \frac{\text{ai\_tilt}+1}{2}
        """)
        
        st.markdown("""
         **Practical read:** 
        - High |z_ar| with supportive headlines → likely **idiosyncratic** catalyst.  
        - High |z_day| but low |z_ar| → could be **market/sector** move.  
        - **High score, UpShock, rising trend** → cleaner bullish setup.  
        - **High score, DownShock, falling trend** → cleaner bearish setup.
        - Trend t-stat reinforces or fights the shock: alignment adds conviction, divergence suggests fade risk.  
        - Keywords help surface the right headlines; adjust them if the context looks off.
        """)
    # st.text_input("Keywords (comma-separated)", value="apple, iphone, ipad, macbook, tim cook")



# =========================
# Shared UI (renders static snapshot)
# =========================
row = row_df.iloc[0]

with st.container(border=True):
    st.subheader("Summary")
    st.write(row.get("summary", "—"))

    a, b, c = st.columns(3)
    a.metric("Ticker", str(row.get("ticker", "—")), border=True,height=150)
    dval = row.get("date", "—")
    if isinstance(dval, (pd.Timestamp, date)):
        dval = pd.to_datetime(dval).date().isoformat()
    b.metric("Date", str(dval), border=True,height=150)
    c.metric("Label", str(row.get("label", "—")), _conf_delta(row.get("confidence")), border=True,height=150)

    a, b, c = st.columns(3)
    a.metric("Day Return", _pct(row.get("day_ret")), border=True,height=150)
    b.metric("z_day", _sigma(row.get("z_day")), border=True,height=150)
    c.metric("z_ar (abnormal)", _sigma(row.get("z_ar")), border=True,height=150)

    a, b, c = st.columns(3)
    a.metric("10-day t-stat", _num(row.get("t_stat")), border=True,height=150)
    b.metric("Signal Score", _pct(row.get("signal_score"), 1), border=True,height=150)

    zd = row.get("z_day", 0) or 0.0
    za = row.get("z_ar", 0) or 0.0
    dominant_is_day = abs(zd) >= abs(za)
    up = (zd >= 0) if dominant_is_day else (za >= 0)
    c.metric("Direction", "UpShock" if up else "DownShock", border=True,height=150)

    

