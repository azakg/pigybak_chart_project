# -*- coding: utf-8 -*-

import os
import io
import re
import ssl
import time
import math
from typing import Optional, Tuple, Dict, List

import certifi
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========= CONFIG (НЕ ХРАНИМ ПРИВАТНЫЕ ССЫЛКИ В КОДЕ) =========
# Укажи путь/URL к данным через ENV:
#   export INTERNS_SOURCE="path_or_url_to_csv_or_xlsx"
# Опционально: export INTERNS_SHEET="Sheet1"
SOURCE = os.getenv("INTERNS_SOURCE", "")
SHEET: Optional[str] = os.getenv("INTERNS_SHEET") or None

# Имена колонок во входной таблице (PII НЕ ВЫВОДИМ, НО МОЖЕМ ИСПОЛЬЗОВАТЬ ДЛЯ ФИЛЬТРОВ)
FIRST_NAME_COL   = "First Name"
LAST_NAME_COL    = "Last Name"
UNIVERSITY_COL   = "University / School"
LOCATION_COL     = "Location"
ACCEPT_DATE_COL  = "Accept Date"

# Выходные файлы (артефакты — добавь в .gitignore)
OUTPUT_HTML = "interns_map.html"
OUTPUT_CSV  = "interns_by_state.csv"
GEO_CACHE   = "geo_cache.csv"        # автокэш геокодинга (без PII)
OVERRIDES_CSV = "overrides.csv"      # ручные соответствия: University / School,lat,lon,state[,country]

# Визуал
BUBBLE_HEX       = "#7FD858"  # пузыри штатов
UNI_MARKER_COLOR = "#2563eb"  # точки универов
BASE_RADIUS      = 10

# Порог k-анонимности для показа точек университетов
K_MIN_UNI = int(os.getenv("K_MIN_UNI", "3"))
# Порог (опционально) для пузырей штатов; 0 = без порога
K_MIN_STATE = int(os.getenv("K_MIN_STATE", "0"))

# SSL контекст
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ========= HTTP с ретраями (для загрузки CSV) =========
def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

# ========= чтение таблицы (без логирования PII) =========
def _read_table(src: str, sheet: Optional[str]) -> pd.DataFrame:
    if not src:
        raise SystemExit("Set INTERNS_SOURCE env var to your CSV/XLSX path or URL")
    sl = src.lower()
    if sl.endswith(".xlsx") or sl.endswith(".xls"):
        return pd.read_excel(src, sheet_name=sheet, engine="openpyxl")
    if sl.startswith(("http://", "https://")):
        sess = _requests_session()
        r = sess.get(src, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, verify=certifi.where())
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))
    return pd.read_csv(src)

# ========= справочники штатов =========
STATE_ABBR = {
    'ALABAMA':'AL','ALASKA':'AK','ARIZONA':'AZ','ARKANSAS':'AR','CALIFORNIA':'CA','COLORADO':'CO',
    'CONNECTICUT':'CT','DELAWARE':'DE','DISTRICT OF COLUMBIA':'DC','FLORIDA':'FL','GEORGIA':'GA',
    'HAWAII':'HI','IDAHO':'ID','ILLINOIS':'IL','INDIANA':'IN','IOWA':'IA','KANSAS':'KS','KENTUCKY':'KY',
    'LOUISIANA':'LA','MAINE':'ME','MARYLAND':'MD','MASSACHUSETTS':'MA','MICHIGAN':'MI','MINNESOTA':'MN',
    'MISSISSIPPI':'MS','MISSOURI':'MO','MONTANA':'MT','NEBRASKA':'NE','NEVADA':'NV','NEW HAMPSHIRE':'NH',
    'NEW JERSEY':'NJ','NEW MEXICO':'NM','NEW YORK':'NY','NORTH CAROLINA':'NC','NORTH DAKOTA':'ND',
    'OHIO':'OH','OKLAHOMA':'OK','OREGON':'OR','PENNSYLVANIA':'PA','RHODE ISLAND':'RI',
    'SOUTH CAROLINA':'SC','SOUTH DAKOTA':'SD','TENNESSEE':'TN','TEXAS':'TX','UTAH':'UT',
    'VERMONT':'VT','VIRGINIA':'VA','WASHINGTON':'WA','WEST VIRGINIA':'WV','WISCONSIN':'WI','WYOMING':'WY'
}
STATE_RE = re.compile(r"\b([A-Z]{2})\b")

def parse_state_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    u = text.upper()
    m = STATE_RE.search(u)
    if m and m.group(1) in STATE_ABBR.values():
        return m.group(1)
    tokens = re.split(r"[,/;-]\s*|\s+", u)
    for L in range(len(tokens)):
        for R in range(L+1, min(len(tokens), L+3)+1):
            name = " ".join(tokens[L:R])
            if name in STATE_ABBR:
                return STATE_ABBR[name]
    return None

# ========= кэш (без PII) =========
def load_cache(path: str) -> Dict[str, Tuple[float, float, str, str]]:
    """key -> (lat, lon, state_abbr, country); key = University / School"""
    try:
        df = pd.read_csv(path)
        return {
            str(row["key"]): (
                float(row["lat"]), float(row["lon"]),
                str(row.get("state_abbr", "")) or "",
                str(row.get("country", "")) or ""
            )
            for _, row in df.iterrows()
        }
    except Exception:
        return {}

def save_cache(path: str, cache: Dict[str, Tuple[float, float, str, str]]) -> None:
    rows = [{"key": k, "lat": v[0], "lon": v[1], "state_abbr": v[2], "country": v[3]} for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(path, index=False)

def load_overrides(path: str) -> Dict[str, Tuple[float, float, str, str]]:
    """Ручные переопределения: University / School,lat,lon,state[,country]"""
    try:
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            key = str(r.get("University / School", "")).strip()
            if not key:
                continue
            lat = float(r.get("lat"))
            lon = float(r.get("lon"))
            st  = str(r.get("state", "")).strip().upper()
            country = str(r.get("country", "")).strip()
            if st in STATE_ABBR.values() and not country:
                country = "United States"
            out[key] = (lat, lon, st, country)
        return out
    except Exception:
        return {}

# ========= геокодинг с каскадами =========
def geocode_university(univ: str, geocode_fn, cache: Dict[str, Tuple[float, float, str, str]],
                       overrides: Dict[str, Tuple[float, float, str, str]]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if not isinstance(univ, str) or not univ.strip():
        return None, None, None, None
    key = univ.strip()

    # 0) ручное переопределение
    if key in overrides:
        lat, lon, st, country = overrides[key]
        cache[key] = (lat, lon, st or "", country or "")
        save_cache(GEO_CACHE, cache)
        return lat, lon, (st or None), (country or None)

    # 1) кэш
    if key in cache:
        lat, lon, st, country = cache[key]
        return (None if pd.isna(lat) else lat,
                None if pd.isna(lon) else lon,
                st or None,
                country or None)

    # 2) каскад запросов
    queries = [
        f"{key}, United States",
        f"{key}",
        f"{key}, Canada",
    ]
    lat = lon = None
    st_abbr: Optional[str] = None
    country: Optional[str] = None

    for q in queries:
        loc = geocode_fn(q, timeout=10, addressdetails=True)
        if loc:
            lat, lon = float(loc.latitude), float(loc.longitude)
            addr = (getattr(loc, "raw", {}) or {}).get("address", {})
            country = addr.get("country")
            state_name = addr.get("state") or addr.get("state_district") or ""
            st_abbr = parse_state_from_text(state_name)
            break

    cache[key] = (
        lat if lat is not None else float("nan"),
        lon if lon is not None else float("nan"),
        st_abbr or "",
        country or ""
    )
    save_cache(GEO_CACHE, cache)
    time.sleep(1.1)  # вежливый лимит к Nominatim
    return lat, lon, st_abbr, country

# ========= карта =========
def build_map(df_state: pd.DataFrame, df_unis: pd.DataFrame) -> folium.Map:
    fmap = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron")

    # Слой пузырей штатов (агрегаты; PII нет)
    layer_states = folium.FeatureGroup(name="State Bubbles", show=True); fmap.add_child(layer_states)
    df_state_viz = df_state.copy()
    if K_MIN_STATE > 0:
        df_state_viz = df_state_viz[df_state_viz["count"] >= K_MIN_STATE]
    max_count = int(df_state_viz["count"].max()) if not df_state_viz.empty else 1

    def radius(count: int) -> float:
        return BASE_RADIUS + 20 * math.sqrt(count / max_count)

    for _, r in df_state_viz.iterrows():
        folium.Circle(
            radius=radius(int(r["count"])) * 7000,
            location=[float(r["lat"]), float(r["lon"])],
            tooltip=f"{r['state']}: {int(r['count'])} intern(s)",
            color=BUBBLE_HEX,
            fill=True, fill_opacity=0.35, fill_color=BUBBLE_HEX, weight=2,
        ).add_to(layer_states)

    # Слой университетов (k-анонимность, без имён)
    layer_unis = folium.FeatureGroup(name="Universities (k>=%d)" % K_MIN_UNI, show=True); fmap.add_child(layer_unis)
    cluster = MarkerCluster().add_to(layer_unis)

    for _, r in df_unis.iterrows():
        cnt = int(r["count"])
        if cnt < K_MIN_UNI:
            continue  # скрываем редкие значения
        popup = folium.Popup(
            html=f"<b>{r['university']}</b><br>{cnt} intern(s)",
            max_width=360
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6 if cnt == 1 else min(12, 6 + cnt),
            weight=1, color="#1f2937",
            fill=True, fill_color=UNI_MARKER_COLOR, fill_opacity=0.9,
            tooltip=f"{r['university']}: {cnt} intern(s)",
            popup=popup,
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap

# ========= main =========
def main():
    print("Загрузка таблицы…")
    df = _read_table(SOURCE, SHEET)

    needed = [UNIVERSITY_COL, LOCATION_COL, ACCEPT_DATE_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Нет колонок: {missing}. Ожидаются: {needed}")

    # Фильтр: принят и не 'declin'
    mask_nonempty = df[ACCEPT_DATE_COL].astype(str).str.strip() != ""
    mask_not_declined = ~df[ACCEPT_DATE_COL].astype(str).str.lower().str.contains("declin")
    df = df[mask_nonempty & mask_not_declined].copy()
    if df.empty:
        raise SystemExit("Нет принятых после фильтра Accept Date.")

    # Геокодер (правильный user_agent с контактом)
    geolocator = Nominatim(
        user_agent=os.getenv("NOMINATIM_UA", "interns-bubble-map (contact: youremail@example.com)"),
        ssl_context=SSL_CTX,
        scheme="https",
        domain="nominatim.openstreetmap.org"
    )

    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1.0,
        max_retries=3,
        swallow_exceptions=True
    )

    cache = load_cache(GEO_CACHE)
    overrides = load_overrides(OVERRIDES_CSV)

    lat_list: List[Optional[float]] = []
    lon_list: List[Optional[float]] = []
    st_list: List[Optional[str]] = []
    country_list: List[Optional[str]] = []

    print("Геокодирование университетов… (кэш ускорит повторы)")
    for univ, loc_fallback in zip(df[UNIVERSITY_COL], df[LOCATION_COL]):
        lat, lon, st, country = geocode_university(str(univ), geocode, cache, overrides)
        if not st:
            st = parse_state_from_text(str(loc_fallback))
        lat_list.append(lat)
        lon_list.append(lon)
        st_list.append(st)
        country_list.append(country)

    df["lat"] = lat_list
    df["lon"] = lon_list
    df["state"] = st_list
    df["country"] = country_list

    # Оставляем точки с координатами (для слоя универов)
    df_pts = df[df["lat"].notna() & df["lon"].notna()].copy()
    if df_pts.empty:
        raise SystemExit("Нет координат для точек университетов после геокодинга.")

    # Агрегат по штату ТОЛЬКО для США (агрегаты не содержат PII)
    df_us = df_pts[(df_pts["country"].fillna("").str.contains("United States")) &
                   (df_pts["state"].notna()) &
                   (df_pts["state"].isin(STATE_ABBR.values()))].copy()

    grouped_state = (
        df_us.groupby("state")
             .agg(count=("state", "size"), lat=("lat", "mean"), lon=("lon", "mean"))
             .reset_index()
             .sort_values("count", ascending=False)
    )
    # (опционально) можно отсечь маленькие значения при сохранении
    grouped_state.to_csv(OUTPUT_CSV, index=False)

    # Агрегат по университетам (все страны), БЕЗ имён
    grouped_uni = (
        df_pts.groupby([UNIVERSITY_COL])
              .agg(
                  state=("state", "first"),
                  country=("country", "first"),
                  lat=("lat", "mean"),
                  lon=("lon", "mean"),
                  count=("lat", "size"),
              )
              .reset_index()
              .rename(columns={UNIVERSITY_COL: "university"})
    )

    print(f"Сохраняю свод по штатам → {OUTPUT_CSV}")
    print(f"Строю карту → {OUTPUT_HTML}")
    fmap = build_map(grouped_state, grouped_uni)
    fmap.save(OUTPUT_HTML)
    print(f"Готово. Свод: {OUTPUT_CSV}, карта: {OUTPUT_HTML}, кэш: {GEO_CACHE}")

if __name__ == "__main__":
    main()
