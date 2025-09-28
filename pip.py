import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# читаем все файлы
drivers = pd.read_csv('drivers.csv')
vehicles = pd.read_csv('vehicles.csv')
insurance = pd.read_csv('insurance.csv')
driver_vehicle_intervals = pd.read_csv('driver_vehicle_intervals.csv', parse_dates=['start_time', 'end_time'])
telemetry = pd.read_csv('telemetry.csv', parse_dates=['timestamp'])
epd_events = pd.read_csv('epd_events.csv', parse_dates=['date'])

# 1. Сначала связываем телеметрию с интервалами
merged = telemetry.merge(
    driver_vehicle_intervals,
    on='vehicle_id',
    how='left'
)

# 2. Фильтруем только те строки, где timestamp попадает в интервал
merged = merged[
    (merged['timestamp'] >= merged['start_time']) &
    (merged['timestamp'] <= merged['end_time'])
]

# 3. Подтягиваем информацию о водителях
merged = merged.merge(
    drivers,
    on='driver_id',
    how='left'
)

# 4. Подтягиваем данные о ТС
merged = merged.merge(
    vehicles,
    on='vehicle_id',
    how='left'
)

# 5. Подтягиваем страховку (vehicle_id)
merged = merged.merge(
    insurance,
    on='vehicle_id',
    how='left'
)

# 6. Подтягиваем события ЭПД по epd_doc_id
merged = merged.merge(
    epd_events,
    left_on='epd_doc_id',  # из driver_vehicle_intervals
    right_on='doc_id',
    how='left'
)

merged.to_csv("merged_telemetry_intervals.csv", index=False)

"""

Pipeline для:
- агрегации телеметрии по парам (driver_id, vehicle_id),
- вычисления прокси risk_score (эвристика),
- кластеризации (низкий/средний/высокий риск),
- расчёта рекомендованной скидки и простых правил по франшизе,
- подбора совместимости driver <-> vehicle.

"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PATH = "merged_telemetry_intervals.csv"
OUT_CSV = "recommendations.csv"

def load_csv_try(path):
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None

def main(input_path=INPUT_PATH, out_csv=OUT_CSV):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    # стандартизируем имена колонок driver_id / vehicle_id (есть варианты merged с _x/_y)
    if "driver_id" not in df.columns:
        if "driver_id_x" in df.columns:
            df["driver_id"] = df["driver_id_x"]
        elif "driver_id_y" in df.columns:
            df["driver_id"] = df["driver_id_y"]
    if "vehicle_id" not in df.columns:
        if "vehicle_id_x" in df.columns:
            df["vehicle_id"] = df["vehicle_id_x"]
        elif "vehicle_id_y" in df.columns:
            df["vehicle_id"] = df["vehicle_id_y"]

    # Проверка
    if "driver_id" not in df.columns or "vehicle_id" not in df.columns:
        raise ValueError("Ожидаются колонки driver_id и vehicle_id в input CSV")

    # Преобразование дат/чисел
    for col in ["timestamp", "start_time", "end_time", "date", "hired_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["speed_kmh", "accel_m_s2", "odometer_km", "annual_premium_eur", "deductible_eur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Вычислим длительность поездки, если есть start_time/end_time
    if "start_time" in df.columns and "end_time" in df.columns:
        df["trip_seconds"] = (pd.to_datetime(df["end_time"], errors="coerce") - pd.to_datetime(df["start_time"], errors="coerce")).dt.total_seconds().fillna(0)
    else:
        df["trip_seconds"] = 0

    # Группировка по driver_id + vehicle_id
    group_cols = ["driver_id", "vehicle_id"]
    agg = df.groupby(group_cols).agg(
        trips=("trip_seconds", "count"),
        total_trip_seconds=("trip_seconds", "sum"),
        avg_speed_kmh=("speed_kmh", "mean"),
        max_speed_kmh=("speed_kmh", "max"),
        std_speed_kmh=("speed_kmh", "std"),
        avg_accel=("accel_m_s2", "mean"),
        max_accel=("accel_m_s2", "max"),
        std_accel=("accel_m_s2", "std")
    ).reset_index().fillna(0)

    # Считаем типы событий (hard_brake, hard_accel, lane_change, normal)
    def count_events(s):
        s = s.fillna("normal")
        c = s.value_counts()
        return pd.Series({
            "hard_brake_count": int(c.get("hard_brake", 0)),
            "hard_accel_count": int(c.get("hard_accel", 0)),
            "lane_change_count": int(c.get("lane_change", 0)),
            "normal_count": int(c.get("normal", 0))
        })

    if "event" in df.columns:
        events = df.groupby(group_cols)["event"].apply(count_events).reset_index()
    else:
        # если нет колонки event — ставим 0
        events = agg[["driver_id", "vehicle_id"]].copy()
        events["hard_brake_count"] = 0
        events["hard_accel_count"] = 0
        events["lane_change_count"] = 0
        events["normal_count"] = 0

    features = agg.merge(events, on=group_cols, how="left").fillna(0)

    # Попробуем получить odometer/vehicle_age/tenure из объединённого файла, если есть
    if "odometer_km" in df.columns:
        veh_od = df[["vehicle_id", "odometer_km"]].drop_duplicates(subset=["vehicle_id"])
        features = features.merge(veh_od, on="vehicle_id", how="left")
    else:
        features["odometer_km"] = 0

    if "year" in df.columns:
        veh_year = df[["vehicle_id", "year"]].drop_duplicates(subset=["vehicle_id"])
        features = features.merge(veh_year, on="vehicle_id", how="left")
        features["vehicle_age"] = pd.Timestamp.now().year - pd.to_numeric(features["year"], errors="coerce")
    else:
        features["vehicle_age"] = 0

    if "hired_date" in df.columns:
        drv = df[["driver_id", "hired_date"]].drop_duplicates(subset=["driver_id"])
        drv["tenure_days"] = (pd.Timestamp.now() - pd.to_datetime(drv["hired_date"], errors="coerce")).dt.days.fillna(0)
        features = features.merge(drv[["driver_id", "tenure_days"]], on="driver_id", how="left")
    else:
        features["tenure_days"] = 0

    # Заполняем NaN для числовых
    num_cols = features.select_dtypes(include=[np.number]).columns
    features[num_cols] = features[num_cols].fillna(0)

    # Соберём признаки для proxy-risk
    score_cols = [
        "hard_brake_count",
        "hard_accel_count",
        "lane_change_count",
        "avg_speed_kmh",
        "avg_accel",
        "vehicle_age",
        "odometer_km",
        "tenure_days"
    ]
    available = [c for c in score_cols if c in features.columns]

    # Стандартизация
    scaler = StandardScaler()
    X = scaler.fit_transform(features[available])

    # Веса: tenure_days уменьшает риск (отрицательный вес), остальные увеличивают
    weights = np.array([ -1.0 if c == "tenure_days" else 1.0 for c in available ], dtype=float)

    # Вычисляем сырое значение риска и нормируем в [0,1]
    risk_raw = (X * weights).sum(axis=1)
    risk_score = (risk_raw - risk_raw.min()) / (risk_raw.max() - risk_raw.min() + 1e-9)
    features["risk_score"] = risk_score

    # Кластеризация в 3 группы (низкий/средний/высокий)
    kmeans = KMeans(n_clusters=3, random_state=42)
    features["risk_cluster"] = kmeans.fit_predict(features[available])

    # Рекомендация скидки (эвристика)
    base_discount = 0.10        # 10%
    max_extra = 0.10            # до +10% для безопасных
    features["recommended_discount_pct"] = (base_discount + (1 - features["risk_score"]) * max_extra).clip(0, 0.30)

    # Правила по страховке (эвристика)
    def insurance_reco(r):
        if r < 0.33:
            return ("low", "increase_discount", -50)
        elif r > 0.66:
            return ("high", "increase_deductible", 100)
        else:
            return ("medium", "standard", 0)

    features[["risk_bucket", "suggested_action", "suggested_deductible_adj_eur"]] = features["risk_score"].apply(
        lambda x: pd.Series(insurance_reco(x))
    )

    # Совместимость driver <-> vehicle:
    drv_vec = features.groupby("driver_id")[available].mean().fillna(0)
    veh_vec = features.groupby("vehicle_id")[available].mean().fillna(0)

    # Нормализация перед косинусной схожестью
    if len(drv_vec) == 0 or len(veh_vec) == 0:
        # Защита на случай пустых данных
        features["best_vehicle_id"] = None
        features["compatibility"] = 0.0
    else:
        drv_norm = pd.DataFrame(StandardScaler().fit_transform(drv_vec), index=drv_vec.index, columns=drv_vec.columns)
        veh_norm = pd.DataFrame(StandardScaler().fit_transform(veh_vec), index=veh_vec.index, columns=veh_vec.columns)
        sim = cosine_similarity(drv_norm, veh_norm)
        sim_df = pd.DataFrame(sim, index=drv_norm.index, columns=veh_norm.index)

        # Для каждого водителя выбираем лучшую машину
        best = []
        for d in sim_df.index:
            best_vehicle = sim_df.loc[d].idxmax()
            best_score = float(sim_df.loc[d].max())
            best.append({"driver_id": d, "best_vehicle_id": best_vehicle, "compatibility": best_score})
        best_df = pd.DataFrame(best)
        features = features.merge(best_df, on="driver_id", how="left")

    # Подготовка и сохранение выходного файла
    out_cols = [
        "driver_id", "vehicle_id", "trips", "total_trip_seconds",
        "avg_speed_kmh", "max_speed_kmh", "avg_accel", "max_accel",
        "hard_brake_count", "hard_accel_count", "lane_change_count",
        "risk_score", "risk_cluster", "recommended_discount_pct",
        "risk_bucket", "suggested_action", "suggested_deductible_adj_eur",
        "best_vehicle_id", "compatibility"
    ]
    out_cols = [c for c in out_cols if c in features.columns]
    out_df = features[out_cols].copy()

    # Красивое представление скидки в процентах (опционально)
    if "recommended_discount_pct" in out_df.columns:
        out_df["recommended_discount_pct"] = (out_df["recommended_discount_pct"] * 100).round(2)

    out_df.to_csv(out_csv, index=False)
    print(f"Saved recommendations to: {out_csv}")
    # Вывод топ-10 самых безопасных (по risk_score)
    print("Top 10 (lowest risk_score):")
    print(out_df.drop_duplicates().sort_values("risk_score").to_string(index=False))

if __name__ == "__main__":
    main()
