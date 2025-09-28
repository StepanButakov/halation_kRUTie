# Легенда по файлам тестового датасета

## drivers.csv
- driver_id — уникальный идентификатор водителя
- name — имя водителя
- license_number — номер водительского удостоверения
- license_category — категория лицензии (B, C, C+E и т.д.)
- hired_date — дата найма (YYYY-MM-DD)
- phone — телефон

## vehicles.csv
- vehicle_id — уникальный идентификатор ТС
- registration_plate — регистрационный номер
- make — марка (Volvo, Scania и т.п.)
- model — модель
- year — год выпуска
- vin — VIN
- odometer_km — пробег, км

## insurance.csv
- policy_id — ID полиса
- vehicle_id — ID ТС
- provider — страховая компания
- start_date — начало действия полиса (YYYY-MM-DD)
- end_date — окончание действия полиса (YYYY-MM-DD)
- tariff_class — тарифный класс (A/B/C)
- deductible_eur — франшиза, EUR
- annual_premium_eur — годовая премия, EUR

## epd_events.csv (упрощённый)
- doc_id — идентификатор документа
- date — дата события (YYYY-MM-DD)
- driver_id — идентификатор водителя
- vehicle_id — идентификатор ТС

## driver_vehicle_intervals.csv
- interval_id — уникальный id интервала
- driver_id — водитель
- vehicle_id — ТС
- epd_doc_id — связанный документ ЭПД
- start_time — начало интервала (YYYY-MM-DDTHH:MM:SS)
- end_time — конец интервала (YYYY-MM-DDTHH:MM:SS)
- route_start — условная точка старта маршрута
- route_end — условная точка конца маршрута

## telemetry.csv
- telemetry_id — id записи
- timestamp — время замера (YYYY-MM-DDTHH:MM:SS)
- vehicle_id — ТС
- speed_kmh — скорость, км/ч
- accel_m_s2 — ускорение, м/с²
- event — событие (normal, hard_brake, hard_accel, lane_change)
- lat — широта (decimal degrees)
- lon — долгота (decimal degrees)
- heading_deg — курс, градусы
