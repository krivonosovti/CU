---
name: ml-homework-debugging-patterns
description: Паттерны отладки типичных ошибок в ML-домашках: corrupted matplotlib, SSL cert issues, KNNClassifier bugs, Nadaraya-Watson NaN
source: auto-skill
extracted_at: '2026-10-05T12:56:16.106Z'
---

## Типичные ошибки при реализации ML-алгоритмов

### 1. Corrupted matplotlib — circular import

**Симптом:** `ImportError: cannot import name '_api' from partially initialized module 'matplotlib'`

**Причина:** Пакет matplotlib установлен неполно — отсутствуют приватные модули (`_api.py`, `_docstring.py`). Часто бывает при повреждённой установке venv.

**Проверка:**
```bash
ls .venv/lib/python3.X/site-packages/matplotlib/ | grep -E "^_api|^_docstring"
```
Если ничего не найдено — установка повреждена.

**Решение:**
```bash
.venv/bin/pip install --force-reinstall --no-cache-dir matplotlib
```

### 2. SSL certificate error — homebrew Python

**Симптом:** `SSLCertVerificationError: certificate verify failed: Missing Authority Key Identifier` при скачивании данных (sklearn datasets и т.п.)

**Причина:** homebrew Python 3.14 + OpenSSL 3.6 не находят корневые сертификаты macOS.

**Решение:**
```bash
.venv/bin/pip install certifi
```
Затем использовать бандл:
```python
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
```

### 3. KNNClassifier — типичные баги

| Ошибка | Как исправить |
|--------|---------------|
| `X_train` вместо `X` в `predict` | Аргумент метода — `X`, переменная `X_train` не определена |
| `self.classes_[k_indices]` вместо `self.y_train[k_indices]` | `self.classes_` — набор уникальных меток, а не метки точек. Для соседей нужны `self.y_train[k_indices]` |
| `calculated_weights == self._calculate_weights(...)` | Двойной `==` вместо одинарного `=` (сравнение вместо присваивания) |
| `label_counts[most_frequent_labels]` | `most_frequent_labels` — список кортежей, нельзя как ключ. Максимальная частота: `most_frequent_labels[0][1]` |
| `np.zeros(n_test_points, n_classes)` | Второй аргумент — dtype, а не вторая размерность. Нужно: `np.zeros((n_test_points, n_classes))` |
| `len(X[0])` вместо `len(X)` | `X[0]` — первая строка (признаки), `len(X)` — количество строк |

### 4. Nadaraya-Watson — NaN при малом bandwidth

**Симптом:** При `h=0.1` с компактными ядрами (uniform, epanechnikov, triangular) все предсказания = `nan`, кривая на графике отсутствует.

**Причина:** Компактные ядра обнуляют вес при `|d| > 1`. При малом `h` нормализованное расстояние до ближайшей обучающей точки > 1 → все веса = 0 → формула `sum(w*y)/sum(w)` = `0/0` = `nan`.

**Проверка:**
```python
print(np.isnan(y_pred).sum(), "/", len(y_pred), "nan")
```

**Решение для анализа:** Отбрасывать `nan` перед расчётом метрик:
```python
mask = ~np.isnan(y_pred)
y_pred_valid = y_pred[mask]
y_test_valid = y_test[mask]
rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_valid))
```

### 5. Сравнение ядер — эмпирический подход

Для анализа поведения ядер используй таблицу метрик:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

kernels = ['gaussian', 'epanechnikov', 'triangular', 'uniform']
bandwidths = [0.1, 0.3, 1.0, 3.0]

rows = []
for kernel in kernels:
    for h in bandwidths:
        model = NadarayaWatsonRegressor(bandwidth=h, kernel=kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mask = ~np.isnan(y_pred)
        rows.append({
            'Ядро': kernel.capitalize(),
            'h': h,
            'RMSE': np.sqrt(mean_squared_error(y_pred[mask], y_test[mask])),
            'MAE': mean_absolute_error(y_pred[mask], y_test[mask]),
            'nan': (~mask).sum(),
        })

results_df = pd.DataFrame(rows)
```

**Ключевые наблюдения:**
- **Gaussian** — бесконечный носитель, всегда работает, но при большом `h` деградирует (притягивает далёкие точки)
- **Компактные** (uniform, epanechnikov, triangular) — при малом `h` дают `nan`, при большом `h` лучше gaussian
- **uniform** — ступенька, наименее гладкий
- **triangular** — линейный спад, хороший баланс
- **epanechnikov** — параболический спад, оптимальный по теории для MSE

### 6. collections.Counter для majority vote

```python
from collections import Counter, defaultdict

# uniform — считаем частоты
label_counts = Counter(k_labels)
most_frequent_labels = label_counts.most_common()  # [(label, freq), ...]
max_frequency = most_frequent_labels[0][1]
candidates = [l for l, f in most_frequent_labels if f == max_frequency]

# weighted — суммируем веса
label_weights_sum = defaultdict(float)
for label, weight in zip(k_labels, calculated_weights):
    label_weights_sum[label] += weight
label_counts = Counter(label_weights_sum)  # Counter({label: total_weight, ...})
```
