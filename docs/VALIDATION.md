# Проверка реализации — 6 сентября 2026

Статус: локальный рабочий прототип, реальные результаты CPU-бенчмарка, подготовленный статический экспорт. На gero.uz ничего не опубликовано.

## Верификация

- 57 unit/property/integration tests: passed. Включают оба реальных бэкенда, invariants, Hypothesis, NaN/Inf, shape/dtype, integer precision, ULP, recursive exclusion, dedup, baseline и replay.
- 123 ONNX-графа, 2 режима оптимизации, 246 сравнений. 232 pass, 14 divergence, 0 execution errors, 0 unsupported. 14 observations относятся к 7 разным графам; два режима не считаются двумя самостоятельными багами.
- Второй полный запуск: все 246 observation IDs совпали; baseline states — unchanged. Агрегация двух файлов оставила 246 уникальных наблюдений.
- Проверены существование всех экспортированных артефактов и SHA-256 каждого reproducer ZIP.
- ZIP для LayerNormalization распакован и исполнен из вложенных исходников. Восстановлены tolerance и optimization; получен тот же observation_id.
- Node syntax check: app.js passed.
- Browser: реальный JSON загружен; комбинированные operation/dtype/optimization/status filters, empty state, reset, пагинация и карточка работают.
- Сегмент Softmax в Benchmarks даёт ровно 38 записей Softmax, без LogSoftmax и MaskedSoftmax.
- Скачивание reproducer ZIP из карточки: browser download event получен. Ошибок в консоли браузера нет.
- Desktop 1280×720 и mobile 390×844 / 320×740: визуально проверены. После исправления containing block у прокручиваемой таблицы document.scrollWidth совпадает с viewport 390/320; горизонтальная прокрутка остаётся внутри таблицы.
- GitHub-hosted CI и Docker build не запускались. Подготовленные конфигурации доступны в проекте.

## Окружение

Python 3.12.13; ONNX 1.19.0; ORT 1.22.1; NumPy 2.2.6; macOS-15.5-arm64-arm-64bit; CPUExecutionProvider; один поток; sequential; deterministic compute. CPU model недоступна из sandbox, это сохранено в environment.

Run ID: `ecf59104f767379863b0c7913f410831f5fbde7515a943bb22588fcb9332eee3`.

Source digest: `79a49f45e5b7f5c2e1a21debe1ef3d85383fcb8e6152089619073e800cac7c1f`.

## Наблюдения для дальнейшего исследования

Данные ниже относятся к режиму disabled; те же графы имеют отдельные наблюдения для all. Колонки конечности — число finite элементов / размер выхода. Abs error считается только для finite pairs, None означает отсутствие такого измерения.

| Operation | Вариант | Dtype | ORT finite | Reference finite | Max abs finite |
|---|---|---|---:|---:|---:|
| CosineSimilarity | large_offset | float16 | 4/4 | 0/4 | None |
| CosineSimilarity | tiny | float16 | 4/4 | 4/4 | 1.9234619140625 |
| LayerNormalization | large_offset | float32 | 51/68 | 68/68 | 2.307088240981102 |
| LogSoftmax | random-001 | float32 | 420/420 | 279/420 | 0.413299560546875 |
| LogSoftmax | random-022 | float32 | 508/508 | 50/508 | 0.2647857666015625 |
| LpNormalization | large_offset | float16 | 68/68 | 68/68 | 0.2431640625 |
| LpNormalization | tiny | float16 | 68/68 | 51/68 | 0.23779296875 |

В части случаев ReferenceEvaluator выдаёт неконечные значения при конечном результате ORT. Для LayerNormalization с большим смещением 17 из 68 значений ORT неконечны, reference конечен, но сам не проходит заданную проверку относительно float64-формулы. Это иллюстрирует необходимость проверять обе реализации. Причина ошибки и соответствие разрешённому поведению конкретной версии требуют отдельного анализа.

Падение batch-check при неконечном выходе означает неудовлетворённую предпосылку finite-domain contract; само по себе оно не доказывает зависимость результата от батча. Новизна относительно upstream issues не исследована. Регрессии между разными версиями ORT этим запуском не установлены.
