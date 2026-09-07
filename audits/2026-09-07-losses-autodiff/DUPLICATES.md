# Проверка публичных дубликатов

Дата основного запроса, текущие SHA, полный текст запросов и результаты:
`duplicate-search.json`. Дополнительные ответы: `logaddexp-search.json`,
`cancellation-search.json`, `nntrainer-bce-search.json`,
`nntrainer-precision-search.json`. Запросы без фильтра состояния находят и
закрытые issues/PR. Во всех сохранённых ответах `incomplete_results=false`;
лимит страницы 100 не достигнут.

| Ближайшая запись | Почему не точный дубликат |
|---|---|
| [MLX #1280](https://github.com/ml-explore/mlx/pull/1280) | Обработка вероятностей 0/1 при `with_logits=False`; новая находка касается конечных логитов и малых положительных потерь. |
| [MLX #122](https://github.com/ml-explore/mlx/pull/122), [#492](https://github.com/ml-explore/mlx/pull/492) | Добавление BCE и поддержки вероятностного входа; не отчёты о потере хвоста в logits-режиме. |
| [MLX #4227](https://github.com/ml-explore/mlx/pull/4227) | Расширение gradient reference tests в обычных диапазонах; не исправление асимметрии VJP/JVP при разности 20. |
| [MLX #4461](https://github.com/ml-explore/mlx/pull/4461) | Разница точности exp в eager/compiled Metal sigmoid. Здесь CPU и Metal теряют комплемент при вычитании из единицы, независимо от этого вопроса. |
| [MLX #575](https://github.com/ml-explore/mlx/issues/575), [#579](https://github.com/ml-explore/mlx/pull/579) | Распространение NaN во forward `logaddexp`; новые входы конечны, нарушаются производные. |
| [nntrainer #193](https://github.com/nntrainer/nntrainer/pull/193), [#257](https://github.com/nntrainer/nntrainer/pull/257) | Добавление совместного sigmoid/CE и исправление обычного forward/усреднения; не подтверждённый отчёт о малом хвосте loss/gradient. |
| [nntrainer #4329](https://github.com/nntrainer/nntrainer/issues/4329) | Наша уже известная ошибка пропуска `loss_scale`. Новая проверка идёт при `loss_scale=1`, поэтому не объясняется этим дефектом. |

Также сверены локальный `MATH_SOFTWARE_CASES_LEDGER_2026-09-04.md` и результаты
предыдущих раундов. Bloom missing scores уже связан с известным PR #56 и не
считается новой находкой. Ошибки активаций и softmax CE этого дня не учитываются
повторно. Два BCE-случая в разных библиотеках группируются в одно семейство
причины; logaddexp имеет отдельное место исправления в нативном autodiff.

Вывод: точного совпадения в проверенной публичной выдаче нет. Приоритет открытия,
отсутствие приватного дубликата и факт регрессии между версиями не установлены.
