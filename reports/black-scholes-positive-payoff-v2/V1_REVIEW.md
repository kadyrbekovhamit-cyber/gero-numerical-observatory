# Внутренняя рецензия benchmark v1

Дата: 2026-09-26, Asia/Tashkent  
Статус: независимый AI review, не внешнее peer review  
Режим: только чтение; benchmark и тесты повторно не запускались; основные файлы не изменялись

## Вердикт

v1 существенно исправляет методику v0. Контракт exact binary64, high-precision ошибки, отдельное subnormal-округление, разделение development/confirmation, фиксация кандидата и атрибутированный исторический Jäckel baseline реализованы последовательно. Числа 316 случаев, 199 confirmation и результат 15/152/3 на 170 confirmation-ценах, правильно округляющихся в ненулевой binary64, согласуются со структурой сохранённого отчёта.

Первоначальная рецензия выявила два блокирующих вопроса и несколько обязательных ограничений формулировок. Их текущий статус указан ниже.

## Статус устранения — 2026-09-26, Asia/Tashkent

Оба blocker ниже закрыты. Этот статус получен read-only проверкой сохранённых файлов; benchmark и диагностические вычисления повторно не запускались.

1. `lab/run_benchmark_v1.py` теперь задаёт `comparable_same_leg` до агрегации. Для 12 wrong-side строк исходная cross-leg величина сохранена в `cross_leg_contract_discrepancy`, а поля same-leg `absolute` и `real_ulp_error` равны `null`. Сводка использует только 304 сопоставимых значения: 117 development и 187 confirmation; 12 confirmation mismatches считаются отдельно. Максимум остаётся `confirmation-0158`, где стороны совпадают.
2. Первый результат сохранён в `evidence/archive-v1-initial` с manifest, объясняющим исправление измерения. Read-only сравнение всех 316 строк подтвердило идентичность `methods`, `both_legs` и `comparisons` между первоначальным и конечным evidence: численные цены кандидата и price comparisons не менялись.
3. `evidence/diagnostics-v1.json` явно помечен как `post-hoc diagnostic, not an additional holdout`. Для `confirmation-0158`, `0197`, `0222` и `0314` сохранены согласующиеся oracle evaluations при 260/360 decimal digits и положительный integral cross-check при 120 digits; все четыре имеют `passed: true`. Тем самым независимая проверка необычных нулей `0197` и `0314`, затребованная ниже, выполнена в пределах заявленного эмпирического oracle-контракта.
4. `TECHNICAL_NOTE_V1.md` отделяет 12 interface contract failures от same-leg log-error, называет четыре проверки post hoc, сохраняет ограничения `mpmath`/не-interval certificate, не выдаёт нули исторической ревизии за победы кандидата и не заявляет новизну или production superiority.

Блокирующих замечаний в проверенном объёме больше нет. Разделы ниже сохраняются как история первоначальной рецензии; их требования считаются выполненными указанными артефактами. Ограничения относительно синтетического corpus, historical pinned revision, carry mapping, отсутствия interval certificate и внешнего peer review остаются в силе.

## Первоначальные blockers — история замечаний

### 1. Двенадцать log-price сравнений относятся к разным опционам

В 12 `monetary_scale` случаях high-precision oracle выбирает OTM put, поскольку точное log-forward-moneyness замороженных binary64 слегка положительно. `stable_detailed` из-за округления `log(s)-log(k)` до нуля выбирает call. Однако `candidate_log.log_otm_value` затем сравнивается с `ref.log_otm` put и входит в общий log-price aggregate.

Это не корректная ошибка одного и того же математического объекта. Сама потеря ориентации является реальным дефектом кандидата: при `r=q=0` соседние binary64 `S` и `K` могут схлопнуться после маршрута `log` → `exp`, а parity стать нулём. Но log-метрика должна:

- либо вычислять логарифм именно той стороны, которую выбрал oracle;
- либо маркировать эти 12 строк `orientation_mismatch` и исключать их из log-error max/quantiles;
- отдельно сообщать mismatch как самостоятельный результат.

До этого нельзя утверждать, что агрегат `candidate_log` описывает 316 сопоставимых log-price значений. Наблюдаемый общий максимум всё же не создан mismatch: `confirmation-0158` имеет согласованную ориентацию put, но это надо сказать явно.

### 2. Два неожиданных нуля Jäckel не прошли независимый интегральный cross-check

Неожиданные нули находятся в:

- `confirmation-0197`: put-цена около \(1.24\cdot10^{-300}\), округляющаяся в нормальное ненулевое binary64;
- `confirmation-0314`: subnormal put-цена около \(7.39\cdot10^{-318}\).

Оба oracle значения сходятся между двумя precision levels, но эти индексы не входят в 14 заранее выбранных integral checks. Поскольку нули исторического baseline являются необычным и потенциально цитируемым наблюдением, перед технической запиской их надо проверить положительным интегралом или вторым независимым multiprecision implementation. Такой follow-up является проверкой уже найденных исключений, а не новым confirmation test; его следует пометить post hoc.

До этой проверки допустима формулировка «в сохранённом прогоне pinned revision вернула ноль», но не «Jäckel метод теряет эти цены вообще».

## Проверка новых метрик

### Что корректно

- Fixtures восстанавливают все шесть BSM входов из `float.hex()`; oracle получает те же binary64 значения.
- `_error` v0 заменён на high-precision `price_error`; абсолютная и относительная ошибки не округляются в `float` перед сериализацией.
- `round_binary64` отдельно округляет в единицах \(2^{-1074}\), включая tie-to-even в половине minimum subnormal; unit test проверяет \(0.5\eta\), значения по обе стороны и \(1.5\eta\).
- `rounded_ulp_distance` и `real_ulp_error` отвечают на разные вопросы и вычисляются раздельно.
- `unexpected_zero` считается только на `normal` и `subnormal`, поэтому 29 confirmation-случаев, корректно округляющихся в ноль, не загрязняют счёт 15/152/3.
- Oracle convergence хранит обе попытки; 8 эскалаций относятся к development, а failure равен нулю.
- 14 положительных integral checks прошли, но это эмпирический cross-check общей арифметики `mpmath`, не interval certificate; отчёт это честно указывает.
- Jäckel sources, wrapper и binary связаны SHA-256; floating environment фиксирует nearest-even и gradual underflow.
- Для carry случаев отдельно сохранены mapping error и native evaluation error.

### Что требует осторожной терминологии

`rounding_class == "subnormal"` означает, что **округлённый** результат subnormal. Это не отдельный флаг «точная цена меньше minimum subnormal». В corpus также нет цены, специально расположенной непосредственно по обе стороны \(2^{-1075}\): точная midpoint-логика проверяется unit test, а не 316-case corpus. Поэтому нельзя писать, что corpus эмпирически исследовал тесную окрестность half-minimum-subnormal boundary.

`round_binary64` опирается на converged `mpmath` reference, а не на доказанный интервал относительно каждого midpoint. Термины `rounded converged reference` и `empirical multiprecision oracle` корректны; `certified correctly rounded oracle` — нет.

`rounded_ulp_distance` для огромной ошибки удобен как диагностика, но не как линейная мера качества через разные binades. В выводах рядом должны оставаться relative error и обычная absolute error.

## Проверка заявленных итогов

На confirmation partition сохранено:

- 199 случаев: 166 normal, 4 subnormal, 29 rounds-to-zero;
- на 170 ненулевых округлённых ценах candidate против Jäckel: **15 candidate closer, 152 Jäckel closer, 3 equal absolute error**;
- 0 oracle failures, 0 precision escalations, 9 integral checks без failures;
- 0 clipping и 0 отрицательных/NaN/Inf OTM цен у всех трёх методов;
- 12 OTM orientation mismatches кандидата;
- 2 неожиданных нуля pinned Jäckel revision.

Эти win/tie counts являются описанием именно замороженного синтетического corpus. Случаи не являются независимой выборкой рынка, имеют разный масштаб и разную трудность; counts не являются вероятностью превосходства и не поддерживают статистический тест общего качества.

Важные отрицательные результаты кандидата должны находиться в основном выводе, а не только в таблице:

- Jäckel ближе в 152 из 170 ненулевых confirmation случаев;
- worst confirmation relative error кандидата равен примерно 38.05 в `confirmation-0314`;
- worst comparable candidate log error находится в `confirmation-0158`: absolute log error около \(3.25\cdot10^{-5}\), примерно \(4.57\cdot10^9\) ULP log-price;
- конечность log-price во всех строках не означает его высокую точность.

Оба неожиданных нуля Jäckel нельзя представлять как победы кандидата. По absolute error ноль Jäckel ближе к oracle в обоих случаях: candidate имеет relative error около 5.02 в `confirmation-0197` и около 38.05 в `confirmation-0314`, тогда как нулевой результат имеет relative error 1.

## Допустимая формулировка технической записки

> На замороженном синтетическом confirmation corpus из 199 случаев 170 OTM-цен правильно округляются в ненулевую binary64. По абсолютной ошибке относительно converged multiprecision reference замороженный кандидат был ближе pinned historical Jäckel revision в 15 случаях, Jäckel — в 152, ошибки совпали в 3. Результат отвергает универсальное превосходство кандидата. В 12 около-ATM scale cases кандидат потерял знак точного log-forward-moneyness; два extreme-tail cases дали ненулевую reference price, но pinned Jäckel revision вернула zero. Эти исключения требуют отдельного анализа. Исследование не измеряет скорость, production suitability, market fit, hedging quality или научную новизну.

Для 18 carry cases следует отдельно писать, что основное число является end-to-end сравнением: Jäckel получает округлённые \(F\) и \(D\), а mapping error может усиливать или компенсировать native evaluation error. Его нельзя называть чистым сравнением нормированных kernels.

## Условие готовности к технической записке

1. Исправить или исключить 12 несопоставимых log-price строк и пересобрать только соответствующую сводку с новой версией evidence.
2. Независимо проверить oracle для `confirmation-0197` и `confirmation-0314`; сохранить post-hoc receipt.
3. Вынести худшие ошибки и случаи проигрыша кандидата в основной текст.
4. Использовать 15/152/3 только с расшифровкой порядка, denominator 170, метрики absolute error и ограничений corpus.
5. Сохранить формулировки «pinned historical revision», «converged reference» и «internal AI review»; не заявлять новизну, production superiority или внешнее рецензирование.
