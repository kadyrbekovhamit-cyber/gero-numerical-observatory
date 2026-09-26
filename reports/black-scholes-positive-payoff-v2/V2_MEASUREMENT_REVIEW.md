# Read-only review измерений frozen v2

Дата: 2026-09-26, Asia/Tashkent  
Статус: внутренний AI review, не внешнее peer review  
Проверены: `EXPERIMENT_V2.md`, `lab/run_benchmark_v2.py`, `evidence/freeze-v2.json`, `evidence/benchmark-v2.json`  
Режим: сохранённые файлы прочитаны; benchmark, тесты и численные расчёты не запускались; основные файлы не изменялись

## Вердикт

Сохранённый v2 evidence согласован с заявленным экспериментом. Метрики и счётчики поддерживают ограниченный вывод: frozen v2 прошёл заранее заданные practical gates на новом синтетическом corpus и чаще был ближе к multiprecision reference, чем обе сохранённые реализации, на ненулевой части этого corpus. Результат не доказывает правильное округление, глобальное превосходство, новизну, production suitability или преимущество по скорости.

Критических ошибок в вычислении сводки по прочитанному коду не найдено. Есть важное ограничение freeze provenance: заморожены кандидат и corpus, но не весь исполняемый measurement pipeline.

## Проверка чисел и знаменателей

В evidence находятся 288 новых `confirmation_v2` случаев:

- 96 `near_atm`;
- 96 `normalized`;
- 72 `nonzero_carry`;
- 24 `tail_and_range`.

Reference rounding classes: 201 normal, 1 subnormal, 86 rounds-to-zero. Поэтому baseline comparisons выполняются на 202 ненулевых округлённых ценах. Сохранённые counts имеют следующий однозначный порядок:

- v2 против v0: **192 v2 closer / 3 v0 closer / 7 equal absolute error**;
- v2 против pinned historical Jäckel: **146 v2 closer / 30 Jäckel closer / 26 equal absolute error**.

Counts используют high-precision absolute error, а не факт прохождения gate. Они описывают эти 202 fixtures и не являются вероятностью превосходства на рынке. Carry comparison остаётся end-to-end: округление \(F,D\) входит в результат Jäckel и отдельно диагностируется mapping fields.

## Корректность practical gates

Реализация соответствует протоколу:

\[
|V_{v2}-V_*|\le
\max\bigl(10^{-10}V_*,\operatorname{ulp}(\operatorname{RN}_{64}(V_*))\bigr),
\]

\[
|L_{v2}-L_*|\le
\max\bigl(2\cdot10^{-11},16\operatorname{ulp}(\operatorname{RN}_{64}(L_*))\bigr).
\]

Все 288 строк прошли price, log, finite/nonnegative, correct-side и no-unexpected-zero gates; oracle failures, v2 exceptions, orientation mismatches и integral failures равны нулю. Все 288 log values имеют ту же OTM-сторону, что oracle.

Слово `max` существенно:

- price gate означает «relative budget **или** one-ULP absolute budget, смотря что больше»; это не correct-rounding gate;
- log gate означает «absolute \(2\cdot10^{-11}\) **или** 16 ULP, смотря что больше»; это не утверждение `<=16 ULP` во всех случаях.

В частности, сводный worst log ULP равен примерно \(2.263175\cdot10^6\) в `v2-0163`, хотя absolute log error там около \(1.20\cdot10^{-16}\) и gate проходит, потому что около нулевого log-price абсолютный допуск доминирует. Поэтому допустимо писать «все строки прошли заранее заданный max-gate», но нельзя писать «log error не превышает 16 ULP».

Также прохождение `finite_nonnegative_both_legs` проверяет конечность и знак обеих цен, а не точность ITM leg, parity, monotonicity или Greeks на confirmation corpus. Основная измеренная величина — OTM price плюс same-leg log-price.

## Oracle и integral checks

Primary oracle — exact-binary64 `mpmath` erfc formula с проверкой 100/180 digits и предусмотренным retry 260; в v2 retries не потребовались. Это независимое от GL16/GL32 вычислительное представление формулы цены, но не interval certificate.

Десять заранее заданных secondary checks при 100 digits прошли порог абсолютного расхождения log-price \(10^{-50}\). Secondary oracle и кандидат используют одну положительную payoff-integral identity, хотя вычисляют её разными средствами: adaptive multiprecision quadrature против binary64 GL16/GL32. Поэтому эти 10 checks подтверждают реализацию и согласие с primary erfc oracle, но не являются независимым математическим выводом или полной проверкой всех 288 quadratures.

`quadrature_relative_estimate` кандидата — расхождение GL16/GL32 с локальным refinement; tail window имеет отдельную аналитическую оценку. Ни то ни другое не включает всю ошибку `log`, `exp`, подготовки входов и binary64 reconstruction. Evidence и протокол это ограничение признают.

## Freeze audit

Сильная часть freeze:

- corpus содержит только exact binary64 hex inputs и имеет SHA-256 в receipt;
- v2 candidate, v0 baseline, primary oracle, metrics, corpus generator, protocol и v2 tests перечислены в `source_sha256`;
- runner до оценки проверяет эти hashes;
- все 316 v1 cases явно переведены в development, новый seed/corpus не смешан с ними;
- конечный evidence встраивает freeze receipt, environment, runner SHA и pinned Jäckel build manifest;
- protocol запрещает изменение кандидата после просмотра confirmation; дальнейшее изменение `black_scholes_v2.py` требует новой версии и нового confirmation corpus.

Ограничение: `freeze-v2.json` не содержит pre-run hashes для `lab/run_benchmark_v2.py`, импортируемого `lab/corpus.py` и функций `comparison`, `native_batch`, `safe` из `lab/run_benchmark_v1.py`. SHA runner сохранён только в конечном отчёте. Receipt также является локальной записью без независимой внешней временной метки.

Следовательно, доказуемая формулировка — **locally frozen candidate and confirmation corpus with an auditable final runner**. Формулировка «весь исполняемый benchmark pipeline был криптографически заморожен до первого просмотра результатов» этими файлами не подтверждается. Это provenance limitation, а не обнаруженное изменение кандидата; сохранённый кандидат связан с pre-run hash корректно.

## Допустимая интерпретация

> На новом замороженном синтетическом corpus из 288 exact-binary64 BSM случаев frozen v2 прошёл заранее заданные practical price/log/interface gates. Среди 202 цен, округляющихся в ненулевую binary64, v2 имел меньшую absolute error, чем v0, в 192 случаях и меньшую, чем pinned historical Jäckel, в 146 случаях; соответствующие baselines были ближе в 3 и 30 случаях, ещё 7 и 26 сравнений дали равную ошибку. Primary reference является empirically converged multiprecision erfc oracle, а не interval-certified truth. Результат относится только к этому corpus и не содержит заявлений о скорости, рынке, хеджировании, production или глобальной новизне.

Практические ограничения, которые должны остаться рядом с результатом:

- 86 zero-rounding cases исключены из win counts;
- только один ненулевой subnormal case представлен в denominator 202;
- 59 из 72 carry cases округляются в ноль, поэтому ненулевой carry comparison основан на 13 случаях;
- GL quadrature может быть медленнее; timing experiment не проводился;
- unsupported positive-volatility underflow и exhausted quadrature budget должны оставаться явными exceptions вне подтверждённого corpus;
- CPU1/no-GPU — условие сохранённого протокола и последовательной реализации, не benchmark скорости.

## Итоговое решение

Frozen v2 можно использовать как внутренний положительный accuracy result с указанными знаменателями и practical-gate semantics. Изменять candidate v2 после просмотра confirmation нельзя. Любое исправление или расширение численного ядра должно получить новое имя версии и новый confirmation corpus. Перед внешней технической запиской достаточно сохранить перечисленные ограничения; global novelty, production superiority и speed superiority заявлять нельзя.
