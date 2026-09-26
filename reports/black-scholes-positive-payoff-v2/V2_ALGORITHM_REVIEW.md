# Candidate v2: положительный OTM-интеграл и единый log/ordinary контракт

**Статус исполнения после review, 26 сентября:** root до freeze добавил
явное исключение для положительных sigma,T с округлённым v=0 и отдельный
регрессионный тест. Также добавлены предел 32768 quadrature evaluations и
контроль логарифмической границы отброшенных хвостов. Это запись выполненных
действий root, не новая независимая рецензия. Зафиксированный код и результаты
идентифицируются `evidence/freeze-v2.json`; после confirmation код не менялся.

**Дата:** 2026-09-26, Asia/Tashkent  
**Статус:** теоретическая консультация; код и основные файлы не изменялись, вычисления не запускались  
**Ограничения:** только standard-library `math`, один CPU-поток, без GPU и без изменений vendored Jäckel

## Краткий вывод

Для correctness-first candidate v2 рекомендую сделать одну авторитетную ветвь цены: сначала вычислять **логарифм положительного OTM payoff-integral**, затем получать обычную OTM-цену только из этого логарифма. ITM-цена восстанавливается сложением OTM-цены и положительной внутренней стоимости.

Это одновременно устраняет три механизма v1:

1. near-ATM moneyness вычисляется через `log1p`/`frexp`, а не `log(S)-log(K)`;
2. разность близких Mills ratios заменяется одним положительным интегралом;
3. денежный масштаб входит в итоговый `log_price`, поэтому промежуточная нормированная цена не обязана помещаться в binary64.

Цена такого решения — adaptive quadrature. Оно будет существенно медленнее Jäckel и, вероятно, медленнее прямой формулы. Поэтому v2 следует представлять как экспериментальный accuracy baseline или редкий fallback, а не как быстрый production kernel. Локальная quadrature error estimate не является доказательством correct rounding.

---

## 1. Единая положительная формула для OTM call и put

Положим

\[
X=S e^{-qT},\qquad Y=K e^{-rT},\qquad
m=\log(X/Y),\qquad v=\sigma\sqrt T.
\]

Если \(m\le0\), OTM-инструмент — call и \(A=X\). Если \(m>0\), OTM-инструмент — put и \(A=Y\). В обоих случаях определим

\[
z=\frac v2-\frac{|m|}{v}.
\]

Тогда одна и та же формула даёт OTM-цену:

\[
V_{OTM}=\frac{A}{\sqrt{2\pi}}
\int_0^\infty
\exp\!\left[-\frac{(u-z)^2}{2}\right]
\left(1-e^{-vu}\right)du.
\tag{1}
\]

Интегранд неотрицателен. Формула получается из

\[
\Phi(z)=\varphi(z)\int_0^\infty e^{zu-u^2/2}\,du
\]

и тождества, связывающего плотности в \(d_1,d_2\). Для OTM call берётся \(z=d_1\), для OTM put — \(z=-d_2\). После внесения \(\varphi(z)\) под интеграл получается (1).

Практически важное следствие: даже если \(V_{OTM}/\sqrt{XY}\) меньше минимального subnormal, итоговая денежная цена может быть представима. В (1) нормированная цена нигде не округляется в отдельный `float`.

---

## 2. Robust log-moneyness

Нельзя вычислять near-ATM moneyness как `log(S) - log(K)`: два логарифма могут округлиться к одному числу, как в v1 case 0222.

Предлагаемый helper для положительных binary64 `a,b`:

```text
log_ratio_pos(a, b):
    ma, ea = frexp(a)
    mb, eb = frexp(b)
    if abs(ea - eb) <= 1:
        delta = (a - b) / b
        return log1p(delta)
    return fsum([log(ma) - log(mb), (ea - eb) * log(2)])
```

В близком режиме `a-b` сохраняет малую разность, а `log1p` не превращает её в ноль. В далёком режиме `frexp` избегает overflow/underflow отношения `a/b`.

Полный moneyness:

```text
log_sk = log_ratio_pos(S, K)
m = fsum([log_sk, (r-q)*T])
```

Для близких одноимённых конечных binary64-ставок вычитание `r-q` обычно точно по лемме Стербенца; последующее умножение на `T` округляется один раз. Вариант `r*T - q*T` сначала округляет оба произведения и затем может усилить cancellation. Ни одна последовательность не обязана быть точнее для всех входов, поэтому `(r-q)*T` фиксируется как явный binary64 contract v2; математически равные последовательности операций не будут побитово равны.

Денежные лог-масштабы вычисляются отдельно:

```text
logX = fsum([log(S), -(q*T)])
logY = fsum([log(K), -(r*T)])
logA = logX if m <= 0 else logY
```

Обычные `X,Y` для OTM-интеграла не нужны.

---

## 3. Устойчивая quadrature при `z <= 0`: разность Mills как primitive

Пусть \(a=-z\ge0\), \(L=\max(1,a)\), \(u=w/L\), \(b=v/L\). Тогда (1) можно масштабировать без больших или малых значений интегранда.

Определим

\[
H(y)=\frac{-\operatorname{expm1}(-y)}{y},\qquad H(0)=1.
\]

Если \(0<b\le1\), то

\[
J=\int_0^\infty
e^{-(a/L)w-w^2/(2L^2)}\,wH(bw)\,dw,
\]

\[
\log V_{OTM}=\log A-\frac{a^2}{2}-\frac12\log(2\pi)
+\log v-2\log L+\log J.
\tag{2}
\]

Если rounded `b == 0` при положительном `v`, использовать предельный интегранд `w` и всё равно брать prefactor из `log(v)-2*log(L)`. Это не требует представимого `b`.

Если \(b>1\), лучше не выносить большой множитель:

\[
J=\int_0^\infty
e^{-(a/L)w-w^2/(2L^2)}[-\operatorname{expm1}(-bw)]\,dw,
\]

\[
\log V_{OTM}=\log A-\frac{a^2}{2}-\frac12\log(2\pi)
-\log L+\log J.
\tag{3}
\]

Именно (2) является устойчивой реализацией small-vol Mills difference. Для \(a=-z>0\)

\[
M(a)-M(a+v)=
\int_0^\infty e^{-au-u^2/2}(1-e^{-vu})\,du.
\]

Поэтому не нужно вычитать два Mills ratio и не нужно обрезать их асимптотические ряды по отдельности. В режиме больших \(a\), малых \(v\) множитель \(v/L^2\) вынесен аналитически, а оставшийся \(J\) имеет естественный масштаб порядка единицы.

---

## 4. Устойчивая quadrature при `z > 0`

Центр гауссианы лежит внутри области интегрирования. Подстановка \(t=u-z\) даёт

\[
V_{OTM}=\frac{A}{\sqrt{2\pi}}
\int_{-z}^{\infty} e^{-t^2/2}
\left(1-e^{-v(t+z)}\right)dt.
\]

При \(0<v\le1\) вынести малый \(v\):

\[
J=\int_{-z}^{\infty}e^{-t^2/2}(t+z)H(v(t+z))\,dt,
\]

\[
\log V_{OTM}=\log A-\frac12\log(2\pi)+\log v+\log J.
\tag{4}
\]

При \(v>1\) интегрировать `-expm1(-v*(t+z))` напрямую и убрать `log(v)` из (4). Так не возникает искусственного сложения большого `log(v)` с малым `log(J)`.

Экспонента во всех интеграндах не положительна. `expm1` сохраняет малый payoff factor при \(vu\ll1\), а при большом \(vu\) безопасно стремится к `-1`.

---

## 5. Adaptive quadrature без новой зависимости

Для v2 достаточно iterative adaptive Simpson с положительными panel sums. Более дорогой Gauss–Kronrod 15/7 дал бы лучшую локальную оценку, но потребовал бы таблицы узлов и весов; это можно оставить следующей версией.

### 5.1. Конечное усечение

Для `z <= 0` интегрировать по `w` на `[0,W]`. Начальные панели должны обязательно покрывать пик около единицы, например:

```text
[0,1], [1,4], [4,12], [12,W]
```

Нельзя начинать одним Simpson-panel `[0,48]`: значения в `0,24,48` способны выглядеть почти нулевыми и пропустить пик около `w=1`.

Есть простые верхние границы хвоста scaled integral:

- если \(a\ge1\), small-`b` tail не больше \((W+1)e^{-W}\), direct-`b` tail не больше \(e^{-W}\);
- если \(0\le a<1\), small-`b` tail не больше \(e^{-W^2/2}\), direct-`b` tail не больше \(e^{-W^2/2}/W\).

Они следуют из \(0<H(y)\le1\). Увеличивать `W`, пока bound не займёт лишь заданную долю общего error budget.

Для `z > 0` интегрировать `t` на

\[
[\max(-z,-W),W]
\]

с обязательным разбиением в `t=0` и, если присутствуют, в `t=±1`. Для гауссовых хвостов использовать

\[
\int_W^\infty e^{-t^2/2}dt\le e^{-W^2/2}/W,
\qquad
\int_W^\infty t e^{-t^2/2}dt=e^{-W^2/2}.
\]

### 5.2. Локальная оценка

Для панели сравнить один Simpson шаг `S` с двумя половинными шагами `S2`:

```text
err_panel = abs(S2 - S) / 15
```

Принимать `S2` без Richardson correction, чтобы сумма оставалась явно положительной. Накапливать значения и error estimates через `math.fsum`. Обязательны `max_depth`, `max_evaluations` и статус `not_converged`; нельзя возвращать число как успешное после исчерпания бюджета.

Полезная диагностика:

```text
quadrature_value
quadrature_error_estimate
analytic_tail_bound
function_evaluations
max_depth_reached
converged
regime = tail_scaled | centered
```

`abs(S2-S)/15` оценивает truncation/discretization локально, но не ограничивает ошибки `exp`, `expm1`, входного mapping и summation. Это **не интервальный сертификат** и не доказательство correctly rounded binary64.

### 5.3. Ожидаемая скорость

Одна цена потребует от десятков до сотен вызовов `exp/expm1`, а трудный случай — больше. Jäckel использует несколько special-function evaluations и алгебраические ветви. Поэтому integral v2 по определению не должен заявляться быстрым улучшением. После accuracy-проверки его можно оставить:

- независимым float64 cross-formula baseline;
- fallback для плохо обусловленной аналитической разности;
- генератором log-price для случаев промежуточного underflow.

---

## 6. Восстановление ordinary price только из итогового log

Пусть `log_otm` получен из (2)–(4). Не нужно отдельно вычислять ordinary OTM formula: две независимые ветви снова разойдутся.

Безопасное восстановление использует разложение по степени двойки:

```text
exp_from_log(logx):
    if logx == -inf: return 0
    k = floor(logx / log(2))
    r = fsum([logx, -(k * log(2))])
    # adjust k,r if rounding moved r outside [0, log(2))
    y = exp(r)                 # normal-sized: 1 <= y < 2
    return ldexp(y, k)         # единственное место normal/subnormal rounding
```

До `ldexp` проверить overflow и значения ниже half-min-subnormal. При точном пороге \(2^{-1075}\) ties-to-even округляет к нулю. Поскольку сам `log_otm` и `log(2)` округлены, helper не является доказательством правильного округления около границы; это проверяет multiprecision oracle.

Ключевой контракт: `log_otm` сохраняется даже когда ordinary price стал нулём. `underflow=True` означает `ordinary_otm == 0` при конечном `log_otm`.

Такая реконструкция исправляет механизм case 0197: если итоговая денежная цена представима, промежуточная normalized price не успевает округлиться в ноль. Для case 0314 финальное `ldexp` должно вернуть subnormal, если `log_otm` достаточно точен; сам факт конечного лога этого не гарантирует.

---

## 7. Паритет и полная пара цен

Внутреннюю стоимость также лучше строить из логарифма. При \(m\ne0\):

```text
log_high = logX if m > 0 else logY
log_abs_parity = log_high + log(-expm1(-abs(m)))
abs_parity = exp_from_log(log_abs_parity)
```

Далее используются только сложения неотрицательных чисел:

```text
if m <= 0:
    call = otm
    put  = fsum([otm, abs_parity])
else:
    put  = otm
    call = fsum([otm, abs_parity])
```

При `m == 0` parity равен нулю. При `T == 0` или `sigma == 0` вернуть только intrinsic limits и не запускать quadrature.

`time_value` обеих legs равна вычисленной OTM-цене в согласованном parитетном контракте. В расширенной диагностике полезно сохранять `log_abs_parity`, чтобы underflow intrinsic и time value не смешивались.

---

## 8. Компактный pseudocode candidate v2

```text
validate finite inputs, S>0, K>0, T>=0, sigma>=0

log_sk = log_ratio_pos(S, K)
logX = fsum([log(S), -q*T])
logY = fsum([log(K), -r*T])
m = fsum([log_sk, r*T, -q*T])

compute signed parity through log_abs_parity
if T == 0 or sigma == 0:
    return intrinsic pair

v = sigma * sqrt(T)
require finite v; handle rounded v==0 as an explicit limit/status
z = fsum([0.5*v, -(abs(m)/v)])
logA = logX if m <= 0 else logY

if z <= 0:
    log_kernel, estimate = scaled_tail_integral(z, v)
else:
    log_kernel, estimate = centered_integral(z, v)

log_otm = logA - 0.5*log(2*pi) + log_kernel
otm = exp_from_log(log_otm)
reconstruct ITM leg by positive addition with abs_parity
return ordinary pair + log_otm + quadrature diagnostics
```

Здесь `log_kernel` включает все вынесенные множители из (2)–(4), кроме общих `logA - 0.5*log(2*pi)`: например, tail-helper возвращает `-a²/2 + log(v) - 2*log(L) + log(J)` в small-`b` режиме. Нельзя повторно добавить `log(v)`, `-a²/2` или scale factor в caller.

---

## 9. Проверки до новой confirmation выборки

Старый v1 confirmation уже является development data. На нём допустимо отлаживать v2, но итоговые числа должны оцениваться на новой заранее замороженной выборке.

Обязательные development-проверки:

1. case 0222: знак robust `m` не теряется;
2. case 0158: positive integral согласуется с oracle лучше v1 Mills subtraction;
3. case 0197: final monetary price не становится нулём из-за normalized intermediate;
4. case 0314: ordinary/log ветви согласованы на subnormal output;
5. все OTM integrands и panel sums неотрицательны без `max(value,0)`;
6. put–call parity, call/put symmetry и монотонность по \(v\);
7. пределы \(v\to0\), \(m\to0^\pm\), большие \(|m|/v\), normal/subnormal/zero boundaries;
8. `quadrature_error_estimate + tail_bound` коррелирует с фактической ошибкой, но не используется как гарантия;
9. один и тот же `log_otm` является источником ordinary price и `DetailedPrice`;
10. исчерпание quadrature budget возвращает явный failure status, а не молчаливое число.

Новая confirmation должна отдельно стратифицировать:

- near-ATM adjacent floats во многих binades;
- small \(v\) и большие \(|m|/v\);
- конечная денежная цена при normalized underflow;
- значения по обе стороны half-min-subnormal;
- nonzero carry с отменой между spot moneyness и \((r-q)T\);
- центральные обычные случаи, где integral не должен ухудшать точность.

---

## 10. Риски и контрпримеры

- Adaptive Simpson может недооценить ошибку гладкой функции, если начальная сетка пропустила локальную структуру. Поэтому нужны предразбитые панели, tail bound и evaluation cap.
- Положительность устраняет cancellation, но не гарантирует малую relative error: ошибки `m`, `v`, `z` экспоненциально усиливаются в хвосте.
- `log1p((S-K)/K)` решает near-ATM потерю, но carry cancellation остаётся обусловленной задачей; `fsum` уменьшает, а не отменяет input rounding.
- При экстремальном `z` величина `z*z` может overflow. Тогда ordinary price заведомо нулевой в допустимом денежном масштабе, но log API может потерять конечность; нужен явный `log_underflow_beyond_range` status.
- Quadrature v2 может быть точнее на хвостах и одновременно хуже в центральной области из-за накопления сотен libm/summation errors.
- Ошибка функции `exp_from_log` около half-subnormal способна изменить последний бит. Correct rounding остаётся задачей oracle, не следствием формулы.
- Даже успешный v2 улучшает вычисление той же BSM-цены; он не улучшает рыночную модель, implied-vol conditioning или hedging.

## Рекомендация реализации

Для первого v2 реализовать положительный integral path как единый research candidate и вернуть полную диагностику. Не добавлять быстрые эвристические ветви до замораживания его собственного development behaviour. После новой confirmation-проверки решать отдельно:

1. оставить integral path как медленный reference/fallback;
2. заменить Simpson на Gauss–Kronrod;
3. вывести быстрый analytic small-vol approximation с integral path как oracle;
4. отказаться от v2, если центральная ошибка или стоимость не оправдывают хвостовой выигрыш.

Vendored Jäckel должен остаться побитово неизменным baseline.

---

## Дополнение 2026-09-26: read-only проверка реализации GL16/32

Проверены `lab/black_scholes_v2.py` и `EXPERIMENT_V2.md` до заморозки. Код не изменялся и численные прогоны не выполнялись.

### Формулы

Обе реализованные ветви алгебраически соответствуют (1).

- При `z <= 0` код использует \(L=\max(1,-z)\), \(b=v/L\), интегранд
  \[
  e^{(z/L)w-w^2/(2L^2)}\frac{1-e^{-bw}}{b},
  \]
  и внешние множители \(e^{-z^2/2}v/L^2\). Их произведение после замены \(u=w/L\) точно даёт (1). Предельная ветвь `b == 0`, где quotient заменён на `w`, также правильна.
- При `z > 0` интегрируется
  \[
  e^{-(u-z)^2/2}\frac{1-e^{-vu}}{v}
  \]
  на гауссовом окне и снаружи возвращается множитель \(v\). Это та же формула без разрушительного small-\(v\) масштаба.
- `log_relative` содержит ровно те же \(v,L,J,1/\sqrt{2\pi}\) и gaussian scale, что `factors` плюс `log_density` ordinary-ветви. Повторного или пропущенного множителя не найдено.

### `scaled_product`

Разложение `base` и положительных factors через `frexp`, внесение `exp(exponential)` посредством целой степени двойки и окончательный `ldexp` алгебраически корректны. Оно сохраняет больше информации денежного масштаба, чем `exp(log(base) + ...)`, поэтому ordinary price не обязан быть буквально `exp(log_otm_value)`. Корректный контракт здесь такой:

- ordinary и log используют один integral и один набор математических факторов;
- ordinary сохраняет исходную значащую часть `base`;
- log является согласованным логарифмическим представлением, но округляется отдельно через `log(base)`.

Это лучше описать как **two-view common-factor contract**, а не «ordinary восстановлена из округлённого полного log». Около subnormal boundary возможна двойная округлённость: сначала `mantissa*exp(residual)`, затем `ldexp`. Поэтому helper не является correct-rounding proof; предусмотренный отдельный boundary test и допуск в один ULP остаются необходимыми.

### Единственный pre-freeze blocker

Сейчас

```python
v = sigma * math.sqrt(t)
...
if v == 0.0:
    return deterministic_result
```

смешивает два разных случая:

1. математически детерминированный `sigma == 0` или `t == 0`;
2. положительные `sigma,t`, произведение которых округлилось к binary64 zero.

Во втором случае OTM-цена не обязана округляться к нулю после умножения на большой денежный масштаб. Например, положительный total volatility порядка half-min-subnormal при ATM и денежном масштабе порядка максимальной нормальной степени даёт вполне представимую временную стоимость, хотя `sigma*sqrt(t)` уже равно нулю в binary64.

До freeze нужен один из двух явных вариантов:

- поддержать этот диапазон через `log_v = log(sigma) + 0.5*log(t)` и отдельный предельный path;
- либо при `sigma > 0`, `t > 0`, `v == 0` возвращать явный unsupported-range error, а не детерминированную цену.

Второй вариант достаточен для заявленного v2 corpus, где total volatility существенно выше этой границы. Молчаливый deterministic result противоречит текущему широкому `_validate` contract и является blocker.

### Границы, которые не блокируют заявленный эксперимент

- В `z <= 0` фиксированное усечение `w=64` имеет аналитически малый пропущенный хвост: при \(-z\ge1\) scaled integrand ограничен хвостом типа \(w e^{-w}\), а при \(-z<1\) — gaussian tail. Он намного ниже практических gates, но не включён в `quadrature_relative_estimate`; это следует прямо записать в evidence schema/описании.
- В `z > 0` окно `[max(0,z-12), z+12]` пропускает только gaussian tails за 12 стандартными отклонениями. Условие OTM даёт `z <= v/2`, поэтому при большом `z` payoff factor около центра не создаёт отдельного узкого пика вне окна. Tail также не входит в возвращённый estimate.
- Разность GL32–GL16 плюс relative roundoff floor — полезный индикатор, но не верхняя граница общей ошибки. Relative floor применяется на каждом принятом leaf и может суммарно превысить `QUAD_TOL`; документация уже не называет estimate сертификатом.
- Реализация всегда выносит `v` даже при большом `b`/`v`. Это точное тождество и устойчиво в заявленном confirmation диапазоне `v <= 20`. Для универсального API на всех конечных binary64 оно может превратить integral в subnormal и сделать GL16/32 agreement ложно оптимистичным; либо ограничить поддерживаемый диапазон, либо позднее добавить direct-payoff branch для `b > 1`/`v > 1`.
- `m = fsum(log_ratio(S,K), (r-q)*T)` является допустимым зафиксированным binary64 mapping. Он отличается от `fsum(log_ratio, r*T, -q*T)`; в evidence надо хранить именно выбранную последовательность и не смешивать её с oracle mapping error.

### Решение перед freeze

После исправления или явного отклонения positive-input `v == 0` формульных blockers в заявленном диапазоне не видно. GL truncation, estimator semantics и two-view ordinary/log contract должны остаться явно задокументированными ограничениями; они не требуют изменения vendored baseline.
