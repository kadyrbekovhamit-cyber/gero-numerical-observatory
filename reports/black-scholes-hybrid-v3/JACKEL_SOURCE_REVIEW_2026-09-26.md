# Peter Jäckel, *Let's Be Rational*: обзор источника и callable contract

**Дата проверки:** 2026-09-26, Asia/Tashkent  
**Назначение:** выбрать законный, маленький и неизменённый upstream baseline для проекта Black–Scholes  
**Действия:** код и архив не скачивались, не собирались и не запускались; тяжёлые вычисления не выполнялись

## Краткая рекомендация

Использовать исходный C++ core из официального архива Jäckel как vendored baseline, не изменяя ни одной upstream-строки. Добавить только отдельный `jackel_cli.cpp`, который вызывает:

```cpp
double black(double F, double K, double sigma, double T, double q);
double normalised_black(double x, double s, double q);
double normalised_black_call(double x, double s);
```

Первый контракт возвращает **недисконтированную Black/Black-76 цену**; `q` здесь означает знак опциона (`+1` call, `-1` put), а не dividend yield. Для BSM со spot-входами результат равен

\[
V_{BSM}=D\,\operatorname{black}(F,K,\sigma,T,\omega),\quad
F=S_0e^{(r-q_d)T},\quad D=e^{-rT},
\]

где \(q_d\) — dividend yield, а \(\omega\in\{+1,-1\}\) — тип опциона.

Официальный URL архива: [https://www.jaeckel.org/LetsBeRational.7z](https://www.jaeckel.org/LetsBeRational.7z). Его прямо указывает издательская страница статьи [Wiley, DOI 10.1002/wilm.10395](https://onlinelibrary.wiley.com/doi/10.1002/wilm.10395). На момент этой проверки официальный домен не ответил через доступные web/HEAD-каналы. Поэтому **состав текущего `.7z` и его хеш не проверены**, а зеркало нельзя объявлять побитовой копией официального архива до сравнения хешей.

## 1. Что подтверждено первичными источниками

Статья опубликована 12 марта 2015 года в *Wilmott*, выпуск 75, страницы 40–53. В abstract заявлены четыре рациональные ветви начального приближения, Householder order 4 и достижение максимально доступной binary64-точности за очень малое число итераций; там же официальный архив назван reference implementation [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/wilm.10395).

Ссылка `[Jäc13]` в статье описывает `LetsBeRational.7z` как «Implementing Let's be rational», November 2013. Это дата первой цитируемой реализации, а не обязательно дата последнего файла в доступном архиве.

Прозрачное GitHub-зеркало [vollib/lets_be_rational](https://github.com/vollib/lets_be_rational) показывает C++ core с C ABI. Его `version.h` содержит:

- generated: `2014-11-01 14:47:05`;
- `REVISION 990`;
- `VERSION "1.0.0.990"`;
- copyright `© 2013-2014 Peter Jäckel`.

Это сильное свидетельство более поздней ревизии core, но не доказательство того, что официальный URL 2026 года отдаёт тот же набор байтов. Версия Python/SWIG-пакета `1.0.9` в `setup.py` относится к оболочке vollib и не должна использоваться как версия алгоритма.

## 2. Лицензия и attribution

В заголовках Jäckel-файлов указано свободное разрешение использовать, копировать, изменять и распространять код при условии сохранения copyright/permission notice; далее следует отказ от гарантий. Формулировка является собственным коротким permissive notice. Не следует автоматически переименовывать её в SPDX `MIT`, хотя wrapper vollib маркирует свой пакет как MIT.

Практический контракт распространения:

1. сохранить upstream headers дословно во всех исходных файлах;
2. не смешивать локальные исправления с vendored core; wrapper и adapter держать отдельно;
3. положить рядом `UPSTREAM_NOTICE.md` с автором, официальным URL, статьёй/DOI, датой получения, именем архива, SHA-256 и перечнем файлов;
4. если core изменён, явно назвать patch и публиковать diff; такой вариант уже не считать «неизменённым baseline»;
5. сохранить provenance `erf_cody.cpp`: файл говорит, что это переведённый и вручную адаптированный Netlib/SPECFUN Fortran W. J. Cody. В зеркале этот файл содержит provenance и warranty disclaimer, но не повторяет полное permission grant Jäckel. Поэтому сохранять весь header, отдельно кредитовать Cody/Netlib и не делать более сильного юридического вывода без содержимого официального архива.

Тексты разрешения видны в [LetsBeRational.cpp](https://raw.githubusercontent.com/vollib/lets_be_rational/master/src/LetsBeRational.cpp), [rationalcubic.cpp](https://raw.githubusercontent.com/vollib/lets_be_rational/master/src/rationalcubic.cpp) и headers; provenance Cody — в [erf_cody.cpp](https://raw.githubusercontent.com/vollib/lets_be_rational/master/src/erf_cody.cpp).

## 3. Минимальный C++ baseline и зависимости

По прозрачному зеркалу минимальный standalone core состоит из:

```text
LetsBeRational.cpp
erf_cody.cpp
normaldistribution.cpp
normaldistribution.h
rationalcubic.cpp
rationalcubic.h
importexport.h
```

`version.h` полезен для provenance, но core из зеркала его напрямую не включает. `LetsBeRational.i`, сгенерированный SWIG-код, Python package и `setup.py` для CLI не нужны.

Зависимости runtime/build:

- только C++ standard library и системная math library;
- внешних numerical libraries, Python headers, SWIG, OpenMP или thread runtime для standalone core нет;
- код скалярный, сам потоков не создаёт;
- `clang++` на macOS должен собрать core одним процессом.

Предлагаемая команда после законного получения файлов:

```sh
clang++ -std=c++11 -O3 -DNDEBUG \
  jackel_cli.cpp erf_cody.cpp normaldistribution.cpp \
  rationalcubic.cpp LetsBeRational.cpp -o jackel_cli
```

Эта команда **не запускалась**. Upstream-комментарий предлагает для shared library `g++ ... -Ofast`, но для accuracy baseline сначала лучше использовать `-O3` без `-ffast-math`: `-Ofast` допускает преобразования, меняющие IEEE-результат, NaN/Inf и порядок операций. Если нужен профиль с upstream-suggested flags, собирать его вторым артефактом с отдельным именем и не смешивать результаты.

Wrapper должен:

- объявить upstream functions через `extern "C"`;
- читать много строк из `stdin` в одном процессе, чтобы launch overhead не попал в benchmark;
- печатать `double` как минимум с `max_digits10`, а лучше одновременно decimal и C99 hexfloat;
- проверять входной contract до вызова и помечать invalid rows;
- не превращать sentinel `±DBL_MAX` implied-vol API в обычную volatility.

## 4. Точный callable contract

### 4.1. Forward-price function

Исходник экспортирует:

```cpp
double black(double F, double K, double sigma, double T, double q /* q=±1 */);
```

По реализации:

```cpp
return max(intrinsic,
           sqrt(F)*sqrt(K) *
           normalised_black(log(F/K), sigma*sqrt(T), q));
```

Смысл аргументов:

| Аргумент | Смысл | Допустимая область для baseline |
|---|---|---|
| `F` | forward underlying на срок `T` | конечный `F > 0` |
| `K` | strike в тех же единицах | конечный `K > 0` |
| `sigma` | годовая Black volatility | конечная `sigma >= 0` |
| `T` | year fraction | конечная `T >= 0` |
| `q` | option sign | ровно `+1.0` call или `-1.0` put |

Возвращаемое значение — цена в forward units, без discount factor:

\[
B(F,K,\sigma,T,\omega)
=\omega\left[F\Phi(\omega d_1)-K\Phi(\omega d_2)\right],
\]

\[
d_{1,2}=\frac{\log(F/K)}{\sigma\sqrt T}\pm\frac12\sigma\sqrt T.
\]

Код не выполняет полный public input validation. `q=0`, отрицательные числа, NaN/Inf и отрицательный `T` не являются поддержанным контрактом, даже если отдельный вызов вернул какое-то число. `T=0`/`sigma=0` реализуются как intrinsic limit.

### 4.2. Нормированное ядро

```cpp
double normalised_black_call(double x, double s);
double normalised_black(double x, double s, double q /* q=±1 */);
```

Здесь

\[
x=\log(F/K),\qquad s=\sigma\sqrt T,
\]

а для call

\[
b(x,s)=e^{x/2}\Phi(x/s+s/2)
       -e^{-x/2}\Phi(x/s-s/2)
       =\frac{B_{call}}{\sqrt{FK}}.
\]

`normalised_black(x,s,q)` использует reciprocal-strike call/put equivalence; `q` снова только `±1`. Это лучший контракт для чистого kernel benchmark: он исключает rounding при построении `F`, discounting и денежное масштабирование.

### 4.3. Implied-vol functions

```cpp
double implied_volatility_from_a_transformed_rational_guess(
    double price, double F, double K, double T, double q);

double normalised_implied_volatility_from_a_transformed_rational_guess(
    double beta, double x, double q);
```

Для первой функции `price` должен быть **undiscounted price**. Она возвращает annualized `sigma`. Для второй:

\[
\beta=\frac{price}{\sqrt{FK}},\quad x=\log(F/K),
\]

а результат — total standard deviation \(s=\sigma\sqrt T\), а не annualized volatility. Стандартный solver использует максимум две итерации по умолчанию в показанной ревизии.

Недопустимая цена ниже intrinsic возвращает `-DBL_MAX`, цена не ниже теоретического maximum — `DBL_MAX`. Это sentinels, не физические значения volatility.

## 5. Переход от BSM spot-контракта и rounding

Пусть вход проекта — одинаковые binary64 числа `S, K, r, qd, T, sigma`. Тогда adapter должен один раз вычислить:

```text
D = exp(-r*T)
F = S*exp((r-qd)*T)
undiscounted_price = discounted_price / D
```

и передать `F,K,sigma,T,sign` в Jäckel. В implied-vol benchmark лучше вычислять `undiscounted_price = discounted_price * exp(r*T)`, если это ровно заранее выбранная последовательность; деление на уже округлённый `D` может дать соседний binary64.

Контракт `(F,K,...)` не побитово эквивалентен прямому BSM-контракту `(S,K,r,q_d,T,...)`:

1. `F = S*exp((r-q_d)T)` сначала округляется; затем Jäckel считает `log(F/K)`.
2. Прямой log-domain код может считать `log(S/K)+(r-q_d)T`, `log(S)-log(K)+(r-q_d)T` или `log1p((S-K)/K)+(r-q_d)T`. Эти формы математически равны, но дают разные binary64.
3. Jäckel масштабирует через `sqrt(F)*sqrt(K)`. Код, использующий `K*exp(x/2)`, `sqrt(F*K)` или spot/discount factors, округляется иначе; `sqrt(F*K)` дополнительно может преждевременно переполниться.
4. Умножение результата на `D` добавляет ещё одно округление. Если candidate строит discounted terms напрямую, различие включает adapter, а не только Black kernel.
5. При экстремальных carry `F` может overflow/underflow, хотя `log F=log S+(r-q_d)T` ещё конечен. Аналогично `F/K` может потерять диапазон до `log`, хотя `log F-log K` определён.

Поэтому нужны два разных теста:

### A. Kernel accuracy

Заморозить binary64 `x`, `s` и `sign`; сравнить `normalised_black(x,s,sign)` с candidate normalised kernel и high-precision oracle. Это отвечает на вопрос о самом численном ядре.

### B. End-to-end BSM accuracy

Заморозить decimal fixtures, явно определить преобразование decimal → binary64, затем дать обоим методам исходные `S,K,r,qd,T,sigma`. Зафиксировать каждое промежуточное binary64 (`D,F,x,s`). Это отвечает на вопрос о production pipeline.

Дополнительно полезен промежуточный **shared-forward test**: adapter один раз строит binary64 `F,D`, после чего оба метода получают одинаковые `F,K,sigma,T` и одинаковое финальное умножение на `D`. Нельзя смешивать результаты трёх контрактов в одной таблице «кто точнее».

## 6. Что сохранить при получении архива

До включения baseline в confirmation set нужно сохранить:

- URL и UTC-время получения;
- HTTP `Last-Modified`, `ETag`, `Content-Length`, если сервер их отдаёт;
- SHA-256 исходного `.7z`;
- точный listing архива с размерами и timestamps;
- SHA-256 каждого выбранного core-файла;
- полный license/provenance header;
- `version.h` и отличие даты/version от цитаты November 2013;
- compiler version, target architecture и полный набор flags;
- wrapper отдельно от upstream, с собственным SHA-256.

Если официальный домен остаётся недоступным, допустимый fallback — взять mirror snapshot по фиксированному Git commit, пометить его как **mirror baseline**, сохранить commit SHA и позднее сравнить каждый core-файл с официальным архивом. До этого не называть mirror «оригиналом Jäckel» и не утверждать его побитовую неизменность.

## 7. Проверенные ссылки

- Издатель и официальный source link: [Wiley — Let's Be Rational](https://onlinelibrary.wiley.com/doi/10.1002/wilm.10395).
- Официальный архив автора: [www.jaeckel.org/LetsBeRational.7z](https://www.jaeckel.org/LetsBeRational.7z) — ссылка подтверждена издателем, доступность архива 2026-09-26 не подтверждена.
- Прозрачное зеркало/оболочка: [vollib/lets_be_rational](https://github.com/vollib/lets_be_rational).
- Callable definitions: [LetsBeRational.cpp](https://github.com/vollib/lets_be_rational/blob/master/src/LetsBeRational.cpp).
- Exported signatures: [LetsBeRational.i](https://raw.githubusercontent.com/vollib/lets_be_rational/master/src/LetsBeRational.i).
- Version marker: [version.h](https://github.com/vollib/lets_be_rational/blob/master/src/version.h).
- Cody source provenance: [Netlib SPECFUN](https://www.netlib.org/specfun/) и mirror-файл [erf_cody.cpp](https://raw.githubusercontent.com/vollib/lets_be_rational/master/src/erf_cody.cpp).

## Итог решения

Технически upstream подходит как сильный однопоточный baseline: core мал, не требует сторонних библиотек и допускает отдельный CLI wrapper. Главный текущий пробел — не алгоритм, а provenance текущего официального архива: без его фактического listing и SHA-256 нельзя доказать, что зеркало соответствует последней официальной выдаче. После получения архива следует сначала зафиксировать байты и лицензию, затем собрать неизменный core одним `clang++` вызовом и вести kernel, shared-forward и end-to-end результаты раздельно.
