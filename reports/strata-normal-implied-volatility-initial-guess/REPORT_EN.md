# Strata returns zero implied volatility for a positive option price

In the official **OpenGamma Strata 2.12.74** release, a zero initial guess makes `NormalFormulaRepository.impliedVolatility` return **0** for an at-the-money option whose price is **1**. The independent normal-model result is **√(2π) ≈ 2.5066282746310002**.

Repricing with the returned zero volatility gives **0**, leaving an absolute price residual of **1** against the supplied price. A fresh execution of the local candidate returns **2.5066282746310002** and reprices to **1.0**, an observed residual of **0.0** in this example. Restoring the original source restores the zero result. This is a synthetic library calculation; no customer loss or production exposure was measured.

The finding was reported before publication in [OpenGamma/Strata issue #2796](https://github.com/OpenGamma/Strata/issues/2796). The correction below is a local candidate; no upstream acceptance is claimed.

## Minimal example

The seven arguments are option price, forward, strike, time to expiry, initial normal-volatility guess, numeraire and option type:

```java
import com.opengamma.strata.pricer.impl.option.NormalFormulaRepository;
import com.opengamma.strata.product.common.PutCall;

public class Repro {
  public static void main(String[] args) {
    double iv = NormalFormulaRepository.impliedVolatility(
        1d, 100d, 100d, 1d, 0d, 1d, PutCall.CALL);
    System.out.println(iv); // observed: 0.0
    System.out.println(Math.sqrt(2 * Math.PI)); // 2.5066282746310002
    System.out.println(
        NormalFormulaRepository.price(100d, 100d, 1d, iv, PutCall.CALL));
    // observed repriced value: 0.0
  }
}
```

At the money, the independent Bachelier identity is:

```text
price = numeraire × sigma × sqrt(T) / sqrt(2π)
sigma = price × sqrt(2π) / (numeraire × sqrt(T))
```

For the stated inputs, the analytical inverse has zero mathematical repricing residual. **Normal volatility is an absolute scale**, measured in the forward's units per square root of time. If `T` is in years, the result is about 2.5066 forward units per square-root year. It is not a 2.5066% Black implied volatility.

## Cause and earlier work

The [inspected initializer](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/option/NormalFormulaRepository.java#L282) is:

```java
double sigma = Math.min(Math.abs(initialNormalVol), 1e-10);
double maxChange = 0.5 * sigma;
```

A zero guess gives both a zero volatility and a zero step bound. The solver terminates without matching the supplied positive price. In the same minimal example, starting values `1e-20` and `1e-16` also return prematurely. The [method documentation](https://github.com/OpenGamma/Strata/blob/987932ee95bf53e2baaff9a6b8e738a00f558b10/modules/pricer/src/main/java/com/opengamma/strata/pricer/impl/option/NormalFormulaRepository.java#L251) does not impose a strictly positive initial-guess requirement.

This initializer dates to the correction in [PR #2372](https://github.com/OpenGamma/Strata/pull/2372) for the negative-forward problem in [issue #2238](https://github.com/OpenGamma/Strata/issues/2238). That prior work is explicitly credited. The present case concerns the remaining zero/tiny-start behavior, demonstrated even with a positive forward.

The local one-line candidate replaces `Math.min` with `Math.max`, establishing a positive starting floor. Its validation is limited to the tests below.

## Executed verification

The official released distribution was executed directly. The pinned current `NormalFormulaRepository` class was also compiled separately against the same official dependencies, followed by candidate and restored-original variants. This was **not a full build of Strata main**.

| Configuration | Failing observations out of 1,750 |
| --- | ---: |
| Official release 2.12.74 | 864 |
| Pinned current class | 864 |
| Local candidate | 0 |
| Original class restored | 864 |

The observations comprise:

- **1,728 analytical at-the-money scenarios:** four forwards, three positive maturities, three numeraires, three target volatilities, eight initial guesses, calls and puts.
- **18 zero-intrinsic controls.**
- **Four negative-forward scenarios transcribed from the existing upstream regression test**, executed as direct Java calls.

All **886 previously passing observations still pass** with the candidate. Of these, **450 actual-output strings are identical** and **436 change within the declared tolerance**. It would be inaccurate to describe all passing controls as unchanged.

The analytical scenarios use absolute tolerance `1e-9 × max(1, expected sigma)`; the four upstream-derived cases use `1e-8`; zero-intrinsic controls require exact zero.

The release, current and restored CSVs are byte-identical. A fresh run of the packaged portable reproducer matched **all four archived CSVs**, including the candidate, byte-for-byte on the tested platform. It also executed the minimal example in all four configurations: release/current/restored returned volatility 0.0 and repriced value 0.0; the candidate returned volatility 2.5066282746310002 and repriced value 1.0. Cross-platform bit identity is not promised. These are 864 failing observations of one initialization defect.

## Versions, evidence and limits

The tested current commit is `987932ee95bf53e2baaff9a6b8e738a00f558b10`. The refreshed source was unchanged and byte-identical to the release's corresponding file. At the 18 September 2026, 09:09 UTC review, [v2.12.74](https://github.com/OpenGamma/Strata/releases/tag/v2.12.74) remained the latest release.

The official report-tool ZIP matched the publisher's SHA-256:

```text
42be9278993782e7b44c616a13c9bd1f34892ed05af32e854f107e28de3db4f3
```

Execution used Temurin Java 25.0.4.1 on macOS arm64, with `-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Xint`. The archive contains the reproducer, grid, source, candidate patch, recorded observations, original license and verification receipts.

The duplicate review found the related #2238/#2372 history and the already-submitted #2796 report; no additional exact zero-start report appeared in the bounded searches. This is not proof that no private or unindexed report exists.

No full JUnit suite, complete non-at-the-money survey, performance study, actual bank deployment, production model or customer outcome was tested. Passing this finite grid does not establish correctness for every input. Research and report preparation were AI-assisted; actual Java executions produced the recorded outputs.
