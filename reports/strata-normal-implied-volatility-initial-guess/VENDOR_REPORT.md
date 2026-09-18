### Reproducer (Strata 2.12.74)

`NormalFormulaRepository.impliedVolatility` returns zero for a positive ATM option price when the initial normal-volatility guess is zero:

```java
import com.opengamma.strata.pricer.impl.option.NormalFormulaRepository;
import com.opengamma.strata.product.common.PutCall;

public class Repro {
  public static void main(String[] args) {
    double iv = NormalFormulaRepository.impliedVolatility(
        1d, 100d, 100d, 1d, 0d, 1d, PutCall.CALL);
    System.out.println(iv); // observed: 0.0
    System.out.println(Math.sqrt(2 * Math.PI)); // 2.5066282746310002
    System.out.println(NormalFormulaRepository.price(100d, 100d, 1d, iv, PutCall.CALL));
    // observed repriced value: 0.0, versus input price 1.0
  }
}
```

The independent ATM Bachelier identity is `price = numeraire * sigma * sqrt(T) / sqrt(2*pi)`. Thus the requested implied volatility is positive and exactly `sqrt(2*pi)` in this example. The method Javadoc describes a nonzero initial guess being used for the search; it does not require callers to provide a strictly positive guess.

### Cause and relation to earlier work

At current main `987932ee95bf53e2baaff9a6b8e738a00f558b10`, the initializer is:

```java
double sigma = Math.min(Math.abs(initialNormalVol), 1e-10);
double maxChange = 0.5 * sigma;
```

A zero start gives a zero step bound and terminates without matching the supplied price. Small positive starts also return prematurely: with the same example, starts `1e-20` and `1e-16` return those starting values.

This initializer was introduced by #2372 while fixing the negative-forward issue #2238. I reviewed that history and its nonzero-start regression cases. This report concerns the remaining zero/tiny-start behavior with an ordinary positive forward; it is not a new report of the earlier negative-forward problem. The apparent intended change is `Math.max(Math.abs(initialNormalVol), 1e-10)`.

### Verification scope

I executed the official 2.12.74 report-tool distribution, then separately compiled the pinned current `NormalFormulaRepository` class against the same official dependency jar. This is not a full build of main.

- 1,728 independent analytical ATM scenarios: four forwards (including negative and zero), three positive maturities, three numeraires, three target volatilities, eight initial guesses, calls and puts.
- 18 zero-intrinsic controls and the four negative-forward scenarios from the existing regression test, reproduced as direct Java calls.
- 1,750 total observations: release/current/local one-line candidate/restored source failures **864 / 864 / 0 / 864**.
- The release, current and restored CSV outputs are byte-identical.
- The four existing negative-forward scenarios pass both before and after the candidate.

Java 25.0.4.1, macOS arm64, `-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Xint`. The official distribution SHA-256 was verified against the GitHub release asset digest: `42be9278993782e7b44c616a13c9bd1f34892ed05af32e854f107e28de3db4f3`.

The candidate has only the focused validation described above, not the full Strata suite or performance testing. These are synthetic public-library inputs; I have no evidence of production exposure or customer losses. I can provide the complete small Java grid and a focused regression patch if useful.

Research and report preparation were AI-assisted; the Java executions and outputs above were actually recorded. Submitted by Xamit Kadirbekov / GERO.
