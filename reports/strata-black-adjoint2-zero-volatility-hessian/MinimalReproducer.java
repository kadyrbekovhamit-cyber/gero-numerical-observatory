import java.util.Arrays;
import com.opengamma.strata.pricer.impl.option.BlackFormulaRepository;
import com.opengamma.strata.pricer.impl.option.SabrExtrapolationRightFunction;
import com.opengamma.strata.pricer.impl.volatility.smile.SabrFormulaData;
import com.opengamma.strata.pricer.impl.volatility.smile.VolatilityFunctionProvider;
import com.opengamma.strata.product.common.PutCall;

public class MinimalReproducer {
  public static void main(String[] args) {
    var direct = BlackFormulaRepository.priceAdjoint2(110d, 100d, 1d, 0d, true);
    System.out.println("price=" + direct.getFirst().getValue());
    System.out.println("Hessian=" + Arrays.deepToString(direct.getSecond()));
    var constantZero = new VolatilityFunctionProvider<SabrFormulaData>() {
      @Override public double volatility(double f, double k, double t, SabrFormulaData data) {
        return 0d;
      }
      @Override public double volatilityAdjoint2(double f, double k, double t, SabrFormulaData data,
          double[] first, double[][] second) {
        Arrays.fill(first, 0d);
        for (double[] row : second) Arrays.fill(row, 0d);
        return 0d;
      }
    };
    try {
      var extrapolator = SabrExtrapolationRightFunction.of(
          .03, SabrFormulaData.of(.05, .5, -.25, .5), .06, 1d, 4d, constantZero);
      System.out.println("call_below_cutoff=" + extrapolator.price(.015, PutCall.CALL));
      System.out.println("put_above_cutoff=" + extrapolator.price(.09, PutCall.PUT));
    } catch (RuntimeException ex) {
      System.out.println(ex.getClass().getName() + ": " + ex.getMessage());
    }
  }
}
