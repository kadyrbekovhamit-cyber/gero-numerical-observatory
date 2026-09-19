import java.util.Arrays;
import java.util.Locale;
import com.opengamma.strata.basics.value.ValueDerivatives;
import com.opengamma.strata.collect.array.DoubleArray;
import com.opengamma.strata.pricer.impl.option.SabrExtrapolationRightFunction;
import com.opengamma.strata.pricer.impl.volatility.smile.SabrFormulaData;
import com.opengamma.strata.pricer.impl.volatility.smile.VolatilityFunctionProvider;
import com.opengamma.strata.product.common.PutCall;

public class ExtrapolationProbe {
  // GERO test fixture through the library's public provider interface.
  // A constant function has exact zero first and second derivatives.
  static class ConstantVol extends VolatilityFunctionProvider<SabrFormulaData> {
    final double sigma;
    ConstantVol(double sigma) { this.sigma = sigma; }
    @Override public double volatility(double f, double k, double t, SabrFormulaData d) { return sigma; }
    @Override public ValueDerivatives volatilityAdjoint(double f, double k, double t, SabrFormulaData d) {
      return ValueDerivatives.of(sigma, DoubleArray.filled(6));
    }
    @Override public double volatilityAdjoint2(double f, double k, double t, SabrFormulaData d, double[] first, double[][] second) {
      Arrays.fill(first, 0d); for (double[] row : second) Arrays.fill(row, 0d); return sigma;
    }
  }
  static void run(String id, String provider, double f, double cutoff, double t, double sigma, double alpha) {
    SabrFormulaData data = SabrFormulaData.of(alpha, .5, -.25, .5);
    String prefix=id+","+provider+","+f+","+cutoff+","+t+","+sigma+","+alpha+",";
    try {
      SabrExtrapolationRightFunction fn = provider.equals("constant")
        ? SabrExtrapolationRightFunction.of(f, data, cutoff, t, 4d, new ConstantVol(sigma))
        : SabrExtrapolationRightFunction.of(f, t, data, cutoff, 4d);
      double below=f*.5, above=cutoff*1.5;
      double[] vals={fn.price(below,PutCall.CALL),fn.price(below,PutCall.PUT),
        fn.price(cutoff,PutCall.CALL),fn.price(above,PutCall.CALL),fn.price(above,PutCall.PUT),
        fn.priceDerivativeStrike(above,PutCall.CALL),fn.priceDerivativeForward(above,PutCall.CALL)};
      System.out.print(prefix+"OK,,");
      for (double v: vals) System.out.print(v+",");
      double[] params=fn.getParameter();
      System.out.println(params[0]+","+params[1]+","+params[2]);
    } catch (RuntimeException e) {
      String error=e.getClass().getName()+":"+String.valueOf(e.getMessage()).replace(',',';').replace('\n',' ');
      System.out.println(prefix+"ERROR,"+error+",,,,,,,,,,");
    }
  }
  public static void main(String[] args) {
    Locale.setDefault(Locale.ROOT);
    System.out.println("id,provider,F,cutoff,T,sigma,alpha,status,error,call_below,put_below,call_cutoff,call_above,put_above,strike_derivative_above,forward_derivative_above,a,b,c");
    int i=0;
    for(double f: new double[]{.03,.05,1d}) for(double t:new double[]{1d,5d}) for(double vol:new double[]{0d,1e-200,.2}) {
      run("c"+(i++),"constant",f,f*2,t,vol,.05);
    }
    for(double alpha:new double[]{0d,.05,.2}) run("d"+(i++),"default",.03,.1,1d,Double.NaN,alpha);
  }
}
