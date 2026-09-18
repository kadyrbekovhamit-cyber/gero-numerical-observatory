import com.opengamma.strata.pricer.impl.option.NormalFormulaRepository;
import com.opengamma.strata.product.common.PutCall;
public class Repro {
 public static void main(String[] args) {
  double result = NormalFormulaRepository.impliedVolatility(1,100,100,1,0,1,PutCall.CALL);
  System.out.println("actual normal volatility: " + result);
  System.out.println("ATM analytical value: " + Math.sqrt(2*Math.PI));
  System.out.println("repriced option: " + NormalFormulaRepository.price(100,100,1,result,PutCall.CALL));
 }
}
