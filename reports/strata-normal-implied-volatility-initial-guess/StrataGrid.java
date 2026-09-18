import java.util.Locale;
import com.opengamma.strata.pricer.impl.option.NormalFormulaRepository;
import com.opengamma.strata.product.common.PutCall;
public class StrataGrid {
 static int id=0;
 static void check(String group,double price,double f,double k,double t,double start,double n,PutCall pc,double expected,double tol) {
  double actual; String error="";
  try { actual=NormalFormulaRepository.impliedVolatility(price,f,k,t,start,n,pc); } catch(Exception e){actual=Double.NaN;error=e.getClass().getSimpleName();}
  boolean ok=Double.isFinite(actual)&&Math.abs(actual-expected)<=tol;
  System.out.printf(Locale.ROOT,"%d,%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%s%n",id++,group,pc,price,f,k,t,start,n,expected,actual,tol,ok,error);
 }
 public static void main(String[]args){
  System.out.println("id,group,putcall,price,forward,strike,time,start,numeraire,expected,actual,tolerance,pass,error");
  for(double f:new double[]{-100,-0.005,0,100})for(double t:new double[]{0.25,1,4})for(double n:new double[]{0.5,1,2})for(double sigma:new double[]{0.01,1,5})for(double start:new double[]{0,1e-20,1e-16,1e-15,1e-10,0.1,2.5,-0.1})for(PutCall pc:PutCall.values()){
   double price=n*sigma*Math.sqrt(t)/Math.sqrt(2*Math.PI);
   check("ATM_analytic",price,f,f,t,start,n,pc,sigma,1e-9*Math.max(1,sigma));
  }
  double[]k={0.015,-0.01,-0.005,0},s={0.0075,0.01,0.005,0.02},v={0.01,0.001,0.0049,0.04},t={1,5,0.25,1};PutCall[]pc={PutCall.PUT,PutCall.CALL,PutCall.CALL,PutCall.PUT};
  for(int i=0;i<4;i++){double price=NormalFormulaRepository.price(-0.005,k[i],t[i],s[i],pc[i]);check("upstream_negative_forward",price,-0.005,k[i],t[i],v[i],1,pc[i],s[i],1e-8);}
  for(double f:new double[]{-100,0,100})for(double start:new double[]{0,1e-20,0.1})for(PutCall opt:PutCall.values())check("intrinsic_control",0,f,f,1,start,1,opt,0,0);
 }
}
