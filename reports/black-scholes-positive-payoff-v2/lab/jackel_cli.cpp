// Local adapter only. The Jaeckel/Cody sources in vendor/jackel are unchanged.
#include <cfenv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

extern "C" double black(double F, double K, double sigma, double T, double sign);

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--environment") {
        volatile double smallest = std::numeric_limits<double>::denorm_min();
        volatile double twice = smallest + smallest;
        std::cout << "{\"nearest_even\":"
                  << (std::fegetround() == FE_TONEAREST ? "true" : "false")
                  << ",\"gradual_underflow_probe\":"
                  << (twice > smallest && smallest > 0 ? "true" : "false")
                  << "}\n";
        return 0;
    }
    double F, K, sigma, T;
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    while (std::cin >> F >> K >> sigma >> T) {
        if (!(std::isfinite(F) && std::isfinite(K) && std::isfinite(sigma)
              && std::isfinite(T) && F > 0 && K > 0 && sigma >= 0 && T >= 0)) {
            std::cerr << "Invalid Black input\n";
            return 2;
        }
        std::cout << black(F, K, sigma, T, 1.0) << ' '
                  << black(F, K, sigma, T, -1.0) << '\n';
    }
    return std::cin.eof() ? 0 : 3;
}
