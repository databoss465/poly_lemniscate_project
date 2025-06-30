#include <complex>
#include <random>
#include <vector>
#include <cmath>
#include <omp.h>

extern "C" double monte_carlo_estimate (
    const double* roots_re, const double* roots_im, int degree,
    double x_min, double x_max, double y_min, double y_max, int n_pts) {

        // Array of roots as complex numbers
        std::vector<std::complex<double>> roots(degree);
        for (int i = 0; i < degree; ++i) {
            roots.emplace_back(roots_re[i], roots_im[i]);
        }

        // Build polynomial coefficients
        std::vector<std::complex<double>> coeffs = {1.0};
        for (const auto& root : roots) {
            std::vector<std::complex<double>> new_coeffs(coeffs.size() + 1, 0.0);
            for (size_t j = 0; j < coeffs.size(); ++j) {
                new_coeffs[j] += -root * coeffs[j];
                new_coeffs[j + 1] += coeffs[j];
            }
            coeffs = std::move(new_coeffs);
        }

        int inside_points = 0;

        #pragma omp parallel 
        {
            // RNG
            std::mt19937 gen(465 + omp_get_thread_num()); 
            //Using thread number for seed ensures each thread geerates random nos. independently
            std::uniform_real_distribution<double> dist_x(x_min, x_max);
            std::uniform_real_distribution<double> dist_y(y_min, y_max);

            int local_count = 0;

            #pragma omp for
            for (int i = 0; i < n_pts; ++i) {
                std::complex<double> z(dist_x(gen), dist_y(gen));
                //This is called Horner's method it seems
                std::complex<double> val = 0.0;
                for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
                    val = val * z + *it;
                }
                if (std::norm(val) <= 1.0) 
                    ++local_count;

            }

        #pragma omp atomic
        inside_points += local_count;
        }
        double total_area = (x_max - x_min) * (y_max - y_min);
        return total_area * (static_cast<double>(inside_points) / n_pts);      
    }