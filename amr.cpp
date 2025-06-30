#include <cmath>
#include <complex>
#include <vector>

extern "C" double hybrid_amr_estimate(
    const double* roots_re, const double* roots_im, int degree,
    double x_min, double x_max, double y_min, double y_max, int initial_divs,
    double min_cell_size, int max_depth);


using Complex = std::complex<double>;
using CVector = std::vector<Complex>;

/*
We will not use this function. We use Horner's method instead.
Multiplying (z - r) is O(2*degree*n) but Horner's if )(d**2 + d*n)
Maybe later I will switch to using coeffs even in Python, maybe directly pass coeffs to this.

Complex evaluate_polynomial(
    const double* roots_re, const double* roots_im, int degree, Complex z) {
        Complex val(1.0, 0.0):
        for (int i = 0; i < degree; ++i) {
            Complex root(roots_re[i], roots_im[i]);
            val *= (z - root);
        }
        return val;
    }
*/

CVector build_coefficients(
    const double* roots_re, const double* roots_im, int degree) {
        CVector coeffs(1, 1.0); // Start with the constant term
        for (int i = 0; i < degree; ++i) {
            Complex root(roots_re[i], roots_im[i]);
            CVector new_coeffs(coeffs.size() + 1, 0.0);
            for (size_t j = 0; j < coeffs.size(); ++j) {
                new_coeffs[j] += -root * coeffs[j];
                new_coeffs[j + 1] += coeffs[j];
            }
            coeffs = std::move(new_coeffs);
        }
        return coeffs;
    }

Complex evaluate_polynomial_horner(const CVector& coeffs, Complex z) {
        Complex val(0.0, 0.0);
        for (int i = coeffs.size() - 1; i >= 0; --i) {
            val = val * z + coeffs[i];
        }
        return val;
    }

bool is_inside(const CVector& coeffs, Complex z) {
    return std::abs(evaluate_polynomial_horner(coeffs, z)) < 1.0;
}

double recursive_refine(
    const CVector& coeffs, double x_min, double x_max, double y_min, 
    double y_max, double min_cell_size, int depth, int max_depth) {
        Complex c1(x_min, y_min), c2(x_max, y_min), c3(x_min, y_max), c4(x_max, y_max);
        Complex center((x_min + x_max) / 2.0, (y_min + y_max) / 2.0);
        double cell_size = std::max(x_max - x_min, y_max - y_min);

        bool b1 = is_inside(coeffs, c1);
        bool b2 = is_inside(coeffs, c2);
        bool b3 = is_inside(coeffs, c3);
        bool b4 = is_inside(coeffs, c4);
        bool b_center = is_inside(coeffs, center);

        int in_count = b1 + b2 + b3 + b4 + b_center;
        double area = (x_max - x_min) * (y_max - y_min);

        if (in_count == 5) return area;
        if (in_count == 0 && depth > 0) return 0.0; //depth > 0 to account for pre-tiling

        if (depth >= max_depth || cell_size < min_cell_size) {
            return area * (static_cast<double>(in_count) / 5.0);
        }

        double x_mid = (x_min + x_max) / 2.0;
        double y_mid = (y_min + y_max) / 2.0;

        double area1 = recursive_refine(coeffs, x_min, x_mid, y_min, y_mid, min_cell_size, depth + 1, max_depth);
        double area2 = recursive_refine(coeffs, x_mid, x_max, y_min, y_mid, min_cell_size, depth + 1, max_depth);
        double area3 = recursive_refine(coeffs, x_min, x_mid, y_mid, y_max, min_cell_size, depth + 1, max_depth);
        double area4 = recursive_refine(coeffs, x_mid, x_max, y_mid, y_max, min_cell_size, depth + 1, max_depth);

        return area1 + area2 + area3 + area4;
    }


extern "C" double hybrid_amr_estimate(
    const double* roots_re, const double* roots_im, int degree,
    double x_min, double x_max, double y_min, double y_max,
    int initial_divs, double min_cell_size, int max_depth) {
    // Build polynomial coefficients
    CVector coeffs = build_coefficients(roots_re, roots_im, degree);

    // Initial area
    double total_area = 0.0;
    double dx = (x_max - x_min) / initial_divs;
    double dy = (y_max - y_min) / initial_divs;

    for (int i = 0; i < initial_divs; ++i) {
        for (int j = 0; j < initial_divs; ++j) {
            double x_start = x_min + i * dx;
            double x_end = x_min + (i + 1) * dx;
            double y_start = y_min + j * dy;
            double y_end = y_min + (j + 1) * dy;

            total_area += recursive_refine(coeffs, x_start, x_end, y_start, y_end, min_cell_size, 0, max_depth);
        }
    }

    return total_area;
}
    
