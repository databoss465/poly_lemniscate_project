#include <cmath>
#include <complex>
#include <vector>
#include <unordered_map>
#include <omp.h>

// g++ -O3 -fPIC -fopenmp -shared amr.cpp -o libamr.so

extern "C" double hybrid_amr_estimate(
    const double* roots_re, const double* roots_im, int degree,
    double x_min, double x_max, double y_min, double y_max, int initial_divs,
    double min_cell_size, int max_depth, int n_threads);

using Complex = std::complex<double>;
using CVector = std::vector<Complex>;

struct Point {
    double x, y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template<>
    struct hash<Point> {
        std::size_t operator()(const Point& p) const {
            auto h1 = std::hash<double>{}(p.x);
            auto h2 = std::hash<double>{}(p.y);
            return h1 ^ (h2 << 1);
        }
    };
}

using Memo = std::unordered_map<Point, bool>;

CVector build_coefficients(
    const double* roots_re, const double* roots_im, int degree) {
    CVector coeffs(1, 1.0);
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

bool is_inside(const CVector& coeffs, const Point& pt, Memo& memo) {
    auto it = memo.find(pt);
    if (it != memo.end()) {
        return it->second;
    }
    Complex z(pt.x, pt.y);
    bool result = std::abs(evaluate_polynomial_horner(coeffs, z)) < 1.0;
    memo[pt] = result;
    return result;
}

double recursive_refine(
    const CVector& coeffs, double x_min, double x_max, double y_min,
    double y_max, double min_cell_size, int depth, int max_depth, Memo& memo) {

    Point p1{x_min, y_min}, p2{x_max, y_min}, p3{x_min, y_max}, p4{x_max, y_max};
    Point center{(x_min + x_max) / 2.0, (y_min + y_max) / 2.0};
    double cell_size = std::max(x_max - x_min, y_max - y_min);

    bool b1 = is_inside(coeffs, p1, memo);
    bool b2 = is_inside(coeffs, p2, memo);
    bool b3 = is_inside(coeffs, p3, memo);
    bool b4 = is_inside(coeffs, p4, memo);
    bool b_center = is_inside(coeffs, center, memo);

    int in_count = b1 + b2 + b3 + b4 + b_center;
    double area = (x_max - x_min) * (y_max - y_min);

    if (in_count == 5) return area;
    if (in_count == 0 && depth > 0) return 0.0;

    if (depth >= max_depth || cell_size < min_cell_size) {
        return area * (static_cast<double>(in_count) / 5.0);
    }

    double x_mid = (x_min + x_max) / 2.0;
    double y_mid = (y_min + y_max) / 2.0;

    double area1 = recursive_refine(coeffs, x_min, x_mid, y_min, y_mid, min_cell_size, depth + 1, max_depth, memo);
    double area2 = recursive_refine(coeffs, x_mid, x_max, y_min, y_mid, min_cell_size, depth + 1, max_depth, memo);
    double area3 = recursive_refine(coeffs, x_min, x_mid, y_mid, y_max, min_cell_size, depth + 1, max_depth, memo);
    double area4 = recursive_refine(coeffs, x_mid, x_max, y_mid, y_max, min_cell_size, depth + 1, max_depth, memo);

    return area1 + area2 + area3 + area4;
}

extern "C" double hybrid_amr_estimate(
    const double* roots_re, const double* roots_im, int degree,
    double x_min, double x_max, double y_min, double y_max,
    int initial_divs, double min_cell_size, int max_depth, int n_threads) {

    omp_set_num_threads(n_threads);

    CVector coeffs = build_coefficients(roots_re, roots_im, degree);
    double total_area = 0.0;
    double dx = (x_max - x_min) / initial_divs;
    double dy = (y_max - y_min) / initial_divs;

    #pragma omp parallel for collapse(2) reduction(+:total_area)
    for (int i = 0; i < initial_divs; ++i) {
        for (int j = 0; j < initial_divs; ++j) {
            double x_start = x_min + i * dx;
            double x_end = x_min + (i + 1) * dx;
            double y_start = y_min + j * dy;
            double y_end = y_min + (j + 1) * dy;

            Memo memo;
            double local_area = recursive_refine(coeffs, x_start, x_end, y_start, y_end, min_cell_size, 0, max_depth, memo);
            total_area += local_area;
        }
    }

    return total_area;
}