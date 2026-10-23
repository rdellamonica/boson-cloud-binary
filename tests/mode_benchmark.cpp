#include "CrankNicolsonModes.hpp"
#include <omp.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

// Run from a disposable directory; the solver writes its usual diagnostics there.
int main(int argc, char** argv)
{
    const int lmax = argc > 1 ? std::stoi(argv[1]) : 8;
    const int nr = argc > 2 ? std::stoi(argv[2]) : 256;
    const int steps = argc > 3 ? std::stoi(argv[3]) : 11;
    const int threads = argc > 4 ? std::stoi(argv[4]) : 1;
    const int scenario = argc > 5 ? std::stoi(argv[5]) : 0;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);
    std::ofstream("benchmark.toml") << "# Synthetic benchmark fixture\n";
    const auto start = std::chrono::steady_clock::now();
    CrankNicolsonModes cn(nr, 200.0, 0.25, 0.1, lmax, 0.2, 0.01,
                         0.1, 0.0, 30.0, scenario == 2, scenario != 0,
                         steps + 1, 0.0, 8.0, "benchmark.toml");
    cn.evolve_orbit = scenario != 0;
    cn.Omega = cn.Omega_0;
    Eigen::VectorXcd initial(nr * cn.n_modes);
    for (int m = 0; m < cn.n_modes; ++m)
        for (int i = 0; i < nr; ++i)
            initial[m * nr + i] = (i == 0 || i == nr - 1) ? dcomplex(0.0) :
                std::exp(-std::pow((cn.r[i] - 40.0) / 15.0, 2)) *
                std::polar(1.0 / (m + 1), 0.02 * i + 0.1 * m);
    initial.normalize();
    cn.set_initial_state_full(initial);
    cn.use_block_preconditioner = !(argc > 6 && std::string(argv[6]) == "ilut");
    cn.initialize();
    if (scenario == 2) {
        cn.build_cap(0.8, 0.01, 3);
        cn.build_damping_mask(0.8, 0.01, 3);
    }
    const auto ready = std::chrono::steady_clock::now();
    cn.run(steps, 100.0);
    const auto done = std::chrono::steady_clock::now();
    std::ofstream state("state.txt");
    state << std::setprecision(17);
    for (Eigen::Index i = 0; i < cn.psi.size(); ++i)
        state << cn.psi[i].real() << ' ' << cn.psi[i].imag() << '\n';
    std::ofstream matrices("matrices.txt");
    matrices << std::setprecision(17);
    for (const auto* mat : {&cn.H0, &cn.V1, &cn.V2, &cn.V, &cn.dVdt, &cn.H, &cn.A, &cn.B})
        for (int row = 0; row < mat->rows(); ++row)
            for (int col = 0; col < mat->cols(); ++col) {
                // Only dump small fixtures; do not distort the timed workload.
                if (lmax > 3) break;
                const auto v = mat->coeff(row, col);
                if (v != dcomplex(0.0)) matrices << row << ' ' << col << ' ' << v.real() << ' ' << v.imag() << '\n';
            }
    std::cout << "BENCHMARK modes=" << cn.n_modes << " nr=" << nr << " threads=" << threads
              << " init_seconds=" << std::chrono::duration<double>(ready-start).count()
              << " run_seconds=" << std::chrono::duration<double>(done-ready).count()
              << " residual=" << cn.last_solver_residual << " iterations=" << cn.last_solver_iterations << '\n';
    return cn.psi.allFinite() ? 0 : 1;
}
