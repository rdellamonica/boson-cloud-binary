// The absorber must act only on the s-wave about the black hole: a hydrogenic
// 100 state loses norm, while a p-state centered on the same black hole does not.
#include "CrankNicolsonModes.hpp"
#include "InitialData.hpp"
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

constexpr double mu = 0.5, width = 2.0, length = 40.0;
constexpr int nr = 401, lmax = 8;

std::unique_ptr<CrankNicolsonModes> make_solver(double offset, double dt)
{
    auto cn = std::make_unique<CrankNicolsonModes>(nr, length, dt, 0, lmax, mu, mu, 0, 0, 1, false, true, 1000, 0, width, "fixture.toml");
    cn->configure_isolated_black_hole(offset);
    return cn;
}

Eigen::VectorXcd normalized(Eigen::VectorXcd state, double dr)
{
    return state / std::sqrt(state.squaredNorm() * dr);
}

// Instantaneous number-loss rate per unit probability, <psi|Gamma|psi>/<psi|psi>.
double loss_rate(const CrankNicolsonModes &cn, const Eigen::VectorXcd &state)
{
    return state.dot(cn.apply_sink(state)).real() / state.squaredNorm();
}

// A p-orbital centered on the BH: d/dx_BH of the displaced 100 state.
Eigen::VectorXcd p_state(const CrankNicolsonModes &cn, double offset)
{
    const double h = 1e-4;
    return displaced_100_state(cn.r, cn.modes, mu, offset + h) -
           displaced_100_state(cn.r, cn.modes, mu, offset - h);
}

void check_offset(double offset)
{
    auto solver = make_solver(offset, 0.05);
    CrankNicolsonModes &cn = *solver;
    const Eigen::VectorXcd s = normalized(displaced_100_state(cn.r, cn.modes, mu, offset), cn.dr);
    const Eigen::VectorXcd p = normalized(p_state(cn, offset), cn.dr);
    cn.set_initial_state_full(s);
    cn.initialize();
    require(cn.absorption_sink_active() && cn.sink_strength(1) > 0.0, "Isolated s-wave sink is not active");

    const double alpha = mu, analytic = 16 * mu * std::pow(alpha, 5);
    const double rate_s = loss_rate(cn, s), rate_p = loss_rate(cn, p);
    std::cout << std::setprecision(4) << "offset " << offset << ": s-wave rate/16 mu alpha^5 = "
              << rate_s / analytic << ", p-wave/s-wave rate = " << rate_p / rate_s << '\n';
    require(rate_s > 0.2 * analytic && rate_s < 1.5 * analytic,
            "s-wave absorption rate is not of the order of the analytic result");
    require(rate_p < 1e-6 * rate_s, "A p-state centered on the black hole is absorbed");

    // Evolve both states: the s-state decays, the p-state keeps its norm.
    cn.run(1, 30);
    cn.t.resize(41);
    for (int step = 1; step <= 40; ++step)
        cn.step(step);
    const double s_loss = 1.0 - cn.psi.squaredNorm() * cn.dr;

    auto control = make_solver(offset, 0.05);
    CrankNicolsonModes &cp = *control;
    cp.set_initial_state_full(p);
    cp.initialize();
    cp.run(1, 30);
    cp.t.resize(41);
    for (int step = 1; step <= 40; ++step)
        cp.step(step);
    const double p_loss = 1.0 - cp.psi.squaredNorm() * cp.dr;
    std::cout << "  norm lost in t=2: s-state " << s_loss << ", p-state " << p_loss << '\n';
    // The sink is strong for mu=0.5 (4*alpha*M/h = 1), so the evolved loss falls
    // below the first-order rate; it must still be a clear loss channel.
    require(s_loss > 0.2 * rate_s * 2.0, "s-state did not decay under evolution");
    require(std::abs(p_loss) < 1e-3 * s_loss, "p-state lost norm under evolution");
}
}

int main()
{
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
        std::ofstream("fixture.toml") << "# s-wave projector regression\n";
        // Offsets small compared with the support radius, as for the primary at small q.
        for (double offset : {0.0, 1.0})
            check_offset(offset);
        std::cout << "S-wave projector checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
