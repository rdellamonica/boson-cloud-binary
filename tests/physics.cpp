// Physical consistency of the evolution: conservation laws, known eigenstates,
// symmetries, orbital decay, accretion and the outer-boundary absorbers.
#include "CrankNicolsonModes.hpp"
#include "TestUtils.hpp"
#include <omp.h>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>

using test::close;
using test::require;

namespace {

struct Setup
{
    int Nr = 65;
    double L = 64.0, dt = 2.0, epsilon = 0.0;
    int l_max = 2;
    double mu = 0.2, M_c = 0.1, q = 0.5, R = 20.0;
    bool counter = false, absorption = false;
    double ramp_periods = 0.0, smear = 4.0;
};

std::unique_ptr<CrankNicolsonModes> make(const Setup &s)
{
    std::ofstream("fixture.toml") << "# physics fixture\n";
    return std::make_unique<CrankNicolsonModes>(s.Nr, s.L, s.dt, s.epsilon, s.l_max, s.mu, s.M_c, s.q, 0.0, s.R,
                                                s.counter, s.absorption, 1000, s.ramp_periods, s.smear,
                                                "fixture.toml");
}

int mode_index(const CrankNicolsonModes &cn, int l, int m)
{
    for (int j = 0; j < cn.n_modes; ++j)
        if (cn.modes[j].l == l && cn.modes[j].m == m) return j;
    return -1;
}

/// Smooth, complex, mode-dependent unit-norm state vanishing at both ends.
Eigen::VectorXcd generic_state(const CrankNicolsonModes &cn)
{
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn.Nr * cn.n_modes);
    for (int j = 0; j < cn.n_modes; ++j) {
        const auto [l, m] = cn.modes[j];
        const double width = 6.0 + 1.5 * l, centre = 18.0 + 2.0 * m;
        const dcomplex weight = std::polar(1.0 / (1.0 + l), 0.7 * m + 0.3 * l);
        for (int i = 1; i < cn.Nr - 1; ++i) {
            const double x = (cn.r[i] - centre) / width;
            psi[j * cn.Nr + i] = weight * std::exp(-x * x) * std::sin(M_PI * i / (cn.Nr - 1))
                                 * std::polar(1.0, 0.05 * cn.r[i]);
        }
    }
    return psi / std::sqrt(psi.squaredNorm() * cn.dr);
}

/// Mirror image y -> -y of a state: U'_{l,m} = (-1)^m U_{l,-m}.
Eigen::VectorXcd mirror(const CrankNicolsonModes &cn, const Eigen::VectorXcd &psi)
{
    Eigen::VectorXcd out(psi.size());
    for (int j = 0; j < cn.n_modes; ++j) {
        const int partner = mode_index(cn, cn.modes[j].l, -cn.modes[j].m);
        require(partner >= 0, "Mode list is not closed under m -> -m");
        const double sign = (cn.modes[j].m % 2 == 0) ? 1.0 : -1.0;
        out.segment(j * cn.Nr, cn.Nr) = sign * psi.segment(partner * cn.Nr, cn.Nr);
    }
    return out;
}

/// Mass mu * dr * sum |U|^2 of the field.
double cloud_mass(const CrankNicolsonModes &cn, const Eigen::VectorXcd &psi)
{
    return cn.mu * cn.dr * psi.squaredNorm();
}

void hamiltonian_is_hermitian()
{
    test::ScratchDir dir("hermitian");
    Setup s;
    s.l_max = 4;
    s.absorption = true;
    auto cn = make(s);
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    cn->t.resize(4);
    for (int step = 1; step <= 3; ++step) cn->step(step);
    // Dirichlet rows (r_0 and r_{N-1}) only pin the boundary values to zero, so
    // hermiticity is required on the interior nodes.
    Eigen::VectorXd interior = Eigen::VectorXd::Ones(cn->Nr * cn->n_modes);
    for (int j = 0; j < cn->n_modes; ++j) interior[j * cn->Nr] = interior[j * cn->Nr + cn->Nr - 1] = 0;
    const SparseComplex P = SparseComplex(interior.cast<dcomplex>().asDiagonal().toDenseMatrix().sparseView());
    const SparseComplex h = P * cn->H * P;
    require((h - SparseComplex(h.adjoint())).norm() < 1e-12 * h.norm(), "Conservative Hamiltonian is not Hermitian");
    require((cn->dVdt - SparseComplex(cn->dVdt.adjoint())).norm() < 1e-12 * (1 + cn->dVdt.norm()),
            "dV/dt is not Hermitian");
    // The sink is Hermitian and positive semi-definite.
    const Eigen::VectorXcd a = generic_state(*cn), b = mirror(*cn, a);
    const dcomplex ab = a.dot(cn->apply_sink(b)), ba = b.dot(cn->apply_sink(a));
    require(std::abs(ab - std::conj(ba)) < 1e-12 * (1 + std::abs(ab)), "Sink is not Hermitian");
    require(a.dot(cn->apply_sink(a)).real() >= 0 && b.dot(cn->apply_sink(b)).real() >= 0,
            "Sink is not positive semi-definite");
}

void unitarity_without_absorption()
{
    test::ScratchDir dir("unitarity");
    Setup s;
    s.l_max = 4;
    s.ramp_periods = 0.2;
    auto cn = make(s);
    cn->evolve_orbit = false;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    const double norm0 = cn->psi.squaredNorm();
    cn->t.resize(31);
    for (int step = 1; step <= 30; ++step) {
        cn->step(step);
        close(cn->psi.squaredNorm(), norm0, 1e-9, "Norm changed without absorption at step " + std::to_string(step));
    }
    require(cn->last_solver_residual <= 1e-9, "Solver residual above tolerance");
}

void hydrogen_eigenstates()
{
    test::ScratchDir dir("hydrogen");
    // Isolated BH at the origin: V = -mu / r, Bohr radius a0 = 1 / mu^2.
    for (const auto &[n, l] : {std::pair{1, 0}, std::pair{2, 1}}) {
        Setup s;
        s.Nr = 513;
        s.L = 80.0;
        s.mu = 0.5;
        s.l_max = 2;
        s.dt = 0.5;
        s.M_c = 0.5;
        auto cn = make(s);
        cn->configure_isolated_black_hole(0.0);
        const double a0 = 1.0 / (s.mu * s.mu);
        Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(cn->Nr * cn->n_modes);
        const int j = mode_index(*cn, l, l);
        for (int i = 0; i < cn->Nr; ++i) {
            const double r = cn->r[i];
            psi[j * cn->Nr + i] = n == 1 ? r * std::exp(-r / a0) : r * r * std::exp(-r / (2 * a0));
        }
        psi /= std::sqrt(psi.squaredNorm() * cn->dr);
        cn->set_initial_state_full(psi);
        cn->initialize();
        const Eigen::VectorXcd initial = cn->psi;
        const double norm = initial.squaredNorm();
        const double energy = initial.dot(cn->H * initial).real() / norm;
        const double exact = -std::pow(s.mu, 3) / (2.0 * n * n);
        close(energy, exact, 2e-3 * std::abs(exact) / std::max(1.0, std::abs(exact)),
              "Hydrogenic energy of n=" + std::to_string(n));

        cn->run(1, 60);
        const int steps = 40;
        cn->t.resize(steps + 1);
        for (int step = 1; step <= steps; ++step) cn->step(step);
        const dcomplex overlap = initial.dot(cn->psi) / norm;
        require(std::abs(overlap) > 1 - 1e-4, "Hydrogenic eigenstate is not stationary");
        // CN advances an eigenstate by the phase -2 atan(E dt / 2) per step.
        const double expected_phase = -2.0 * steps * std::atan(energy * s.dt / 2.0);
        close(std::remainder(std::arg(overlap) - expected_phase, 2 * M_PI), 0.0, 1e-3,
              "Eigenstate phase does not follow its energy");
        const double leaked = cn->psi.squaredNorm() - cn->psi.segment(j * cn->Nr, cn->Nr).squaredNorm();
        require(leaked < 1e-20 * norm, "Spherical potential coupled different (l, m) modes");
    }
}

/// Largest relative drift of the Jacobi constant <H> - Omega <L_z> over two
/// orbits on a fixed circular orbit, with @p per_period steps per period.
double jacobi_drift(bool counter, int per_period)
{
    // In the frame co-rotating with a fixed circular orbit the Hamiltonian is
    // time independent, so H - Omega L_z is conserved by the exact evolution.
    Setup s;
    s.l_max = 4;
    s.counter = counter;
    auto cn = make(s);
    cn->evolve_orbit = false;
    cn->dt = cn->P / per_period;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    auto expectation = [&](bool jacobi) {
        const Eigen::VectorXcd &psi = cn->psi;
        double lz = 0;
        for (int j = 0; j < cn->n_modes; ++j)
            lz += cn->modes[j].m * psi.segment(j * cn->Nr, cn->Nr).squaredNorm();
        const double energy = psi.dot(cn->H * psi).real();
        return (energy - (jacobi ? (counter ? -1.0 : 1.0) * cn->Omega * lz : 0.0)) * cn->dr;
    };
    const int steps = 2 * per_period;
    cn->t.resize(steps + 1);
    cn->step(1);
    const double j0 = expectation(true), e0 = expectation(false);
    double drift = 0, energy_change = 0;
    for (int step = 2; step <= steps; ++step) {
        cn->step(step);
        drift = std::max(drift, std::abs(expectation(true) - j0) / std::abs(j0));
        energy_change = std::max(energy_change, std::abs(expectation(false) - e0) / std::abs(e0));
    }
    // The energy alone is not conserved, so the Jacobi check is not vacuous.
    require(energy_change > 20 * drift, "Rotating potential did no work on the cloud");
    return drift;
}

void jacobi_constant_on_fixed_orbit()
{
    test::ScratchDir dir("jacobi");
    for (bool counter : {false, true}) {
        const double coarse = jacobi_drift(counter, 200), fine = jacobi_drift(counter, 800);
        require(coarse < 1e-3, "Jacobi constant drifts by " + std::to_string(coarse));
        // The residual drift is time-discretisation error and shrinks with dt.
        require(fine < coarse / 3, "Jacobi drift does not converge with the timestep");
    }
}

void co_counter_mirror_symmetry()
{
    test::ScratchDir dir("mirror");
    // Reflecting y -> -y maps a co-rotating binary onto a counter-rotating one.
    Setup s;
    s.l_max = 3;
    s.absorption = true;
    s.ramp_periods = 0.3;
    auto co = make(s);
    s.counter = true;
    auto counter = make(s);
    for (auto *cn : {co.get(), counter.get()}) cn->evolve_orbit = false;
    const Eigen::VectorXcd initial = generic_state(*co);
    co->set_initial_state_full(initial);
    counter->set_initial_state_full(mirror(*counter, initial));
    co->initialize();
    counter->initialize();
    co->run(1, 40);
    counter->run(1, 40);
    co->t.resize(41);
    counter->t.resize(41);
    for (int step = 1; step <= 40; ++step) {
        co->step(step);
        counter->step(step);
    }
    require(std::abs(co->theta + counter->theta) < 1e-12, "Orbits are not mirror images");
    const double error = (mirror(*co, co->psi) - counter->psi).norm() / co->psi.norm();
    require(error < 1e-7, "Counter-rotating run is not the mirror image of the co-rotating one ("
                              + std::to_string(error) + ")");
    require((co->psi - counter->psi).norm() > 1e-3 * co->psi.norm(), "Mirror test is vacuous");
}

void probability_flux_continuity()
{
    test::ScratchDir dir("continuity");
    // For CN, the change of any quadratic form equals dt times its rate on the
    // midpoint state. With the 3-point Laplacian, the mass inside node k (with
    // half weight on k) changes at minus the centred flux written to flux.txt,
    // which is the mass flux mu * j = Im(conj(U) dU/dr).
    Setup s;
    s.l_max = 4;
    s.ramp_periods = 0.0;
    auto cn = make(s);
    cn->evolve_orbit = false;
    cn->dt = cn->P / 64;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    const double r_ex = 30.0;
    const int k = static_cast<int>(std::floor(r_ex / cn->dr));
    cn->run(1, r_ex);
    auto inner_mass = [&](const Eigen::VectorXcd &psi) {
        double sum = 0;
        for (int j = 0; j < cn->n_modes; ++j) {
            sum += psi.segment(j * cn->Nr, k).squaredNorm() + 0.5 * std::norm(psi[j * cn->Nr + k]);
        }
        return cn->mu * cn->dr * sum;
    };
    auto flux = [&](const Eigen::VectorXcd &psi) {
        double total = 0;
        for (int j = 0; j < cn->n_modes; ++j) {
            const int c = j * cn->Nr + k;
            const dcomplex dU = (psi[c + 1] - psi[c - 1]) / (2 * cn->dr) - psi[c] / cn->r[k];
            total += (std::conj(psi[c]) * dU).imag();
        }
        return total;
    };
    cn->t.resize(41);
    double largest = 0;
    for (int step = 1; step <= 40; ++step) {
        const Eigen::VectorXcd previous = cn->psi;
        cn->step(step);
        const double change = inner_mass(cn->psi) - inner_mass(previous);
        const double predicted = -cn->dt * flux(0.5 * (previous + cn->psi));
        largest = std::max(largest, std::abs(predicted));
        require(std::abs(change - predicted) < 1e-8 * cloud_mass(*cn, previous),
                "Mass inside r_ex does not change at the rate set by the flux");
    }
    require(largest > 1e-6 * s.M_c, "Continuity test is vacuous: no flux crossed r_ex");
}

void gravitational_wave_inspiral()
{
    test::ScratchDir dir("inspiral");
    // A negligible cloud: the separation follows dR/dt = -(64/5) M1 M2 M / R^3,
    // switched on over one initial period, i.e. R^4 decreases linearly afterwards.
    Setup s;
    s.q = 1.0;
    s.R = 16.0;
    s.M_c = 1e-12;
    s.l_max = 0;
    auto cn = make(s);
    cn->evolve_orbit = true;
    cn->steps_per_period = 200;
    const double P0 = cn->P;
    cn->dt = P0 / 200;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    const int steps = 600;
    cn->t.resize(steps + 1);
    double R_ref = s.R, t_ref = 0.0;
    auto ramp = [&](double t) {
        const double x = std::clamp(t / P0, 0.0, 1.0);
        return 3 * x * x - 2 * x * x * x;
    };
    double R4_start = 0, t_start = 0;
    for (int step = 1; step <= steps; ++step) {
        cn->step(step);
        // Independent RK4 integration of the same law to the solver's time.
        const double t_end = cn->t[step];
        const int sub = 20;
        const double h = (t_end - t_ref) / sub;
        auto f = [&](double t, double R) { return -ramp(t) * (64.0 / 5.0) * 2.0 / (R * R * R); };
        for (int n = 0; n < sub; ++n, t_ref += h) {
            const double k1 = f(t_ref, R_ref), k2 = f(t_ref + h / 2, R_ref + h * k1 / 2);
            const double k3 = f(t_ref + h / 2, R_ref + h * k2 / 2), k4 = f(t_ref + h, R_ref + h * k3);
            R_ref += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        }
        t_ref = t_end;
        close(cn->R, R_ref, 2e-3, "Separation departs from the GW law at step " + std::to_string(step));
        close(cn->Omega, std::sqrt(2.0 / std::pow(cn->R, 3)), 1e-12, "Omega is not Keplerian");
        close(cn->R1, cn->R / 2, 1e-12, "R1 is not at the centre-of-mass position");
        close(cn->dt, cn->P / 200, 1e-12, "Post-ramp timestep does not follow the period");
        if (t_end > 1.5 * P0 && t_start == 0) {
            t_start = t_end;
            R4_start = std::pow(cn->R, 4);
        }
    }
    require(cn->R < s.R - 0.5, "Binary did not inspiral");
    const double expected_R4 = R4_start - (256.0 / 5.0) * 2.0 * (cn->t[steps] - t_start);
    close(std::pow(cn->R, 4), expected_R4, 5e-3, "R^4 does not decrease linearly (Peters)");
}

void accretion_balances_absorbed_mass()
{
    test::ScratchDir dir("accretion");
    // Mass removed from the field by the sinks is added to the black holes.
    Setup s;
    s.q = 1.0;
    s.R = 20.0;
    s.l_max = 2;
    s.M_c = 0.05;
    s.absorption = true;
    auto cn = make(s);
    cn->evolve_orbit = true;
    cn->steps_per_period = 16;
    cn->dt = cn->P / 16;
    cn->set_initial_state_full(generic_state(*cn));
    cn->initialize();
    cn->run(1, 40);
    const double total0 = cloud_mass(*cn, cn->psi) + cn->M1 + cn->M2;
    const double cloud0 = cloud_mass(*cn, cn->psi);
    cn->t.resize(41);
    for (int step = 1; step <= 40; ++step) {
        cn->step(step);
        close(cloud_mass(*cn, cn->psi) + cn->M1 + cn->M2, total0, 1e-10,
              "Field plus black-hole mass is not conserved at step " + std::to_string(step));
        close(cn->q, cn->M2 / cn->M1, 1e-14, "q does not track the masses");
    }
    require(cloud0 - cloud_mass(*cn, cn->psi) > 1e-3 * cloud0, "Accretion test is vacuous");
    require(cn->M1 > 1 && cn->M2 > 1, "Both black holes must accrete in an equal-mass binary");
}

void damping_mask_acts_outside_only()
{
    test::ScratchDir dir("mask");
    Setup s;
    auto plain = make(s);
    auto masked = make(s);
    for (auto *cn : {plain.get(), masked.get()}) {
        cn->evolve_orbit = false;
        cn->set_initial_state_full(generic_state(*cn));
        cn->initialize();
    }
    masked->build_damping_mask(0.75, 3.0, 2);
    plain->run(1, 30);
    masked->run(1, 30);
    plain->t.resize(2);
    masked->t.resize(2);
    plain->step(1);
    masked->step(1);
    const int edge = static_cast<int>(std::floor(0.75 * s.L / plain->dr));
    for (int j = 0; j < plain->n_modes; ++j)
        for (int i = 0; i < plain->Nr; ++i) {
            const int k = j * plain->Nr + i;
            const double factor = i < edge ? 1.0
                : std::exp(-3.0 * std::pow((plain->r[i] - 0.75 * s.L) / (0.25 * s.L), 2));
            require(std::abs(masked->psi[k] - factor * plain->psi[k]) < 1e-14 * (1 + std::abs(plain->psi[k])),
                    "Mask is not the documented profile applied after the step");
        }
    require(masked->psi.squaredNorm() < plain->psi.squaredNorm(), "Mask removed no norm");
}

void complex_absorbing_potential()
{
    test::ScratchDir dir("cap");
    Setup s;
    s.Nr = 129;
    s.L = 128;
    auto plain = make(s);
    auto absorbing = make(s);
    for (auto *cn : {plain.get(), absorbing.get()}) {
        cn->evolve_orbit = false;
        cn->set_initial_state_full(generic_state(*cn));
        cn->initialize();
    }
    absorbing->build_cap(0.8, 0.5, 2);
    plain->run(1, 40);
    absorbing->run(1, 40);
    const SparseComplex difference = absorbing->H - plain->H;
    const int edge = static_cast<int>(std::floor(0.8 * s.L / plain->dr));
    for (int k = 0; k < difference.outerSize(); ++k)
        for (SparseComplex::InnerIterator it(difference, k); it; ++it) {
            if (std::abs(it.value()) == 0) continue;
            require(it.row() == it.col(), "CAP is not diagonal");
            require(it.row() % s.Nr >= edge, "CAP acts inside its start radius");
            require(std::isfinite(it.value().imag()) && it.value().imag() <= 0 && it.value().real() == 0,
                    "CAP is not a finite negative imaginary potential");
        }
    require(difference.norm() > 0, "CAP is missing from H");
    absorbing->t.resize(21);
    for (int step = 1; step <= 20; ++step) {
        const Eigen::VectorXcd previous = absorbing->psi;
        absorbing->step(step);
        const Eigen::VectorXcd mid = 0.5 * (previous + absorbing->psi);
        // |psi_new|^2 - |psi|^2 = 2 dt <mid| Im(H) |mid> for CN.
        const double predicted = 2 * absorbing->dt * mid.dot(absorbing->CAP * mid).imag();
        const double change = absorbing->psi.squaredNorm() - previous.squaredNorm();
        require(change <= 1e-12 * previous.squaredNorm(), "CAP increased the norm");
        require(std::abs(change - predicted) < 1e-9 * previous.squaredNorm(), "CAP norm balance violated");
    }
}

} // namespace

int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    Eigen::setNbThreads(2);
    return test::run_all(std::pair{"Hamiltonian and sink are Hermitian", &hamiltonian_is_hermitian},
                         std::pair{"unitarity without absorption", &unitarity_without_absorption},
                         std::pair{"hydrogenic eigenstates", &hydrogen_eigenstates},
                         std::pair{"Jacobi constant on a fixed orbit", &jacobi_constant_on_fixed_orbit},
                         std::pair{"co/counter mirror symmetry", &co_counter_mirror_symmetry},
                         std::pair{"probability flux continuity", &probability_flux_continuity},
                         std::pair{"gravitational-wave inspiral", &gravitational_wave_inspiral},
                         std::pair{"accretion balances absorbed mass", &accretion_balances_absorbed_mass},
                         std::pair{"damping mask", &damping_mask_acts_outside_only},
                         std::pair{"complex absorbing potential", &complex_absorbing_potential});
}
