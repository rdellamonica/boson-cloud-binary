#include "CrankNicolsonModes.hpp"
#include "SphericalAbsorptionReference.hpp"
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void test_preconditioner(int threads)
{
    omp_set_num_threads(threads);
    for (const int block_size : {1, 9}) {
        const int blocks = 12, size = block_size * blocks;
        Eigen::MatrixXcd dense = Eigen::MatrixXcd::Zero(size, size);
        for (int row = 0; row < size; ++row) {
            dense(row,row) = dcomplex(3.0 + row * 0.01, 0.2);
            if (row % block_size > 0) dense(row,row-1) = dcomplex(-0.2, 0.1);
            if (row % block_size + 1 < block_size) dense(row,row+1) = dcomplex(-0.4, 0.1);
        }
        SparseComplex matrix = dense.sparseView();
        // The preconditioner must ignore couplings between blocks.
        matrix.coeffRef(0, size-1) = 0.5;
        RadialBlockPreconditioner preconditioner;
        preconditioner.setBlockSize(block_size);
        preconditioner.compute(matrix);
        require(preconditioner.info() == Eigen::Success, "Block factorization failed");
        Eigen::MatrixXcd rhs(size, 2);
        for (int row = 0; row < size; ++row) {
            rhs(row,0) = dcomplex(std::sin(row), std::cos(row));
            rhs(row,1) = dcomplex(std::cos(row), -std::sin(row));
        }
        const Eigen::MatrixXcd result = preconditioner.solve(rhs);
        require((dense * result - rhs).norm() / rhs.norm() < 1e-13, "Incorrect radial block solve");
    }
    RadialBlockPreconditioner singular;
    singular.setBlockSize(1);
    SparseComplex zero(1,1);
    singular.compute(zero);
    require(singular.info() == Eigen::NumericalIssue, "Singular pivot was not detected");
}

// Independent serial triplet/reference formulas, without the optimized caches.
struct Reference {
    Eigen::MatrixXcd h0, v1, v2, dvdt, absorption;
};
Reference reference(const CrankNicolsonModes& cn, double time, double ramp_periods,
                    bool absorb, double smear)
{
    const int size = cn.Nr * cn.n_modes;
    Reference result;
    result.h0 = result.v1 = result.v2 = result.dvdt = result.absorption =
        Eigen::MatrixXcd::Zero(size, size);
    const auto absorption_moments = primary_absorption_moments(cn, smear);
    const double ramp_x = ramp_periods > 0 ? time / (ramp_periods * cn.P) : 0.0;
    const double ramp = time < ramp_periods * cn.P ? 3*ramp_x*ramp_x - 2*ramp_x*ramp_x*ramp_x : 1.0;
    double abs_ramp = 1.0;
    if (time < ramp_periods*cn.P) abs_ramp = 0.0;
    else if (time < (ramp_periods+1)*cn.P) {
        const double x = (time-ramp_periods*cn.P)/cn.P;
        abs_ramp = 3*x*x-2*x*x*x;
    }
    for (int j = 0; j < cn.n_modes; ++j) {
        const auto [l,m] = cn.modes[j];
        for (int i = 1; i < cn.Nr-1; ++i) {
            const int row = j*cn.Nr+i;
            result.h0(row,row) = (2/(cn.dr*cn.dr) + l*(l+1)/(cn.r[i]*cn.r[i]))/(2*cn.mu);
            result.h0(row,row-1) = result.h0(row,row+1) = -1/(2*cn.mu*cn.dr*cn.dr);
        }
        for (int k = 0; k < cn.n_modes; ++k) {
            const auto [l1,m1] = cn.modes[k];
            const int ms = m1-m;
            for (int ls = std::abs(l1-l); ls <= l1+l; ++ls) {
                if (std::abs(ms) > ls) continue;
                double y = gsl_sf_legendre_sphPlm(ls, std::abs(ms), 0.0);
                if (ms < 0 && std::abs(ms)%2) y *= -1;
                const double pref = (m1%2 == 0 ? 1.0 : -1.0) *
                    std::sqrt((2.0*l+1)*(2.0*l1+1)*(2.0*ls+1)/(4*M_PI));
                const dcomplex angular = pref * gsl_sf_coupling_3j(2*ls,2*l1,2*l,0,0,0) *
                    gsl_sf_coupling_3j(2*ls,2*l1,2*l,-2*ms,2*m1,-2*m) * y;
                const dcomplex phase = std::exp(I*double(ms)*cn.theta);
                for (int i = 1; i < cn.Nr-1; ++i) {
                    const int row = j*cn.Nr+i, col = k*cn.Nr+i;
                    const double ri = cn.r[i];
                    const auto radial = [&](double radius) {
                        return std::pow(std::min(ri,radius),ls)/std::pow(std::max(ri,radius),ls+1);
                    };
                    const dcomplex common = -cn.mu*(cn.M1+cn.M2)*4*M_PI/(2*ls+1)*angular*phase;
                    const dcomplex v1 = common/(1+cn.q)*radial(cn.R1);
                    const dcomplex v2 = common*(ms%2 == 0 ? 1.0 : -1.0)*cn.q/(1+cn.q)*radial(cn.R2);
                    result.v1(row,col) += v1;
                    result.v2(row,col) += v2;
                    result.dvdt(row,col) += (v1+ramp*v2)*(cn.counter ? -1.0 : 1.0)*I*double(ms)*cn.Omega;
                    if (absorb && (l+l1+ls)%2 == 0) {
                        // Preserve the user's area/10 and disabled secondary.
                        const double a1 = 16*M_PI*cn.M1*cn.M1*abs_ramp/10.0 * absorption_moments[ls*cn.Nr+i];
                        result.absorption(row,col) += a1*angular*phase;
                    }
                }
            }
        }
    }
    return result;
}

Eigen::VectorXcd test_evolution(int threads, int scenario, bool ilut)
{
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);
    constexpr int nr = 32;
    constexpr double dt = 0.2, smear = 10.0;
    const bool absorb = scenario == 1 || scenario == 2;
    const double initial_period = 2*M_PI/std::sqrt(1.2/std::pow(30.0,3));
    const double ramp_periods = scenario == 3 ? 2*dt/initial_period : 0.0;
    std::ofstream("fixture.toml") << "# correctness fixture\n";
    CrankNicolsonModes cn(nr, 120.0, dt, 0.1, 3, 0.2, scenario == 4 ? 1e-20 : 0.01, 0.2,
                         0.0, 30.0, scenario == 2, absorb, 10,
                         ramp_periods, smear, "fixture.toml");
    cn.evolve_orbit = absorb;
    cn.use_block_preconditioner = !ilut;
    cn.theta = 0.3;
    cn.Omega = cn.Omega_0;
    Eigen::VectorXcd initial(nr * cn.n_modes);
    for (int i = 0; i < initial.size(); ++i)
        initial[i] = i%nr == 0 || i%nr == nr-1 ? dcomplex(0) : std::polar(0.1+0.001*i,0.1*i);
    initial.normalize();
    cn.set_initial_state_full(initial);
    cn.initialize();
    const Reference first = reference(cn, 0.0, ramp_periods, absorb, smear);
    require((Eigen::MatrixXcd(cn.H0)-first.h0).norm() < 1e-12, "Stationary Hamiltonian mismatch");
    require((Eigen::MatrixXcd(cn.V1)-first.v1).norm() < 1e-12, "Initial potential mismatch");
    if (scenario == 2) {
        cn.build_cap(0.8, 0.01, 3);
        cn.build_damping_mask(0.8, 0.01, 3);
    }
    // Compute the independently assembled next Hamiltonian before the orbit
    // update changes R and masses at the end of the step.
    const double old_theta = cn.theta, old_omega = cn.Omega;
    cn.theta += (cn.counter ? -1.0 : 1.0)*cn.Omega*dt;
    if (dt < ramp_periods*cn.P) {
        const double x = dt/(ramp_periods*cn.P);
        cn.Omega = cn.Omega_0*(3*x*x-2*x*x*x);
    }
    const Reference next = reference(cn, dt, ramp_periods, absorb, smear);
    const double x = ramp_periods > 0 ? dt/(ramp_periods*initial_period) : 1.0;
    const double companion_ramp = x < 1 ? 3*x*x-2*x*x*x : 1.0;
    Eigen::MatrixXcd h = next.h0 + next.v1 + companion_ramp*next.v2;
    if (absorb) h -= 0.5*I*next.absorption;
    if (scenario == 2) h += Eigen::MatrixXcd(cn.CAP);
    const auto identity = Eigen::MatrixXcd::Identity(h.rows(),h.cols());
    const Eigen::MatrixXcd a = identity + I*(dt/2)*h;
    const Eigen::VectorXcd rhs = (identity - I*(dt/2)*h)*cn.psi;
    Eigen::VectorXcd expected = a.partialPivLu().solve(rhs);
    if (scenario == 2) {
        const int start = int(std::floor(cn.L*0.8/cn.dr));
        for (int m = 0; m < cn.n_modes; ++m)
            for (int i = start; i < nr; ++i)
                expected[m*nr+i] *= std::exp(-0.01*std::pow((cn.r[i]-cn.L*0.8)/(cn.L*0.2),3));
    }
    cn.theta = old_theta;
    cn.Omega = old_omega;
    cn.run(2, 60.0);
    require((Eigen::MatrixXcd(cn.V1)-next.v1).norm() < 1e-12, "V1 reference mismatch");
    require((Eigen::MatrixXcd(cn.V2)-next.v2).norm() < 1e-12, "V2 reference mismatch");
    require((Eigen::MatrixXcd(cn.dVdt)-next.dvdt).norm() < 1e-12, "Potential derivative mismatch");
    require((Eigen::MatrixXcd(cn.H)-h).norm() < 1e-12, "Full Hamiltonian reference mismatch");
    require((cn.psi-expected).norm()/expected.norm() < 1e-8, "CN differs from independent dense solve");
    require(cn.last_solver_residual <= 1e-9, "CN residual exceeded tolerance");
    return cn.psi;
}
}

int main()
{
    try {
        omp_set_dynamic(0);
        for (int threads : {1,4}) test_preconditioner(threads);
        for (int scenario = 0; scenario < 5; ++scenario) {
            const auto serial = test_evolution(1, scenario, false);
            const auto parallel = test_evolution(4, scenario, false);
            const auto ilut = test_evolution(1, scenario, true);
            require((serial-parallel).norm()/serial.norm() < 1e-13, "Thread count changed solution");
            require((serial-ilut).norm()/serial.norm() < 1e-8, "Radial and ILUT solutions differ");
        }
        std::cout << "All parallel correctness checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
