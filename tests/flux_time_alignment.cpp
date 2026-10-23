#include "CrankNicolsonModes.hpp"
#include <omp.h>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, const char* message)
{
    require(std::abs(actual-expected) <= 2e-11*std::max(1e-10, std::abs(expected)), message);
}

double flux_at(const CrankNicolsonModes& cn, const Eigen::VectorXcd& state,
               const Eigen::VectorXcd& derivative, int mode)
{
    const int k = mode*cn.Nr + static_cast<int>(std::floor(42.5/cn.dr));
    return (std::conj(derivative[k])*(state[k+1]-state[k-1])/(2*cn.dr)).real()/cn.mu;
}

void check_endpoints(int threads, bool mask)
{
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);
    namespace fs = std::filesystem;
    const auto original_directory = fs::current_path();
    const auto work = original_directory / ("endpoint_" + std::to_string(threads) + "_"
        + std::to_string(mask) + "_"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(work);
    fs::current_path(work);
    std::ofstream("fixture.toml") << "# flux endpoint regression\n";
    CrankNicolsonModes cn(129,64,2,0.1,3,0.2,0.1,0.2,0,12,false,true,
                         1,0.02,4,"fixture.toml");
    cn.evolve_orbit = false;
    Eigen::VectorXcd initial(cn.Nr*cn.n_modes);
    for (int m=0; m<cn.n_modes; ++m)
        for (int i=0; i<cn.Nr; ++i)
            initial[m*cn.Nr+i] = std::polar(
                std::sin(M_PI*i/(cn.Nr-1))/(1+m), 0.15*i+0.3*m);
    initial.normalize();
    cn.set_initial_state_full(initial);
    cn.initialize();
    // In production the CAP is enabled after initialize(), before run().
    cn.build_cap(0.75,0.02,3);
    if (mask) cn.build_damping_mask(0.3,3,2);
    cn.run(1,42.5);
    cn.t.resize(5);
    fs::path output;
    for (const auto& entry : fs::directory_iterator(work))
        if (entry.is_directory()) output = entry.path();
    require(!output.empty(),"Missing simulation output directory");

    Eigen::VectorXcd saved_initial_derivative(initial.size());
    std::ifstream snapshot(output/"psi_dt/snapshot_000000.txt");
    for (int i=0; i<initial.size(); ++i) {
        double re, im;
        require(bool(snapshot >> re >> im), "Missing initial derivative snapshot");
        saved_initial_derivative[i] = dcomplex(re,im);
    }
    const Eigen::VectorXcd expected_initial = -I*(cn.H*cn.psi);
    require((saved_initial_derivative-expected_initial).norm() < 5e-6*expected_initial.norm(),
            "Initial derivative does not match the initialized endpoint (including CAP)");
    require((cn.H-(cn.H0+cn.V+cn.CAP)).norm() < 1e-12,
            "Initial Hamiltonian omitted a CAP enabled after initialize");

    bool detects_previous_derivative = false, detects_pre_mask_derivative = false;
    for (int step=1; step<=4; ++step) {
        // Endpoint pairing must not depend on a constant diagnostic timestep.
        cn.dt = step%2 == 0 ? 3.1 : 2.0;
        const Eigen::VectorXcd previous_derivative = -I*(cn.H*cn.psi);
        cn.step(step);
        const Eigen::VectorXcd derivative = -I*(cn.H*cn.psi);
        const Eigen::VectorXcd potential_state = cn.V*cn.psi;
        dcomplex expected_dv = 0, stale_dv = 0;
        for (int m=0; m<cn.n_modes; ++m)
            for (int i=0; i<=static_cast<int>(std::floor(42.5/cn.dr)); ++i) {
                const int j=m*cn.Nr+i;
                expected_dv += std::conj(derivative[j])*potential_state[j];
                stale_dv += std::conj(previous_derivative[j])*potential_state[j];
            }
        detects_previous_derivative |= std::abs(expected_dv.real()-stale_dv.real()) > 1e-8;

        std::ifstream flux_file(output/"energy_flux.txt");
        std::ifstream dv_file(output/"dV_bin_dt.txt");
        std::ifstream time_file(output/"t.txt");
        std::vector<double> flux(cn.n_modes);
        double dv=0, time=0;
        for (int row=0; row<step; ++row) {
            for (auto& value : flux) require(bool(flux_file >> value), "Missing flux row");
            require(bool(dv_file >> dv) && bool(time_file >> time), "Missing diagnostic row");
        }
        close(time,cn.t[step],"Diagnostic timestamp differs from state timestamp");
        close(dv,expected_dv.real()*cn.dr,"Potential diagnostic used the previous derivative");
        for (int m=0; m<cn.n_modes; ++m)
            close(flux[m],flux_at(cn,cn.psi,derivative,m),"Saved flux mixes endpoint states");
        if (mask) {
            Eigen::VectorXcd before_mask = cn.psi;
            for (int m=0; m<cn.n_modes; ++m)
                for (int i=static_cast<int>(std::floor(0.3*cn.L/cn.dr)); i<cn.Nr; ++i)
                    before_mask[m*cn.Nr+i] /= std::exp(-3*std::pow((cn.r[i]-0.3*cn.L)/(0.7*cn.L),2));
            const Eigen::VectorXcd bad_derivative = -I*(cn.H*before_mask);
            detects_pre_mask_derivative |= std::abs(flux_at(cn,cn.psi,bad_derivative,0)-flux[0]) > 1e-8;
        }
    }
    require(detects_previous_derivative,"Fixture must detect the old-time derivative bug");
    require(!mask || detects_pre_mask_derivative,"Fixture must detect the pre-mask derivative bug");
    fs::current_path(original_directory);
}
}

int main()
{
    try {
        omp_set_dynamic(0);
        for (int threads : {1,4})
            for (bool mask : {false,true}) check_endpoints(threads,mask);
        std::cout << "Flux endpoint alignment checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
