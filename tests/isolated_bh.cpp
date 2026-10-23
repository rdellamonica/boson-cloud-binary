#include "CrankNicolsonModes.hpp"
#include "InitialData.hpp"
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void check_projection()
{
    const double mu=0.5, a0=4;
    Eigen::VectorXd r=Eigen::VectorXd::LinSpaced(257,0,40);
    std::vector<mode> modes;
    for (int l=0; l<=12; ++l)
        for (int m=-l; m<=l; ++m)
            if ((l-m)%2 == 0) modes.push_back({l,m});
    const auto centered=displaced_100_state(r,modes,mu,0);
    for (int i=1; i<r.size()-1; ++i)
        require(std::abs(centered[i]-2*r[i]*std::exp(-r[i]/a0)/std::pow(a0,1.5)) < 1e-13,
                "Centered 100 radial function disagrees with analytic hydrogenic state");
    require(centered.tail(centered.size()-r.size()).norm() == 0,
            "Centered spherical state has nonzero higher angular modes");
    require(std::abs(centered.squaredNorm()*(r[1]-r[0])-1) < 1e-5,
            "Continuum 100 normalization is incorrect");
    for (double offset : {-2.,2.}) {
        const auto state=displaced_100_state(r,modes,mu,offset);
        for (double x : {-5.,5.}) {
            const int i=32; // r=5
            dcomplex reconstructed=0;
            for (std::size_t j=0; j<modes.size(); ++j) {
                const auto m=modes[j];
                double y=gsl_sf_legendre_sphPlm(m.l,std::abs(m.m),0);
                if (m.m<0 && std::abs(m.m)%2) y=-y;
                if (x<0 && std::abs(m.m)%2) y=-y;
                reconstructed += state[j*r.size()+i]*y/r[i];
            }
            const double expected=std::exp(-std::abs(x-offset)/a0)/std::sqrt(M_PI*a0*a0*a0);
            require(std::abs(reconstructed-expected) < 1e-5*expected,
                    "Translated state is not centered at the requested signed offset");
        }
    }
}

void check_evolution(double offset, bool absorb, int threads)
{
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);
    CrankNicolsonModes cn(257,40,0.02,0,6,0.5,0.5,0,0,1,false,absorb,100,0,2,"fixture.toml");
    cn.configure_isolated_black_hole(offset);
    auto initial=displaced_100_state(cn.r,cn.modes,cn.mu,offset);
    initial /= std::sqrt(initial.squaredNorm()*cn.dr);
    cn.set_initial_state_full(initial);
    cn.initialize();
    require(cn.V2.norm()==0 && cn.dVdt.norm()==0,"Isolated run retains a companion or binary forcing");
    require(cn.M1==1 && cn.M2==0 && cn.q==0 && cn.Omega==0 && !cn.evolve_orbit,
            "Isolated BH configuration is not stationary and mass-1");
    // For any displacement d, the potential monopole is -mu/max(r,d).
    for (int i=1; i<cn.Nr-1; ++i)
        require(std::abs(cn.V.coeff(i,i)+cn.mu/std::max(cn.r[i],std::abs(offset))) < 1e-12,
                "Isolated potential monopole has the wrong mass or displacement");
    const SparseComplex h=cn.H;
    const double initial_norm=cn.psi.squaredNorm(), theta=cn.theta;
    cn.run(1,30);
    cn.t.resize(9);
    for (int step=1; step<=8; ++step) {
        const Eigen::VectorXcd previous=cn.psi;
        cn.step(step);
        const Eigen::VectorXcd midpoint=0.5*(previous+cn.psi);
        const double measured=previous.squaredNorm()-cn.psi.squaredNorm();
        const double predicted=cn.dt*midpoint.dot(cn.apply_sink(midpoint)).real();
        require(std::abs(measured-predicted) < 1e-9*initial_norm,
                "Isolated norm loss does not match the discrete CN absorption balance");
        require((cn.H-h).norm()<1e-13 && cn.theta==theta && cn.phase==0 && cn.Omega==0,
                "Isolated geometry or Hamiltonian changed with time");
        require(cn.R1==std::abs(offset) && cn.M1==1 && cn.M2==0,"BH moved or accreted in isolated mode");
    }
    const double loss=initial_norm-cn.psi.squaredNorm();
    if (absorb) require(loss>1e-4*initial_norm,"Isolated absorber was not active from t=0");
    else require(std::abs(loss)<1e-9*initial_norm,"Conservative isolated control lost norm");
}
}

int main()
{
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
        std::ofstream("fixture.toml") << "# isolated BH regression\n";
        check_projection();
        for (int threads : {1,4}) {
            check_evolution(0,true,threads);
            check_evolution(2,true,threads);
            check_evolution(-2,false,threads);
        }
        std::cout << "Isolated BH projection and absorption checks passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
