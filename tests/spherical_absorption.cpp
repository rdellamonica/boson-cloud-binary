// Geometry of the spherical s-wave absorber: the production sink must equal an
// independent distance-space construction, be volume normalised, centred on the
// black hole, compactly supported, isotropic at the origin, rotate with the
// binary, and be rebuilt when the black hole moves or grows.
#include "CrankNicolsonModes.hpp"
#include "SphericalAbsorptionReference.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

int mode_index(const CrankNicolsonModes &cn, int l, int m)
{
    for (int j=0;j<cn.n_modes;++j) if (cn.modes[j].l==l && cn.modes[j].m==m) return j;
    return -1;
}

// Production projector of the primary, recovered from the sink alone:
// Gamma e_k = g v conj(v_k) with k on the (0,0) mode, where v_k is real and positive.
Eigen::VectorXcd production_projector(const CrankNicolsonModes &cn, double g)
{
    const int j00 = mode_index(cn,0,0);
    int best = -1;
    double best_value = 0;
    for (int i=1;i<cn.Nr-1;++i) {
        Eigen::VectorXcd e = Eigen::VectorXcd::Zero(cn.Nr*cn.n_modes);
        e[j00*cn.Nr+i] = 1;
        const double value = cn.apply_sink(e)[j00*cn.Nr+i].real();
        if (value > best_value) { best_value = value; best = i; }
    }
    require(best > 0, "Absorber has no s-wave support");
    Eigen::VectorXcd e = Eigen::VectorXcd::Zero(cn.Nr*cn.n_modes);
    e[j00*cn.Nr+best] = 1;
    return cn.apply_sink(e) / std::sqrt(g*best_value);
}

// int K Y_j^* dOmega at radius r_i, by direct quadrature in COM polar angles.
Eigen::VectorXcd direct_angular_projection(const CrankNicolsonModes &cn, int i, double width)
{
    constexpr int nt=512,np=1024;
    auto *table=gsl_integration_glfixed_table_alloc(nt);
    Eigen::VectorXcd result=Eigen::VectorXcd::Zero(cn.n_modes);
    for(int it=0;it<nt;++it) {
        double u,w;
        gsl_integration_glfixed_point(-1,1,it,&u,&w,table);
        for(int ip=0;ip<np;++ip) {
            const double phi=2*M_PI*ip/np;
            const double x=cn.r[i]*std::sqrt(1-u*u)*std::cos(phi)-cn.R1*std::cos(cn.theta);
            const double y=cn.r[i]*std::sqrt(1-u*u)*std::sin(phi)-cn.R1*std::sin(cn.theta);
            const double z=cn.r[i]*u;
            const double s2=(x*x+y*y+z*z)/(width*width);
            if(s2>=1) continue;
            for(int j=0;j<cn.n_modes;++j) {
                const auto mode=cn.modes[j];
                double value=gsl_sf_legendre_sphPlm(mode.l,std::abs(mode.m),u);
                if(mode.m<0 && std::abs(mode.m)%2) value=-value;
                result[j]+=w*(2*M_PI/np)*std::exp(-1/(1-s2))*value*std::exp(-I*double(mode.m)*phi);
            }
        }
    }
    gsl_integration_glfixed_table_free(table);
    return result;
}

Eigen::VectorXcd verify(CrankNicolsonModes &cn, double smear, bool direct)
{
    const double width = smear*cn.M1;
    const double g = 16*M_PI*cn.M1*cn.M1*cn.dr; // absorption ramp complete
    require(cn.absorption_sink_active(), "Absorber is not active");
    require(cn.sink_strength(2) == 0.0, "Fixture's unresolved secondary must not absorb");

    // 1. Whole operator against the independent construction.
    const Eigen::VectorXcd reference = swave_projector_reference(cn, smear, 1);
    std::mt19937 generator(7);
    std::normal_distribution<double> normal;
    for (int trial=0; trial<3; ++trial) {
        Eigen::VectorXcd x(cn.Nr*cn.n_modes);
        for (auto &value : x) value = dcomplex(normal(generator), normal(generator));
        const Eigen::VectorXcd expected = g*reference.dot(x)*reference;
        require((cn.apply_sink(x)-expected).norm() < 1e-10*expected.norm(),
                "Sink differs from the independent distance-space projector");
    }
    require(std::abs(cn.sink_strength(1) - g*reference.squaredNorm()) < 1e-10*cn.sink_strength(1),
            "sink_strength differs from g |v|^2");

    // 2. Geometry of the production kernel.
    const Eigen::VectorXcd v = production_projector(cn, g);
    const int j00 = mode_index(cn,0,0), j11 = mode_index(cn,1,1);
    double volume = 0;
    dcomplex centroid = 0;
    for (int i=1;i<cn.Nr-1;++i) {
        volume += std::sqrt(4*M_PI)*cn.dr*cn.r[i]*v[j00*cn.Nr+i].real();
        if (j11 >= 0)
            centroid -= std::sqrt(8*M_PI/3)*cn.dr*cn.r[i]*cn.r[i]*std::conj(v[j11*cn.Nr+i]);
        for (int j=0;j<cn.n_modes;++j) {
            if (std::abs(cn.r[i]-cn.R1) >= width)
                require(std::abs(v[j*cn.Nr+i]) < 1e-13, "Absorber has support outside its radius");
            if (cn.R1 == 0 && j != j00)
                require(std::abs(v[j*cn.Nr+i]) < 1e-13, "Central absorber is not isotropic");
        }
    }
    require(std::abs(volume-1) < 1e-12, "Discrete spherical kernel is not volume normalised");
    const double centroid_error = std::abs(centroid/volume - cn.R1*std::exp(I*cn.theta));
    require(centroid_error < 2e-4, "Spherical absorber is displaced from the black hole");

    // 3. Angular structure against direct 3D quadrature, near the black hole.
    if (direct) {
        const int i = std::max(1, int(std::round(cn.R1/cn.dr)));
        Eigen::VectorXcd actual(cn.n_modes);
        for (int j=0;j<cn.n_modes;++j) actual[j] = v[j*cn.Nr+i]/cn.r[i];
        const Eigen::VectorXcd quadrature = direct_angular_projection(cn, i, width);
        const double scale = actual[j00].real()/quadrature[j00].real();
        const double error = (actual-scale*quadrature).norm()/actual.norm();
        std::cout << "Direct angular quadrature relative error=" << error << '\n';
        require(error < 2e-8, "Angular projection disagrees with direct 3D quadrature");
    }
    std::cout << "Spherical geometry R1=" << cn.R1 << " volume=" << volume
              << " centroid_error=" << centroid_error << '\n';
    return v;
}

void test_geometry(double radius, double smear, bool counter, int threads, int lmax=3)
{
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);
    // q = 0.01: the secondary's absorber (0.08) is below dr = 0.25 and disabled.
    CrankNicolsonModes cn(129,32,1,0,lmax,.2,0,.01,0,90,counter,true,100,0,smear,"fixture.toml");
    cn.evolve_orbit=false;
    cn.R1=radius;
    cn.theta=.73;
    cn.set_initial_state_full(Eigen::VectorXcd::Zero(cn.Nr*cn.n_modes));
    cn.initialize();
    cn.run(1,20);
    cn.t.resize(4);
    cn.dt=cn.P; // past the one-period absorption ramp
    cn.step(1);
    const Eigen::VectorXcd before = verify(cn,smear,radius>0 && lmax<=3);
    const double theta_before=cn.theta;
    cn.dt=cn.P/7;
    cn.step(2);
    const Eigen::VectorXcd after = production_projector(cn, 16*M_PI*cn.M1*cn.M1*cn.dr);
    const double angle=cn.theta-theta_before;
    for(int j=0;j<cn.n_modes;++j) for(int i=1;i<cn.Nr-1;++i) {
        const dcomplex expected = before[j*cn.Nr+i]*std::exp(-I*double(cn.modes[j].m)*angle);
        require(std::abs(after[j*cn.Nr+i]-expected)<1e-12, "Absorber did not rotate with the binary");
    }
    // Geometry and mass changes must invalidate the cached profiles.
    cn.R1+=.4;
    cn.M1=1.1;
    cn.step(3);
    verify(cn,smear,false);
}
}

int main()
{
    try {
        omp_set_dynamic(0);
        std::ofstream("fixture.toml")<<"# spherical absorber regression\n";
        test_geometry(0,8,false,1);
        test_geometry(1e-10,8,true,1);
        test_geometry(90*.01/1.01,8,true,4);
        test_geometry(12,3,false,4);
        test_geometry(90*.01/1.01,8,true,4,12);
        std::cout<<"Spherical absorption checks passed.\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
