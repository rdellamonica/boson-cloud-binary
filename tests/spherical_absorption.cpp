#include "CrankNicolsonModes.hpp"
#include "SphericalAbsorptionReference.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
SparseComplex sink(const CrankNicolsonModes &cn)
{
    SparseComplex gamma=2.0*I*(cn.H-cn.H0-cn.V);
    // The conservative diagonal is real. Avoid subtracting its large kinetic
    // term to measure a small sink (especially at small r and large l).
    for(int row=0;row<gamma.rows();++row)
        for(SparseComplex::InnerIterator it(gamma,row);it;++it)
            if(it.col()==row) it.valueRef()=-2.0*cn.H.coeff(row,row).imag();
    return gamma;
}
Eigen::MatrixXcd block(const CrankNicolsonModes &cn, const SparseComplex &gamma, int i)
{
    Eigen::MatrixXcd result(cn.n_modes,cn.n_modes);
    for(int j=0;j<cn.n_modes;++j) for(int k=0;k<cn.n_modes;++k)
        result(j,k)=gamma.coeff(j*cn.Nr+i,k*cn.Nr+i);
    return result;
}
Eigen::MatrixXcd direct_angular_block(const CrankNicolsonModes &cn, int i, double width)
{
    // Independent integration in ordinary COM polar/azimuthal coordinates.
    // No Gaunt coefficients, Legendre moments, or angular-cap rotation.
    constexpr int nt=512,np=1024;
    auto *table=gsl_integration_glfixed_table_alloc(nt);
    Eigen::MatrixXcd result=Eigen::MatrixXcd::Zero(cn.n_modes,cn.n_modes);
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
            Eigen::VectorXcd harmonics(cn.n_modes);
            for(int j=0;j<cn.n_modes;++j) {
                const auto mode=cn.modes[j];
                double value=gsl_sf_legendre_sphPlm(mode.l,std::abs(mode.m),u);
                if(mode.m<0 && std::abs(mode.m)%2) value=-value;
                harmonics[j]=value*std::exp(I*double(mode.m)*phi);
            }
            result+=w*(2*M_PI/np)*std::exp(-1/(1-s2))*
                    (harmonics.conjugate()*harmonics.transpose());
        }
    }
    gsl_integration_glfixed_table_free(table);
    return result;
}
void verify(CrankNicolsonModes &cn, double width, bool direct)
{
    const auto gamma=sink(cn);
    const auto moments=primary_absorption_moments(cn,width/cn.M1);
    const double area=16*M_PI*cn.M1*cn.M1/10.0;
    double volume=0,cx=0,cy=0;
    int j11=-1;
    for(int j=0;j<cn.n_modes;++j) if(cn.modes[j].l==1 && cn.modes[j].m==1) j11=j;
    for(int i=1;i<cn.Nr-1;++i) {
        const auto b=block(cn,gamma,i);
        const double expected00=area*moments[i]/(4*M_PI);
        require(std::abs(b(0,0).real()-expected00)<1e-12*area,
                "Radial profile does not match independent distance-space integration");
        volume+=4*M_PI*cn.dr*cn.r[i]*cn.r[i]*b(0,0).real();
        const double factor=4*M_PI*std::sqrt(2.0/3.0)*cn.dr*std::pow(cn.r[i],3);
        cx-=factor*b(j11,0).real();
        cy+=factor*b(j11,0).imag();
        require((b-b.adjoint()).norm()<1e-12*std::max(1.0,b.norm()),"Absorption is not Hermitian");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig(b);
        require(eig.eigenvalues().minCoeff()>-1e-12*std::max(1.0,b.norm()),
                "Spherical sink is not positive semidefinite");
        if(cn.R1==0) {
            require((b-Eigen::MatrixXcd::Identity(cn.n_modes,cn.n_modes)*b(0,0)).norm()<1e-12*area,
                    "Central absorber has a preferred direction or mode dependence");
        }
        if(std::abs(cn.r[i]-cn.R1)>=width)
            require(b.norm()<1e-13,"Absorber has support outside its spherical radius");
    }
    require(std::abs(volume/area-1)<1e-12,"Discrete spherical kernel is not volume normalized");
    const double centroid_error=std::hypot(cx/volume-cn.R1*std::cos(cn.theta),
                                         cy/volume-cn.R1*std::sin(cn.theta));
    // dr=0.25, width>=3; quadrature/radial sampling leaves a small centroid error.
    require(centroid_error<2e-4,"Spherical absorber is displaced from the black hole");
    if(direct) {
        const int i=std::max(1,int(std::round(cn.R1/cn.dr)));
        const auto actual=block(cn,gamma,i);
        const auto reference=direct_angular_block(cn,i,width);
        const double scale=actual(0,0).real()/reference(0,0).real();
        const double relative_error=(actual-scale*reference).norm()/actual.norm();
        std::cout<<"Direct angular quadrature relative error="<<relative_error<<'\n';
        require(relative_error<2e-8,
                "Angular matrix disagrees with direct 3D spherical-kernel quadrature");
    }
    std::cout<<"Spherical geometry R1="<<cn.R1<<" volume/area="<<volume/area
             <<" centroid_error="<<centroid_error<<'\n';
}
void test_geometry(double radius, double width, bool counter, int threads, int lmax=3)
{
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);
    CrankNicolsonModes cn(129,32,1,0,lmax,.2,0,.01,0,90,counter,true,100,0,width,"fixture.toml");
    cn.evolve_orbit=false;
    cn.R1=radius;
    cn.theta=.73;
    cn.set_initial_state_full(Eigen::VectorXcd::Zero(cn.Nr*cn.n_modes));
    cn.initialize();
    cn.run(1,20);
    cn.t.resize(4);
    cn.dt=cn.P;
    cn.step(1);
    verify(cn,width,radius>0 && lmax<=3);
    const auto before=sink(cn);
    const double theta_before=cn.theta;
    cn.dt=cn.P/7;
    cn.step(2);
    const auto after=sink(cn);
    const double angle=cn.theta-theta_before;
    for(int j=0;j<cn.n_modes;++j) for(int k=0;k<cn.n_modes;++k) for(int i=1;i<cn.Nr-1;++i) {
        const auto expected=before.coeff(j*cn.Nr+i,k*cn.Nr+i)*
                            std::exp(I*double(cn.modes[k].m-cn.modes[j].m)*angle);
        require(std::abs(after.coeff(j*cn.Nr+i,k*cn.Nr+i)-expected)<1e-12,
                "Cached absorber did not rotate with the binary");
    }
    // Geometry and mass changes must invalidate cached normalized profiles.
    cn.R1+=.4;
    cn.M1=1.1;
    cn.step(3);
    verify(cn,width*cn.M1,false);
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
