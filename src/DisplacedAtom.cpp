/**
 * @file DisplacedAtom.cpp
 * @brief Mode projection of a hydrogenic 1s state centred off the origin.
 */
#include "InitialData.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <algorithm>
#include <cmath>
#include <memory>

Eigen::VectorXcd displaced_100_state(const Eigen::VectorXd &r,
    const std::vector<mode> &modes, double mu, double x_bh)
{
    if (!std::isfinite(mu) || mu <= 0 || !std::isfinite(x_bh) || r.size() < 3 || modes.empty())
        throw std::invalid_argument("Invalid displaced 100 state parameters");
    if (!r.allFinite() || r[0] < 0)
        throw std::invalid_argument("Radial coordinates must be finite and nonnegative");
    for (int i=1; i<r.size(); ++i)
        if (r[i] <= r[i-1]) throw std::invalid_argument("Radial coordinates must increase");
    int lmax = 0;
    for (const auto &mode : modes) {
        if (mode.l < 0 || std::abs(mode.m) > mode.l)
            throw std::invalid_argument("Invalid spherical harmonic mode");
        lmax = std::max(lmax, mode.l);
    }
    const double a0 = 1.0/(mu*mu), distance = std::abs(x_bh);
    const double amplitude = 1.0/std::sqrt(M_PI*a0*a0*a0);
    const int nq = std::max(128, 2*lmax+32);
    std::unique_ptr<gsl_integration_glfixed_table, decltype(&gsl_integration_glfixed_table_free)>
        table(gsl_integration_glfixed_table_alloc(nq), gsl_integration_glfixed_table_free);
    if (!table) throw std::bad_alloc();
    std::vector<double> nodes(nq), weights(nq);
    for (int k=0; k<nq; ++k)
        gsl_integration_glfixed_point(0,1,k,&nodes[k],&weights[k],table.get());
    std::vector<dcomplex> direction(modes.size());
    for (std::size_t j=0; j<modes.size(); ++j) {
        const auto &mode = modes[j];
        double y = gsl_sf_legendre_sphPlm(mode.l,std::abs(mode.m),0.0);
        if (mode.m < 0 && std::abs(mode.m)%2) y = -y;
        // Y_lm^*(pi/2, phi_BH), phi_BH=0 or pi. Use exact real parity.
        if (x_bh < 0 && std::abs(mode.m)%2) y = -y;
        direction[j] = y;
    }
    Eigen::VectorXcd state = Eigen::VectorXcd::Zero(r.size()*modes.size());
#pragma omp parallel for schedule(static) if (r.size() >= 128)
    for (int i=1; i<r.size()-1; ++i) {
        std::vector<double> moments(lmax+1,0.0);
        if (distance == 0.0) {
            moments[0] = 4*M_PI*amplitude*std::exp(-r[i]/a0);
        } else {
            // Integrate in body-centered distance s; ds gives dcos(gamma)=s ds/(r*d).
            // The discarded exponential tail is below exp(-40) relative to its peak.
            const double low = std::abs(r[i]-distance);
            const double high = std::min(r[i]+distance, low+40*a0);
            for (int k=0; k<nq; ++k) {
                const double s = low+(high-low)*nodes[k];
                const double u = std::clamp((r[i]*r[i]+distance*distance-s*s)/(2*r[i]*distance),-1.0,1.0);
                const double weight = 2*M_PI*(high-low)*weights[k]*s/(r[i]*distance)
                    * amplitude*std::exp(-s/a0);
                double previous=1.0, current=u;
                moments[0] += weight;
                if (lmax > 0) moments[1] += weight*current;
                for (int l=2; l<=lmax; ++l) {
                    const double next=((2*l-1)*u*current-(l-1)*previous)/l;
                    moments[l] += weight*next;
                    previous=current;
                    current=next;
                }
            }
        }
        for (std::size_t j=0; j<modes.size(); ++j)
            state[j*r.size()+i] = r[i]*moments[modes[j].l]*direction[j];
    }
    return state;
}
