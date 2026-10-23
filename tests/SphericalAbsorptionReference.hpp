#pragma once
#include "CrankNicolsonModes.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>
#include <algorithm>
#include <cmath>
#include <vector>

// Independent distance-space quadrature (production integrates angular caps).
inline std::vector<double> primary_absorption_moments(const CrankNicolsonModes &cn, double smear)
{
    std::vector<double> moments((2*cn.l_max+1)*cn.Nr, 0.0);
    const double width = smear*cn.M1, d = cn.R1;
    if (width < cn.dr || width <= 0) return moments;
    auto *table = gsl_integration_glfixed_table_alloc(256);
    for (int i=1; i<cn.Nr-1; ++i) {
        const double ri=cn.r[i], low=std::abs(ri-d), high=std::min(width,ri+d);
        if (d < 1e-8*width) { // Central limit avoids cancellation in distance endpoints.
            if (ri<width) moments[i]=4*M_PI*std::exp(-1/(1-ri*ri/(width*width)));
        } else if (low<high) {
            for (int k=0;k<256;++k) {
                double s,w;
                gsl_integration_glfixed_point(low,high,k,&s,&w,table);
                const double u=std::clamp((ri*ri+d*d-s*s)/(2*ri*d),-1.0,1.0);
                for (int l=0;l<=2*cn.l_max;++l)
                    moments[l*cn.Nr+i]+=2*M_PI/(ri*d)*w*s*std::exp(-1/(1-s*s/(width*width)))*
                                             gsl_sf_legendre_Pl(l,u);
            }
        }
    }
    gsl_integration_glfixed_table_free(table);
    double norm=0;
    for(int i=1;i<cn.Nr-1;++i) norm+=cn.dr*cn.r[i]*cn.r[i]*moments[i];
    if(norm>0) for(double &v:moments) v/=norm;
    return moments;
}
