/**
 * @file isolated_bh_main.cpp
 * @brief Command-line driver for one stationary black hole with a displaced 1s cloud.
 *
 * Run @c IsolatedBH --help for the options.
 */
#include "CrankNicolsonModes.hpp"
#include "InitialData.hpp"
#include <omp.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {
void usage()
{
    std::cout << "Usage: IsolatedBH mu x_bh smear_radius [l_max=6] [options]\n"
              << "One stationary mass-1 BH at (x_bh,0,0), with a displaced hydrogenic 100 state.\n"
              << "smear_radius is the physical support radius in units of M.\n"
              << "Options:\n"
              << "  --N INTEGER          Radial points (default 2049)\n"
              << "  --L REAL             Outer radius (default |x_bh|+20/mu^2+2*smear_radius)\n"
              << "  --dt REAL            Timestep (default min(1,0.05/mu^3,0.01*h^3))\n"
              << "  --steps INTEGER      Number of time samples, including t=0 (default 2001)\n"
              << "  --save-every INTEGER Snapshot interval (default 100)\n"
              << "  --threads INTEGER    OpenMP threads (default runtime environment)\n"
              << "  --no-absorption      Conservative control run\n"
              << "  --output DIRECTORY   New output directory (default unique IsolatedBH_...)\n";
}

double number(const std::string &text)
{
    std::size_t end;
    double value = std::stod(text,&end);
    if (end != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid finite number: "+text);
    return value;
}

int integer(const std::string &text)
{
    const double value=number(text);
    if (value != std::floor(value) || value < 0 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid nonnegative integer: "+text);
    return static_cast<int>(value);
}
}

int main(int argc, char **argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        if (argc < 4) { usage(); return 1; }
        const double mu=number(argv[1]), offset=number(argv[2]), width=number(argv[3]);
        if (mu <= 0 || width < 0) throw std::invalid_argument("Require mu > 0 and smear_radius >= 0");
        int lmax=6, nr=2049, steps=2001, save_every=100, threads=0, arg=4;
        bool absorption=true;
        double length=std::abs(offset)+20/(mu*mu)+2*width;
        double dt=std::min(1.0,0.05/(mu*mu*mu));
        if (width > 0) dt=std::min(dt,0.01*width*width*width);
        std::filesystem::path output;
        if (arg < argc && std::string(argv[arg]).rfind("--",0) != 0) lmax=integer(argv[arg++]);
        while (arg < argc) {
            const std::string option=argv[arg++];
            if (option == "--no-absorption") { absorption=false; continue; }
            if (arg == argc) throw std::invalid_argument("Missing value for "+option);
            const std::string value=argv[arg++];
            if (option == "--N") nr=integer(value);
            else if (option == "--L") length=number(value);
            else if (option == "--dt") dt=number(value);
            else if (option == "--steps") steps=integer(value);
            else if (option == "--save-every") save_every=integer(value);
            else if (option == "--threads") threads=integer(value);
            else if (option == "--output") output=value;
            else throw std::invalid_argument("Unknown option: "+option);
        }
        if (nr < 3 || steps < 1 || save_every < 1 || !std::isfinite(length) ||
            length <= std::abs(offset)+width || !std::isfinite(dt) || dt <= 0)
            throw std::invalid_argument("Require N>=3, steps/save-every>=1, dt>0 and L>|x_bh|+smear_radius");
        if (threads > 0) omp_set_num_threads(threads);
        Eigen::setNbThreads(omp_get_max_threads());
        if (output.empty()) output="IsolatedBH_"+std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count());
        output=std::filesystem::absolute(output);
        if (!std::filesystem::create_directories(output))
            throw std::runtime_error("Output directory already exists: "+output.string());
        std::filesystem::current_path(output);
        const double dr=length/(nr-1);
        std::ofstream config("isolated_config.toml");
        config << std::setprecision(17) << std::boolalpha
               << "[grid]\nN = " << nr << "\nL = " << length
               << "\n[binary]\nq = 0.0\nR = " << std::abs(offset) << "\ncounter = false"
               << "\n[atom]\nmu = " << mu << "\nn = 1\nl = 0\nm = 0\na = 0.0\nM_c = " << mu
               << "\n[simulation]\ndt = " << dt << "\nnum_steps = " << steps
               << "\nsave_every = " << save_every << "\nl_max = " << lmax
               << "\nr_ex = " << (nr-2)*dr << "\nramp_periods = 0.0\nepsilon = 0.0"
               << "\nevolve_orbit = false\nabsorption = " << absorption << "\nepsilon_smear = " << width
               << "\n[parallel]\nthreads = " << omp_get_max_threads()
               << "\n[isolated_bh]\nx = " << offset << "\ny = 0.0\nz = 0.0\nmass = 1.0\n"
               << "smear_radius = " << width << "\nstationary = true\n";
        config.close();
        if (!config) throw std::runtime_error("Cannot write isolated_config.toml");

        // The binary constructor's nominal R only initializes unused binary
        // timescales. Explicit isolated configuration replaces its geometry.
        CrankNicolsonModes solver(nr,length,dt,0,lmax,mu,mu,0,0,
            std::max(1.0,std::abs(offset)),false,absorption,save_every,0,width,"isolated_config.toml");
        solver.configure_isolated_black_hole(offset);
        Eigen::VectorXcd initial=displaced_100_state(solver.r,solver.modes,mu,offset);
        const double projected_norm=initial.squaredNorm()*dr;
        if (!std::isfinite(projected_norm) || projected_norm <= 0)
            throw std::runtime_error("Initial state has zero/nonfinite discrete norm");
        initial /= std::sqrt(projected_norm);
        solver.set_initial_state_full(initial); // M_c=mu gives unit probability norm.
        solver.initialize();
        std::cout << std::setprecision(12) << "Isolated BH at x=" << offset
                  << ", mu=" << mu << ", absorber radius=" << width << ", l_max=" << lmax
                  << "\nProjected 100-state norm before discrete normalization: " << projected_norm
                  << "\nGrid dr=" << dr << ", dt=" << dt << ", threads=" << omp_get_max_threads()
                  << "\nOutput root: " << output << '\n';
        if (absorption && width < dr)
            std::cout << "Absorber radius is below dr: the existing kernel is disabled.\n";
        solver.run(steps,(nr-2)*dr);
        std::cout << "Final full-domain probability norm: " << solver.psi.squaredNorm()*dr << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "IsolatedBH: " << error.what() << '\n';
        return 1;
    }
}
