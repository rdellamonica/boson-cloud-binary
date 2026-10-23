/**
 * @file CrankNicolsonModes_matrices.cpp
 * @brief Crank–Nicolson operators and the preconditioned BiCGSTAB solve.
 */
#include "CrankNicolsonModes.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void CrankNicolsonModes::build_CN_matrices()
{
    A = H;
    B = H;
    update_CN_values();

    solver.preconditioner().setBlockSize(Nr);
    solver.setTolerance(1e-10);
    solver.setMaxIterations(500);
    ilut_solver.preconditioner().setFillfactor(50);
    ilut_solver.preconditioner().setDroptol(1e-9);
    ilut_solver.setTolerance(1e-10);
    ilut_solver.setMaxIterations(5000);
}

void CrankNicolsonModes::update_CN_values()
{
    const dcomplex scale = I * (dt * 0.5);
#pragma omp parallel for schedule(static) if(n_modes >= 8)
    for (int row = 0; row < H.rows(); ++row)
    {
        for (Eigen::Index slot = H.outerIndexPtr()[row]; slot < H.outerIndexPtr()[row + 1]; ++slot)
        {
            const dcomplex diagonal = H.innerIndexPtr()[slot] == row ? 1.0 : 0.0;
            A.valuePtr()[slot] = diagonal + scale * H.valuePtr()[slot];
            B.valuePtr()[slot] = diagonal - scale * H.valuePtr()[slot];
        }
    }
}

namespace
{
// Right-preconditioned BiCGSTAB (same recurrence as Eigen's) for an operator
// given as a function. Used when the rank-one sinks are active, so the CN
// operator A + beta*v*v^dagger never has to be stored densely.
template <class ApplyA, class ApplyM>
bool bicgstab(const ApplyA &apply_A, const ApplyM &apply_M, const Eigen::VectorXcd &b,
              Eigen::VectorXcd &x, double tol, int max_iterations, int &iterations)
{
    const double b_norm2 = b.squaredNorm();
    const double threshold = tol * tol * b_norm2;
    Eigen::VectorXcd r = b - apply_A(x);
    Eigen::VectorXcd r0 = r;
    double r0_norm2 = r0.squaredNorm();
    const double restart_threshold = r0_norm2 * 1e-32;
    dcomplex rho = 1.0, alpha = 1.0, w = 1.0;
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(b.size()), p = v, y, z, s, t;
    iterations = 0;
    if (b_norm2 == 0.0)
    {
        x.setZero();
        return true;
    }
    while (r.squaredNorm() > threshold && iterations < max_iterations)
    {
        const dcomplex rho_old = rho;
        rho = r0.dot(r);
        if (std::abs(rho) < restart_threshold)
        {
            // Lost biorthogonality: restart from the current residual.
            r = b - apply_A(x);
            r0 = r;
            rho = r0_norm2 = r.squaredNorm();
        }
        const dcomplex beta = (rho / rho_old) * (alpha / w);
        p = r + beta * (p - w * v);
        y = apply_M(p);
        v = apply_A(y);
        alpha = rho / r0.dot(v);
        s = r - alpha * v;
        z = apply_M(s);
        t = apply_A(z);
        const double t_norm2 = t.squaredNorm();
        w = t_norm2 > 0.0 ? t.dot(s) / t_norm2 : dcomplex(0.0);
        x += alpha * y + w * z;
        r = s - w * t;
        ++iterations;
    }
    return r.squaredNorm() <= threshold;
}
} // namespace

void CrankNicolsonModes::solve_CN(const Eigen::VectorXcd &rhs)
{
    const double rhs_norm = rhs.norm();
    if (rhs_norm == 0.0)
    {
        psi.setZero();
        last_solver_residual = 0.0;
        last_solver_iterations = 0;
        return;
    }

    // Active rank-one sinks: A_full = A + (dt/4) * Gamma, Gamma = sum g_k v_k v_k^dagger.
    std::vector<const SwaveSink *> sinks;
    if (sink_active)
        for (const SwaveSink *sink : {&sink1, &sink2})
            if (sink->strength != 0.0 && sink->v.size() == rhs.size() && sink->v.squaredNorm() > 0.0)
                sinks.push_back(sink);
    const auto apply_A = [&](const Eigen::VectorXcd &x) {
        Eigen::VectorXcd y = A * x;
        for (const SwaveSink *sink : sinks)
            y += (0.25 * dt * sink->strength * sink->v.dot(x)) * sink->v;
        return y;
    };

    const auto residual = [&](const Eigen::VectorXcd &candidate) {
        return candidate.allFinite() ? (apply_A(candidate) - rhs).norm() / rhs_norm :
            std::numeric_limits<double>::infinity();
    };
    // Normalize the system so the stopping threshold is independent of cloud
    // amplitude; the acceptance check below always uses the physical RHS.
    const Eigen::VectorXcd normalized_rhs = rhs / rhs_norm;
    const Eigen::VectorXcd guess = psi / rhs_norm;
    Eigen::VectorXcd candidate;

    if (!sinks.empty())
    {
        // Woodbury: the preconditioner P (radial blocks or ILUT of the sparse A)
        // is corrected exactly for the low-rank sink, so it stays as effective
        // as without absorption.
        const auto krylov_with = [&](const auto &P_solve, int max_iterations) {
            const int k = static_cast<int>(sinks.size());
            Eigen::MatrixXcd W(rhs.size(), k);
            Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(k, k);
            for (int a = 0; a < k; ++a)
                W.col(a) = P_solve(sinks[a]->v);
            for (int a = 0; a < k; ++a)
            {
                G(a, a) += 1.0 / (0.25 * dt * sinks[a]->strength);
                for (int b = 0; b < k; ++b)
                    G(a, b) += sinks[a]->v.dot(W.col(b));
            }
            const Eigen::MatrixXcd G_inverse = G.inverse();
            const auto apply_M = [&](const Eigen::VectorXcd &x) {
                Eigen::VectorXcd y = P_solve(x);
                Eigen::VectorXcd projections(k);
                for (int a = 0; a < k; ++a)
                    projections[a] = sinks[a]->v.dot(y);
                y -= W * (G_inverse * projections);
                return y;
            };
            Eigen::VectorXcd x = guess;
            bicgstab(apply_A, apply_M, normalized_rhs, x, 1e-10, max_iterations, last_solver_iterations);
            return Eigen::VectorXcd(x * rhs_norm);
        };
        if (use_block_preconditioner)
        {
            RadialBlockPreconditioner &block = solver.preconditioner();
            block.compute(A);
            if (block.info() == Eigen::Success)
            {
                candidate = krylov_with([&](const Eigen::VectorXcd &x) {
                    return Eigen::VectorXcd(block.solve(x)); }, 500);
                last_solver_residual = residual(candidate);
                if (last_solver_residual <= 1e-9)
                {
                    psi.swap(candidate);
                    return;
                }
            }
            std::cerr << "\nRadial preconditioner did not converge; retrying with full ILUT.\n";
        }
        auto &ilut = ilut_solver.preconditioner();
        ilut.compute(A);
        if (ilut.info() != Eigen::Success)
            throw std::runtime_error("Crank-Nicolson ILUT factorization failed");
        candidate = krylov_with([&](const Eigen::VectorXcd &x) {
            return Eigen::VectorXcd(ilut.solve(x)); }, 5000);
        last_solver_residual = residual(candidate);
        if (!(last_solver_residual <= 1e-9))
            throw std::runtime_error("Crank-Nicolson solve failed the relative residual check");
        psi.swap(candidate);
        return;
    }

    if (use_block_preconditioner)
    {
        solver.compute(A);
        if (solver.info() == Eigen::Success)
        {
            candidate = solver.solveWithGuess(normalized_rhs, guess);
            candidate *= rhs_norm;
            last_solver_residual = residual(candidate);
            last_solver_iterations = static_cast<int>(solver.iterations());
            if (last_solver_residual <= 1e-9)
            {
                psi.swap(candidate);
                return;
            }
        }
        std::cerr << "\nRadial preconditioner did not converge; retrying with full ILUT.\n";
    }
    ilut_solver.compute(A);
    if (ilut_solver.info() != Eigen::Success)
        throw std::runtime_error("Crank-Nicolson ILUT factorization failed");
    candidate = ilut_solver.solveWithGuess(normalized_rhs, guess);
    candidate *= rhs_norm;
    last_solver_residual = residual(candidate);
    last_solver_iterations = static_cast<int>(ilut_solver.iterations());
    if (!(last_solver_residual <= 1e-9))
        throw std::runtime_error("Crank-Nicolson solve failed the relative residual check");
    psi.swap(candidate);
}
