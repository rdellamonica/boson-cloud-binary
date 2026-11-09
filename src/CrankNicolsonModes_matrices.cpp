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
    block_preconditioner.setBlockSize(Nr);
    ilut_preconditioner.setFillfactor(50);
    ilut_preconditioner.setDroptol(1e-9);
}

Eigen::VectorXcd CrankNicolsonModes::gather_active(const Eigen::VectorXcd &full) const
{
    Eigen::VectorXcd compact(static_cast<Eigen::Index>(active_list.size()) * Nr);
    for (std::size_t b = 0; b < active_list.size(); ++b)
        compact.segment(b * Nr, Nr) = full.segment(static_cast<Eigen::Index>(active_list[b]) * Nr, Nr);
    return compact;
}

void CrankNicolsonModes::scatter_active(const Eigen::VectorXcd &compact, Eigen::VectorXcd &full) const
{
    for (std::size_t b = 0; b < active_list.size(); ++b)
        full.segment(static_cast<Eigen::Index>(active_list[b]) * Nr, Nr) = compact.segment(b * Nr, Nr);
}

void CrankNicolsonModes::apply_hamiltonian(const Eigen::VectorXcd &x, Eigen::VectorXcd &y, bool compact) const
{
    y.resize(x.size());
    const int blocks = compact ? static_cast<int>(active_list.size()) : n_modes;
    const std::size_t inner = Nr - 2;
    const double off = h0_offdiagonal;
    const double *lo = h0_lower.data(), *up = h0_upper.data();
    // Complex arithmetic is spelled out on (re, im) pairs so the inner loops vectorize.
    const double *xs = reinterpret_cast<const double *>(x.data());
#pragma omp parallel for schedule(dynamic) if (blocks >= 8)
    for (int b = 0; b < blocks; ++b)
    {
        const int j = compact ? active_list[b] : b;
        const double *xj = xs + 2 * static_cast<std::size_t>(b) * Nr;
        double *yj = reinterpret_cast<double *>(y.data()) + 2 * static_cast<std::size_t>(b) * Nr;
        const double *d = h0_diagonal.data() + static_cast<std::size_t>(j) * Nr;
        yj[0] = yj[1] = yj[2 * (Nr - 1)] = yj[2 * (Nr - 1) + 1] = 0.0;
        if (uniform_grid)
            for (int i = 1; i < Nr - 1; ++i)
            {
                yj[2 * i] = d[i] * xj[2 * i] + off * (xj[2 * i - 2] + xj[2 * i + 2]);
                yj[2 * i + 1] = d[i] * xj[2 * i + 1] + off * (xj[2 * i - 1] + xj[2 * i + 3]);
            }
        else
            for (int i = 1; i < Nr - 1; ++i)
            {
                yj[2 * i] = d[i] * xj[2 * i] + lo[i] * xj[2 * i - 2] + up[i] * xj[2 * i + 2];
                yj[2 * i + 1] = d[i] * xj[2 * i + 1] + lo[i] * xj[2 * i - 1] + up[i] * xj[2 * i + 3];
            }
        for (std::size_t p = row_pair_start[j]; p < row_pair_start[j + 1]; ++p)
        {
            const int k = mode_pairs[p].col_mode;
            const double *xk = xs + 2 * static_cast<std::size_t>(compact ? mode_position[k] : k) * Nr;
            const double *c = coupling_eff.data() + p * inner - 1; // c[i] for node i
            const double pr = pair_phase[p].real(), pi = pair_phase[p].imag();
            for (int i = 1; i < Nr - 1; ++i)
            {
                const double vr = pr * c[i], vi = pi * c[i];
                yj[2 * i] += vr * xk[2 * i] - vi * xk[2 * i + 1];
                yj[2 * i + 1] += vr * xk[2 * i + 1] + vi * xk[2 * i];
            }
        }
        if (cap)
            for (int i = 0; i < Nr; ++i)
                y[static_cast<Eigen::Index>(b) * Nr + i] += cap_diagonal[static_cast<Eigen::Index>(j) * Nr + i] *
                                                             x[static_cast<Eigen::Index>(b) * Nr + i];
    }
}

Eigen::VectorXcd CrankNicolsonModes::hamiltonian_times(const Eigen::VectorXcd &state) const
{
    Eigen::VectorXcd result;
    apply_hamiltonian(state, result, false);
    return result;
}

SparseComplex CrankNicolsonModes::active_cn_operator(dcomplex scale) const
{
    const SparseComplex h = H();
    std::vector<Eigen::Triplet<dcomplex>> entries;
    for (std::size_t b = 0; b < active_list.size(); ++b)
        for (int i = 0; i < Nr; ++i)
        {
            const Eigen::Index row = static_cast<Eigen::Index>(b) * Nr + i;
            entries.emplace_back(row, row, 1.0);
            for (SparseComplex::InnerIterator it(h, static_cast<Eigen::Index>(active_list[b]) * Nr + i); it; ++it)
            {
                // Active rows only couple to active modes.
                const Eigen::Index col = mode_position[it.col() / Nr] * Nr + it.col() % Nr;
                entries.emplace_back(row, col, scale * it.value());
            }
        }
    const Eigen::Index size = static_cast<Eigen::Index>(active_list.size()) * Nr;
    SparseComplex A(size, size);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
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
    const Eigen::Index n = b.size();
    // The operators write into preallocated vectors: no temporaries per iteration.
    Eigen::VectorXcd r(n), y(n), z(n), s(n), t(n);
    apply_A(x, t);
    r = b - t;
    Eigen::VectorXcd r0 = r;
    double r0_norm2 = r0.squaredNorm();
    const double restart_threshold = r0_norm2 * 1e-32;
    dcomplex rho = 1.0, alpha = 1.0, w = 1.0;
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(n), p = v;
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
            apply_A(x, t);
            r = b - t;
            r0 = r;
            rho = r0_norm2 = r.squaredNorm();
        }
        const dcomplex beta = (rho / rho_old) * (alpha / w);
        p = r + beta * (p - w * v);
        apply_M(p, y);
        apply_A(y, v);
        alpha = rho / r0.dot(v);
        s = r - alpha * v;
        apply_M(s, z);
        apply_A(z, t);
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

    // A = 1 + (i dt/2) H on the active modes. Active rank-one sinks add
    // (dt/4) * Gamma, Gamma = sum g_k v_k v_k^dagger.
    const dcomplex scale = I * (dt * 0.5);
    std::vector<Eigen::VectorXcd> sink_v, sink_w; // Gamma = g v w^dagger
    std::vector<double> sink_g;
    if (sink_active)
        for (const SwaveSink *sink : {&sink1, &sink2})
            if (sink->strength != 0.0 && sink->v.size() == psi.size() && sink->v.squaredNorm() > 0.0)
            {
                sink_v.push_back(gather_active(sink->v));
                sink_w.push_back(gather_active(sink->left()));
                sink_g.push_back(0.25 * dt * sink->strength);
            }
    const int k = static_cast<int>(sink_v.size());
    const auto apply_A = [&](const Eigen::VectorXcd &x, Eigen::VectorXcd &y) {
        apply_hamiltonian(x, y, true);
        y = x + scale * y;
        for (int a = 0; a < k; ++a)
            y += (sink_g[a] * sink_w[a].dot(x)) * sink_v[a];
    };

    Eigen::VectorXcd work(rhs.size());
    const auto residual = [&](const Eigen::VectorXcd &candidate) {
        if (!candidate.allFinite())
            return std::numeric_limits<double>::infinity();
        apply_A(candidate, work);
        return (work - rhs).norm() / rhs_norm;
    };
    // Normalize the system so the stopping threshold is independent of cloud
    // amplitude; the acceptance check below always uses the physical RHS.
    const Eigen::VectorXcd normalized_rhs = rhs / rhs_norm;
    const Eigen::VectorXcd guess = gather_active(psi) / rhs_norm;

    // Woodbury: the preconditioner P (radial blocks or ILUT of the sparse A)
    // is corrected exactly for the low-rank sink, so it stays as effective
    // as without absorption.
    const auto krylov_with = [&](const auto &P_solve, int max_iterations) {
        Eigen::MatrixXcd W(rhs.size(), k);
        Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(k, k);
        Eigen::VectorXcd column(rhs.size());
        for (int a = 0; a < k; ++a)
        {
            P_solve(sink_v[a], column);
            W.col(a) = column;
        }
        for (int a = 0; a < k; ++a)
        {
            G(a, a) += 1.0 / sink_g[a];
            for (int b = 0; b < k; ++b)
                G(a, b) += sink_w[a].dot(W.col(b));
        }
        const Eigen::MatrixXcd G_inverse = G.inverse();
        const auto apply_M = [&](const Eigen::VectorXcd &x, Eigen::VectorXcd &y) {
            P_solve(x, y);
            if (k == 0)
                return;
            Eigen::VectorXcd projections(k);
            for (int a = 0; a < k; ++a)
                projections[a] = sink_w[a].dot(y);
            y -= W * (G_inverse * projections);
        };
        Eigen::VectorXcd x = guess;
        bicgstab(apply_A, apply_M, normalized_rhs, x, 1e-10, max_iterations, last_solver_iterations);
        return Eigen::VectorXcd(x * rhs_norm);
    };

    if (use_block_preconditioner)
    {
        // Tridiagonal blocks of A: H0, the diagonal self-coupling and the CAP.
        const std::size_t inner = Nr - 2;
        block_preconditioner.factorizeBlocks(
            static_cast<Eigen::Index>(active_list.size()),
            [&](Eigen::Index b, int i, dcomplex &lower, dcomplex &diagonal, dcomplex &upper) {
                const int j = active_list[b];
                const bool interior = i > 0 && i < Nr - 1;
                dcomplex h = h0_diagonal[static_cast<std::size_t>(j) * Nr + i];
                if (interior && self_pair[j] >= 0)
                    h += pair_phase[self_pair[j]] * coupling_eff[self_pair[j] * inner + i - 1];
                if (cap)
                    h += cap_diagonal[static_cast<Eigen::Index>(j) * Nr + i];
                diagonal = 1.0 + scale * h;
                lower = interior ? scale * (uniform_grid ? h0_offdiagonal : h0_lower[i]) : dcomplex(0.0);
                upper = interior ? scale * (uniform_grid ? h0_offdiagonal : h0_upper[i]) : dcomplex(0.0);
            });
        if (block_preconditioner.info() == Eigen::Success)
        {
            Eigen::VectorXcd candidate = krylov_with([&](const Eigen::VectorXcd &x, Eigen::VectorXcd &y) {
                y = block_preconditioner.solve(x); }, 500);
            last_solver_residual = residual(candidate);
            if (last_solver_residual <= 1e-9)
            {
                scatter_active(candidate, psi);
                return;
            }
        }
        std::cerr << "\nRadial preconditioner did not converge; retrying with full ILUT.\n";
    }
    ilut_preconditioner.compute(active_cn_operator(scale));
    if (ilut_preconditioner.info() != Eigen::Success)
        throw std::runtime_error("Crank-Nicolson ILUT factorization failed");
    Eigen::VectorXcd candidate = krylov_with([&](const Eigen::VectorXcd &x, Eigen::VectorXcd &y) {
        y = ilut_preconditioner.solve(x); }, 5000);
    last_solver_residual = residual(candidate);
    if (!(last_solver_residual <= 1e-9))
        throw std::runtime_error("Crank-Nicolson solve failed the relative residual check");
    scatter_active(candidate, psi);
}
