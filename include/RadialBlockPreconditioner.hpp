/**
 * @file RadialBlockPreconditioner.hpp
 * @brief Block-Jacobi preconditioner that inverts each mode's radial block exactly.
 */
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cmath>
#include <stdexcept>
#include "Definitions.hpp"

/**
 * @brief Block-Jacobi preconditioner for Eigen's iterative solvers.
 *
 * It retains the complete radial tridiagonal block of each angular mode and
 * solves it exactly with the Thomas algorithm. Factorisation and application
 * are independent across modes and run in parallel. The outer BiCGSTAB
 * iteration still includes all inter-mode couplings of the matrix.
 */
class RadialBlockPreconditioner
{
public:
    using Scalar = dcomplex;
    using StorageIndex = int;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    /// Set the size of each diagonal block (the number of radial nodes).
    void setBlockSize(int size) { block_size_ = size; }
    Eigen::Index rows() const { return inverse_diagonal_.size(); }
    Eigen::Index cols() const { return rows(); }
    Eigen::ComputationInfo info() const { return info_; }

    template<class Matrix>
    RadialBlockPreconditioner& analyzePattern(const Matrix&) { return *this; }

    /// Factorise the tridiagonal blocks of @p matrix.
    template<class Matrix>
    RadialBlockPreconditioner& compute(const Matrix& matrix) { return factorize(matrix); }

    /// Factorise the tridiagonal blocks of @p matrix; info() reports a zero pivot.
    /// @throws std::invalid_argument if the matrix is not square or not a whole number of blocks.
    template<class Matrix>
    RadialBlockPreconditioner& factorize(const Matrix& matrix)
    {
        if (block_size_ < 1 || matrix.rows() != matrix.cols() || matrix.rows() % block_size_ != 0)
            throw std::invalid_argument("Invalid radial preconditioner block dimensions");
        inverse_diagonal_.resize(matrix.rows());
        lower_.resize(matrix.rows());
        upper_.resize(matrix.rows());
        const Eigen::Index blocks = matrix.rows() / block_size_;
        int failed = 0;
#pragma omp parallel for schedule(static) reduction(|:failed) if(blocks >= 8)
        for (Eigen::Index b = 0; b < blocks; ++b)
        {
            const Eigen::Index start = b * block_size_;
            for (int i = 0; i < block_size_; ++i)
            {
                const Eigen::Index row = start + i;
                lower_[row] = i > 0 ? matrix.coeff(row, row - 1) : Scalar(0);
                upper_[row] = i + 1 < block_size_ ? matrix.coeff(row, row + 1) : Scalar(0);
                Scalar pivot = matrix.coeff(row, row);
                if (i > 0)
                    pivot -= lower_[row] * inverse_diagonal_[row - 1] * upper_[row - 1];
                if (std::abs(pivot) == 0.0 || !std::isfinite(std::abs(pivot)))
                {
                    failed = 1;
                    inverse_diagonal_[row] = 0.0;
                }
                else
                    inverse_diagonal_[row] = Scalar(1) / pivot;
            }
        }
        info_ = failed ? Eigen::NumericalIssue : Eigen::Success;
        return *this;
    }

    /// Apply the inverse of the block diagonal (called by Eigen).
    template<class Rhs, class Dest>
    void _solve_impl(const Rhs& rhs, Dest& x) const
    {
        const Eigen::Index blocks = rows() / block_size_;
#pragma omp parallel for schedule(static) if(blocks >= 8)
        for (Eigen::Index b = 0; b < blocks; ++b)
        {
            const Eigen::Index start = b * block_size_;
            const Eigen::Index end = start + block_size_;
            for (Eigen::Index col = 0; col < rhs.cols(); ++col)
            {
                x(start, col) = rhs(start, col) * inverse_diagonal_[start];
                for (Eigen::Index row = start + 1; row < end; ++row)
                    x(row, col) = (rhs(row, col) - lower_[row] * x(row - 1, col)) * inverse_diagonal_[row];
                for (Eigen::Index row = end - 2; row >= start; --row)
                    x(row, col) -= upper_[row] * inverse_diagonal_[row] * x(row + 1, col);
            }
        }
    }

    /// Lazy solve expression, as required by Eigen's preconditioner interface.
    template<class Rhs>
    const Eigen::Solve<RadialBlockPreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs>& rhs) const
    {
        return Eigen::Solve<RadialBlockPreconditioner, Rhs>(*this, rhs.derived());
    }

private:
    int block_size_ = 0;
    Eigen::VectorXcd inverse_diagonal_, lower_, upper_;
    Eigen::ComputationInfo info_ = Eigen::InvalidInput;
};
