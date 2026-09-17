// rocALUTION CG gate: 1D Laplacian SPD (n=256), Jacobi-preconditioned CG
// on gfx1013 vs exact-ones reference. Passes if the recovered solution
// matches ones within 1e-6.
#include <cmath>
#include <cstdio>
#include <vector>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

int main() {
    init_rocalution();
    set_omp_threads_rocalution(1);
    const int n = 256;
    const int nnz = n * 3 - 2;
    // Owned by the matrix after SetDataPtrCSR: allocate with new[], never free here.
    int* row_ptr = new int[n + 1];
    int* col = new int[nnz];
    double* val = new double[nnz];
    int off = 0;
    for (int i = 0; i < n; ++i) {
        row_ptr[i] = off;
        if (i > 0) {
            col[off] = i - 1;
            val[off++] = -1.0;
        }
        col[off] = i;
        val[off++] = 2.0;
        if (i < n - 1) {
            col[off] = i + 1;
            val[off++] = -1.0;
        }
    }
    row_ptr[n] = off;
    LocalMatrix<double> mat;
    mat.SetDataPtrCSR(&row_ptr, &col, &val, "lap1d", nnz, n, n);
    LocalVector<double> x, rhs, e;
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();
    e.MoveToAccelerator();
    x.Allocate("x", n);
    rhs.Allocate("rhs", n);
    e.Allocate("e", n);
    e.Ones();
    mat.Apply(e, &rhs);
    x.Zeros();
    int iters = 0;
    double res = 0.0;
    {
        Jacobi<LocalMatrix<double>, LocalVector<double>, double> p;
        CG<LocalMatrix<double>, LocalVector<double>, double> ls;
        ls.SetOperator(mat);
        ls.SetPreconditioner(p);
        ls.Init(1e-9, 1e-12, 1e8, 1000);
        ls.Build();
        ls.Solve(rhs, &x);
        iters = ls.GetIterationCount();
        res = ls.GetCurrentResidual();
    }
    x.MoveToHost();
    std::vector<double> hx(n);
    x.CopyToHostData(hx.data());
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        const double err = std::fabs(hx[i] - 1.0);
        if (err > max_err) {
            max_err = err;
        }
    }
    const bool ok = max_err < 1.0e-6;
    std::printf("ROCALUTION_CG_VERIFY iters=%d res=%.2e max_err=%.2e %s\n", iters,
                res, max_err, ok ? "OK" : "MISMATCH");
    stop_rocalution();
    std::printf("ROCALUTION_CG_VERIFY_%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
