def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    #    Version September 2007 (and a small change August 2008)
    #    Krister Svanberg <krille@math.kth.se>
    #    Department of Mathematics, SE-10044 Stockholm, Sweden.

    #    This function mmasub performs one MMA-iteration, aimed at
    #    solving the nonlinear programming problem:
    #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #                xmin_j <= x_j <= xmax_j,    j = 1,...,n
    #                z >= 0,   y_i >= 0,         i = 1,...,m
    # *** INPUT:
    #   m    = The number of general constraints.
    #   n    = The number of variables x_j.
    #  iter  = Current iteration number ( =1 the first time mmasub is called).
    #  xval  = Column vector with the current values of the variables x_j.
    #  xmin  = Column vector with the lower bounds for the variables x_j.
    #  xmax  = Column vector with the upper bounds for the variables x_j.
    #  xold1 = xval, one iteration ago (provided that iter>1).
    #  xold2 = xval, two iterations ago (provided that iter>2).
    #  f0val = The value of the objective function f_0 at xval.
    #  df0dx = Column vector with the derivatives of the objective function
    #          f_0 with respect to the variables x_j, calculated at xval.
    #  fval  = Column vector with the values of the constraint functions f_i,
    #          calculated at xval.
    #  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #          f_i with respect to the variables x_j, calculated at xval.
    #          dfdx(i,j) = the derivative of f_i with respect to x_j.
    #  low   = Column vector with the lower asymptotes from the previous
    #          iteration (provided that iter>1).
    #  upp   = Column vector with the upper asymptotes from the previous
    #          iteration (provided that iter>1).
    #  a0    = The constants a_0 in the term a_0*z.
    #  a     = Column vector with the constants a_i in the terms a_i*z.
    #  c     = Column vector with the constants c_i in the terms c_i*y_i.
    #  d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    # *** OUTPUT:
    #  xmma  = Column vector with the optimal values of the variables x_j
    #          in the current MMA subproblem.
    #  ymma  = Column vector with the optimal values of the variables y_i
    #          in the current MMA subproblem.
    #  zmma  = Scalar with the optimal value of the variable z
    #          in the current MMA subproblem.
    #  lam   = Lagrange multipliers for the m general MMA constraints.
    #  xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    #  eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
    #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    #  zet   = Lagrange multiplier for the single constraint -z <= 0.
    #   s    = Slack variables for the m general MMA constraints.
    #  low   = Column vector with the lower asymptotes, calculated and used
    #          in the current MMA subproblem.
    #  upp   = Column vector with the upper asymptotes, calculated and used
    #          in the current MMA subproblem.

    epsimin = 10 ** (- 7)
    raa0 = 1e-05
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.4

    # Calculation of the asymptotes low and upp :
    if iter < 2.5:
        move = 1
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        move = 0.5
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones(n)
        factor[np.where(zzz > 0)] = asyincr
        factor[np.where(zzz < 0)] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 0.1 * (xmax - xmin)
        lowmax = xval - 0.0001 * (xmax - xmin)
        uppmin = xval + 0.0001 * (xmax - xmin)
        uppmax = xval + 0.1 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta :
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)
    # Calculations of p0, q0, P, Q and b.
    xmami = xmax - xmin
    xmamieps = np.ones(n) * 1e-05
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = np.ones(n) / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = np.ones(n) / ux1
    xlinv = np.ones(n) / xl1

    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(- df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2

    P = np.maximum(dfdx, 0)
    Q = np.maximum(- dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.ones(m) * xmamiinv.T
    P = P + PQ
    Q = Q + PQ
    P = P * scipy.sparse.spdiags(ux2, 0, n, n)
    Q = Q * scipy.sparse.spdiags(xl2, 0, n, n)
    b = np.dot(P[np.newaxis], uxinv[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv[np.newaxis].T)[0] - fval

    ## Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b,
                                                          c, d)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp