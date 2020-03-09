def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    # This function subsolv solves the MMA subproblem:
    # minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
    #          + SUM[ ci*yi + 0.5*di*(yi)^2 ],
    # subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
    #            alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
    # Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    # Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.

    epsi = 1
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1
    lam = np.ones(m)
    xsi = np.ones(n) / (x - alfa)
    xsi = np.maximum(xsi, np.ones(n))
    eta = np.ones(n) / (beta - x)
    eta = np.maximum(eta, np.ones(n))
    mu = np.maximum(np.ones(m), 0.5 * c)
    zet = 1
    s = np.ones(m)

    it1 = 0
    while epsi > epsimin:
        it1 = it1 + 1

        epsvecn = epsi * np.ones(n)
        epsvecm = epsi * np.ones(m)
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = np.ones(n) / ux1
        xlinv1 = np.ones(n) / xl1
        plam = p0 + P.T * lam
        qlam = q0 + Q.T * lam
        gvec = np.dot(P[np.newaxis], uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv1[np.newaxis].T)[0]
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T * lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate([rex, rey, rez]).T
        residu2 = np.concatenate([relam, rexsi, reeta, remu, [rezet], res])
        residu = np.concatenate([residu1, residu2])
        residunorm = np.sqrt(np.dot(residu[np.newaxis], residu[np.newaxis].T)[0][
                                 0])  # PROBABLY WRONH; THE VALUE IS TO HIGH WHEN COMPARED TO MATLAB
        residumax = max(abs(residu))

        it2 = 0
        while np.logical_and(residumax > 0.9 * epsi, it2 < 500):
            it2 = it2 + 1

            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = np.ones(n) / ux1
            xlinv1 = np.ones(n) / xl1
            uxinv2 = np.ones(n) / ux2
            xlinv2 = np.ones(n) / xl2
            plam = p0 + P.T * lam
            qlam = q0 + Q.T * lam
            gvec = np.dot(P[np.newaxis], uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv1[np.newaxis].T)[0]
            GG = P * scipy.sparse.spdiags(uxinv2, 0, n, n) - Q * scipy.sparse.spdiags(xlinv2, 0, n, n)
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T * lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = np.ones(n) / diagx
            diagy = d + mu / y
            diagyinv = np.ones(m) / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv
            if m > n:
                blam = dellam + dely / diagy - np.dot(GG[np.newaxis], (delx / diagx)[np.newaxis].T)[0]
                bb = np.concatenate([blam.T, delz]).T
                Alam = diaglamyi[0] + \
                       np.dot((GG * scipy.sparse.spdiags(diagxinv, 0, n, n))[np.newaxis], GG[np.newaxis].T)[0][0]
                AA = [[Alam, a[0]], [a[0], - zet / z]]
                solut = np.linalg.solve(AA, bb)
                dlam = solut[np.arange(0, m)]
                dz = solut[m]
                dx = - delx / diagx - (GG.T * dlam) / diagx
            else:
                diaglamyiinv = np.ones(m) / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = scipy.sparse.spdiags(diagx, 0, n, n) + GG[np.newaxis].T * scipy.sparse.spdiags(diaglamyiinv, 0, m,
                                                                                                     m) * GG
                azz = zet / z + a.T * (a / diaglamyi)
                axz = - GG.T * (a / diaglamyi)
                bx = delx + GG.T * (dellamyi / diaglamyi)
                bz = delz - a.T * (dellamyi / diaglamyi)
                AA1 = np.c_[Axx, axz]
                AA2 = np.concatenate([axz.T, azz])
                AA = np.r_[AA1, AA2[np.newaxis]]
                bb = np.concatenate([- bx.T, - bz])
                solut = np.linalg.solve(AA, bb)
                dx = solut[np.arange(0, n)]
                dz = solut[n]
                dlam = (np.dot(GG[np.newaxis], dx[np.newaxis].T)[0][0]) / diaglamyi - dz * (
                            a / diaglamyi) + dellamyi / diaglamyi
            dy = - dely / diagy + dlam / diagy
            dxsi = - xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = - eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = - mu + epsvecm / y - (mu * dy) / y
            dzet = - zet + epsi / z - zet * dz / z
            ds = - s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y.T, [z], lam, xsi.T, eta.T, mu.T, [zet], s.T), axis=0)
            dxx = np.concatenate((dy.T, [dz], dlam.T, dxsi.T, deta.T, dmu.T, [dzet], ds.T), axis=0)
            stepxx = - 1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = - 1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1)
            steg = 1 / stminv
            xold = x
            yold = y
            zold = z
            lamold = lam
            xsiold = xsi
            etaold = eta
            muold = mu
            zetold = zet
            sold = s
            resinew = 2 * residunorm

            it3 = 0
            while np.logical_and(resinew > residunorm, it3 < 50).any():
                it3 = it3 + 1

                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = np.ones(n) / ux1
                xlinv1 = np.ones(n) / xl1
                plam = p0 + P.T * lam
                qlam = q0 + Q.T * lam
                gvec = np.dot(P[np.newaxis], uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv1[np.newaxis].T)[0]
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T * lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate([rex.T, rey.T, rez]).T
                residu2 = np.concatenate([relam.T, rexsi.T, reeta.T, remu.T, [rezet], res.T]).T
                residu = np.concatenate([residu1.T, residu2.T]).T
                resinew = np.sqrt(np.dot(residu[np.newaxis], residu[np.newaxis].T)[0][0])
                steg = steg / 2
            residunorm = resinew
            residumax = max(abs(residu))
            # steg = 2 * steg
        epsi = 0.1 * epsi
    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
