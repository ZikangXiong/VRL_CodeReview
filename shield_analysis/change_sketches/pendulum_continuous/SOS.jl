#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:2]
f = [0.0*x[1]+1.0*x[2],-11.22253209202239*x[1]+-15.760697675710565*x[2]]
init1 = (x[1] - -0.35)*(0.35-x[1])
init2 = (x[2] - -0.35)*(0.35-x[2])
init3 = (x[1] - (-1.011910429412925))*((0.35)-x[1])
init4 = (x[2] - (-0.5113184737433119))*((0.35)-x[2])
unsafe1 = (x[1] - 0.0)^2 - 0.25
unsafe2 = (x[2] - 0.0)^2 - 0.25

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
@variable m Zinit1 SOSPoly(Z)
@variable m Zinit2 SOSPoly(Z)
@variable m Zinit3 SOSPoly(Z)
@variable m Zinit4 SOSPoly(Z)
Z = monomials(x, 0:1)
@variable m Zunsafe1 SOSPoly(Z)
@variable m Zunsafe2 SOSPoly(Z)
Z = monomials(x, 0:8)
@variable m B Poly(Z)

x1 = x[1] + 0.01*f[1]
x2 = x[2] + 0.01*f[2]

B1 = subs(B, x[1]=>x1, x[2]=>x2)

f1 = -B - Zinit1*init1 - Zinit2*init2 - Zinit3*init3 - Zinit4*init4
f2 = B - B1
f3 = B - Zunsafe1*unsafe1
f4 = B - Zunsafe2*unsafe2


@constraint m f1 >= 0
@constraint m f2 >= 0
@constraint m f3 >= 1
@constraint m f4 >= 1


status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))