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

@polyvar x[1:15]
f = [-0.56060814*x[1]+-0.25517938*x[2]+-1.13701956*x[3]+0.52382169*x[4]+-0.49742993*x[5]+0.02079789*x[6]+0.50805297*x[7]+0.57714024*x[8]+-0.19954886*x[9]+-0.33626847*x[10]+-1.22834498*x[11]+0.08127906*x[12]+0.92109815*x[13]+0.92747549*x[14]+0.92131663*x[15],0.0*x[1]+0.0*x[2]+1.0*x[3]+0.0*x[4]+0.0*x[5]+0.0*x[6]+0.0*x[7]+0.0*x[8]+0.0*x[9]+0.0*x[10]+0.0*x[11]+0.0*x[12]+0.0*x[13]+0.0*x[14]+0.0*x[15],0.2933948599999999*x[1]+-1.66256157*x[2]+-2.85044129*x[3]+1.75724079*x[4]+0.45117438*x[5]+0.10118863*x[6]+-0.44828625*x[7]+-0.43733959*x[8]+-1.96334631*x[9]+-0.28707869999999996*x[10]+-3.33665416*x[11]+0.32791222*x[12]+-0.94733098*x[13]+-0.18012223000000005*x[14]+0.50523544*x[15],0.0*x[1]+0.0*x[2]+0.0*x[3]+0.0*x[4]+1.0*x[5]+0.0*x[6]+0.0*x[7]+0.0*x[8]+0.0*x[9]+0.0*x[10]+0.0*x[11]+0.0*x[12]+0.0*x[13]+0.0*x[14]+0.0*x[15],0.13469527000000003*x[1]+0.27715552*x[2]+1.98752524*x[3]+-2.38741905*x[4]+-1.20511688*x[5]+0.85941809*x[6]+2.0057099899999997*x[7]+1.79157312*x[8]+3.64277301*x[9]+-0.255371*x[10]+2.63013403*x[11]+-0.50305888*x[12]+3.11885439*x[13]+0.49368318*x[14]+0.31001283*x[15],0.0*x[1]+0.0*x[2]+0.0*x[3]+0.0*x[4]+0.0*x[5]+0.0*x[6]+1.0*x[7]+0.0*x[8]+0.0*x[9]+0.0*x[10]+0.0*x[11]+0.0*x[12]+0.0*x[13]+0.0*x[14]+0.0*x[15],-0.25982830999999995*x[1]+1.29592949*x[2]+-1.01206486*x[3]+1.54486765*x[4]+1.9169089799999999*x[5]+-1.67688052*x[6]+-2.9835578099999998*x[7]+-0.53173345*x[8]+-0.45874469000000007*x[9]+0.3536782*x[10]+-0.45861213*x[11]+1.6285154099999999*x[12]+-1.43420224*x[13]+-0.11226733*x[14]+0.9177846*x[15],0.0*x[1]+0.0*x[2]+0.0*x[3]+0.0*x[4]+0.0*x[5]+0.0*x[6]+0.0*x[7]+0.0*x[8]+1.0*x[9]+0.0*x[10]+0.0*x[11]+0.0*x[12]+0.0*x[13]+0.0*x[14]+0.0*x[15],-0.76717137*x[1]+-0.31045513*x[2]+0.75386183*x[3]+-2.02008739*x[4]+-3.23801791*x[5]+0.17027759999999992*x[6]+0.91441605*x[7]+-1.7098257000000001*x[8]+-1.64072187*x[9]+0.37137481*x[10]+0.51984126*x[11]+-1.2242570099999999*x[12]+0.1515243*x[13]+0.22831872000000003*x[14]+-0.9999229199999999*x[15],0.0*x[1]+0.0*x[2]+0.0*x[3]+0.0*x[4]+0.0*x[5]+0.0*x[6]+0.0*x[7]+0.0*x[8]+0.0*x[9]+0.0*x[10]+1.0*x[11]+0.0*x[12]+0.0*x[13]+0.0*x[14]+0.0*x[15],-0.2350115*x[1]+1.5555798900000002*x[2]+1.50484608*x[3]+1.9136928*x[4]+3.0493292800000003*x[5]+1.06474735*x[6]+0.7666584700000001*x[7]+3.09618136*x[8]+1.43212669*x[9]+-0.93070779*x[10]+-2.1590593399999998*x[11]+2.68973341*x[12]+2.18877103*x[13]+0.016380569999999983*x[14]+-0.57351279*x[15],0.0*x[1]+0.0*x[2]+0.0*x[3]+0.0*x[4]+0.0*x[5]+0.0*x[6]+0.0*x[7]+0.0*x[8]+0.0*x[9]+0.0*x[10]+0.0*x[11]+0.0*x[12]+1.0*x[13]+0.0*x[14]+0.0*x[15],0.52715882*x[1]+0.25936567999999993*x[2]+-0.62321709*x[3]+-1.12532887*x[4]+-2.13358908*x[5]+-1.81219168*x[6]+-1.17281008*x[7]+-2.1716428800000003*x[8]+-1.33194713*x[9]+-0.46430009*x[10]+0.5719736999999998*x[11]+-3.8638758*x[12]+-3.3102663*x[13]+1.55955171*x[14]+0.8084679499999999*x[15],0.0*x[1]+0.0*x[2]+0.0*x[3]+0.0*x[4]+0.0*x[5]+0.0*x[6]+0.0*x[7]+0.0*x[8]+0.0*x[9]+0.0*x[10]+0.0*x[11]+0.0*x[12]+0.0*x[13]+0.0*x[14]+1.0*x[15],-1.6836693999999999*x[1]+0.1745422299999999*x[2]+-0.27912268000000007*x[3]+0.47095694999999993*x[4]+-0.23839277000000003*x[5]+0.85571753*x[6]+0.78983638*x[7]+1.72438082*x[8]+1.71088552*x[9]+-0.13160918*x[10]+0.68395837*x[11]+0.8135685100000001*x[12]+0.80315884*x[13]+-0.99469543*x[14]+-4.01882031*x[15]]
init1 = (x[1] - -0.10000000000000142)*(0.10000000000000142-x[1])
init2 = (x[2] - -0.09999999999999998)*(0.10000000000000009-x[2])
init3 = (x[3] - -0.1)*(0.1-x[3])
init4 = (x[4] - -0.09999999999999998)*(0.10000000000000009-x[4])
init5 = (x[5] - -0.1)*(0.1-x[5])
init6 = (x[6] - -0.09999999999999998)*(0.10000000000000009-x[6])
init7 = (x[7] - -0.1)*(0.1-x[7])
init8 = (x[8] - -0.09999999999999998)*(0.10000000000000009-x[8])
init9 = (x[9] - -0.1)*(0.1-x[9])
init10 = (x[10] - -0.09999999999999998)*(0.10000000000000009-x[10])
init11 = (x[11] - -0.1)*(0.1-x[11])
init12 = (x[12] - -0.09999999999999998)*(0.10000000000000009-x[12])
init13 = (x[13] - -0.1)*(0.1-x[13])
init14 = (x[14] - -0.09999999999999998)*(0.10000000000000009-x[14])
init15 = (x[15] - -0.1)*(0.1-x[15])
init16 = (x[1] - (-0.20502938344160415))*((0.10000000000000141)-x[1])
init17 = (x[2] - (-0.09999999999999998))*((0.12355622719725354)-x[2])
init18 = (x[3] - (-0.2187128738947202))*((0.1)-x[3])
init19 = (x[4] - (-0.09999999999999998))*((0.1273341319095862)-x[4])
init20 = (x[5] - (-0.1))*((0.23873148720460155)-x[5])
init21 = (x[6] - (-0.09999999999999998))*((0.2363513638828766)-x[6])
init22 = (x[7] - (-0.12479787047128685))*((0.1)-x[7])
init23 = (x[8] - (-0.09999999999999998))*((0.17882421127082898)-x[8])
init24 = (x[9] - (-0.1))*((0.2637900880719115)-x[9])
init25 = (x[10] - (-0.1681215693229915))*((0.10000000000000007)-x[10])
init26 = (x[11] - (-0.22577254912939743))*((0.10000000000000002)-x[11])
init27 = (x[12] - (-0.2067891094977729))*((0.10000000000000009)-x[12])
init28 = (x[13] - (-0.1))*((0.23340282670552567)-x[13])
init29 = (x[14] - (-0.29576757276232457))*((0.10000000000000009)-x[14])
init30 = (x[15] - (-0.1975877501464063))*((0.1)-x[15])
unsafe1 = (x[1] - 0.0)^2 - 4.0
unsafe2 = (x[2] - -0.2)^2 - 0.48999999999999994
unsafe3 = (x[3] - 0.0)^2 - 1.0
unsafe4 = (x[4] - 0.0)^2 - 0.25
unsafe5 = (x[5] - 0.0)^2 - 1.0
unsafe6 = (x[6] - 0.0)^2 - 0.25
unsafe7 = (x[7] - 0.0)^2 - 1.0
unsafe8 = (x[8] - 0.0)^2 - 0.25
unsafe9 = (x[9] - 0.0)^2 - 1.0
unsafe10 = (x[10] - 0.0)^2 - 0.25
unsafe11 = (x[11] - 0.0)^2 - 1.0
unsafe12 = (x[12] - 0.0)^2 - 0.25
unsafe13 = (x[13] - 0.0)^2 - 1.0
unsafe14 = (x[14] - 0.0)^2 - 0.25
unsafe15 = (x[15] - 0.0)^2 - 1.0

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
@variable m Zinit1 SOSPoly(Z)
@variable m Zinit2 SOSPoly(Z)
@variable m Zinit3 SOSPoly(Z)
@variable m Zinit4 SOSPoly(Z)
@variable m Zinit5 SOSPoly(Z)
@variable m Zinit6 SOSPoly(Z)
@variable m Zinit7 SOSPoly(Z)
@variable m Zinit8 SOSPoly(Z)
@variable m Zinit9 SOSPoly(Z)
@variable m Zinit10 SOSPoly(Z)
@variable m Zinit11 SOSPoly(Z)
@variable m Zinit12 SOSPoly(Z)
@variable m Zinit13 SOSPoly(Z)
@variable m Zinit14 SOSPoly(Z)
@variable m Zinit15 SOSPoly(Z)
@variable m Zinit16 SOSPoly(Z)
@variable m Zinit17 SOSPoly(Z)
@variable m Zinit18 SOSPoly(Z)
@variable m Zinit19 SOSPoly(Z)
@variable m Zinit20 SOSPoly(Z)
@variable m Zinit21 SOSPoly(Z)
@variable m Zinit22 SOSPoly(Z)
@variable m Zinit23 SOSPoly(Z)
@variable m Zinit24 SOSPoly(Z)
@variable m Zinit25 SOSPoly(Z)
@variable m Zinit26 SOSPoly(Z)
@variable m Zinit27 SOSPoly(Z)
@variable m Zinit28 SOSPoly(Z)
@variable m Zinit29 SOSPoly(Z)
@variable m Zinit30 SOSPoly(Z)
Z = monomials(x, 0:1)
@variable m Zunsafe1 SOSPoly(Z)
@variable m Zunsafe2 SOSPoly(Z)
@variable m Zunsafe3 SOSPoly(Z)
@variable m Zunsafe4 SOSPoly(Z)
@variable m Zunsafe5 SOSPoly(Z)
@variable m Zunsafe6 SOSPoly(Z)
@variable m Zunsafe7 SOSPoly(Z)
@variable m Zunsafe8 SOSPoly(Z)
@variable m Zunsafe9 SOSPoly(Z)
@variable m Zunsafe10 SOSPoly(Z)
@variable m Zunsafe11 SOSPoly(Z)
@variable m Zunsafe12 SOSPoly(Z)
@variable m Zunsafe13 SOSPoly(Z)
@variable m Zunsafe14 SOSPoly(Z)
@variable m Zunsafe15 SOSPoly(Z)
Z = monomials(x, 0:4)
@variable m B Poly(Z)

x1 = x[1] + 0.01*f[1]
x2 = x[2] + 0.01*f[2]
x3 = x[3] + 0.01*f[3]
x4 = x[4] + 0.01*f[4]
x5 = x[5] + 0.01*f[5]
x6 = x[6] + 0.01*f[6]
x7 = x[7] + 0.01*f[7]
x8 = x[8] + 0.01*f[8]
x9 = x[9] + 0.01*f[9]
x10 = x[10] + 0.01*f[10]
x11 = x[11] + 0.01*f[11]
x12 = x[12] + 0.01*f[12]
x13 = x[13] + 0.01*f[13]
x14 = x[14] + 0.01*f[14]
x15 = x[15] + 0.01*f[15]

B1 = subs(B, x[1]=>x1, x[2]=>x2, x[3]=>x3, x[4]=>x4, x[5]=>x5, x[6]=>x6, x[7]=>x7, x[8]=>x8, x[9]=>x9, x[10]=>x10, x[11]=>x11, x[12]=>x12, x[13]=>x13, x[14]=>x14, x[15]=>x15)

f1 = -B - Zinit1*init1 - Zinit2*init2 - Zinit3*init3 - Zinit4*init4 - Zinit5*init5 - Zinit6*init6 - Zinit7*init7 - Zinit8*init8 - Zinit9*init9 - Zinit10*init10 - Zinit11*init11 - Zinit12*init12 - Zinit13*init13 - Zinit14*init14 - Zinit15*init15 - Zinit16*init16 - Zinit17*init17 - Zinit18*init18 - Zinit19*init19 - Zinit20*init20 - Zinit21*init21 - Zinit22*init22 - Zinit23*init23 - Zinit24*init24 - Zinit25*init25 - Zinit26*init26 - Zinit27*init27 - Zinit28*init28 - Zinit29*init29 - Zinit30*init30
f2 = B - B1
f3 = B - Zunsafe1*unsafe1
f4 = B - Zunsafe2*unsafe2
f5 = B - Zunsafe3*unsafe3
f6 = B - Zunsafe4*unsafe4
f7 = B - Zunsafe5*unsafe5
f8 = B - Zunsafe6*unsafe6
f9 = B - Zunsafe7*unsafe7
f10 = B - Zunsafe8*unsafe8
f11 = B - Zunsafe9*unsafe9
f12 = B - Zunsafe10*unsafe10
f13 = B - Zunsafe11*unsafe11
f14 = B - Zunsafe12*unsafe12
f15 = B - Zunsafe13*unsafe13
f16 = B - Zunsafe14*unsafe14
f17 = B - Zunsafe15*unsafe15


@constraint m f1 >= 0
@constraint m f2 >= 0
@constraint m f3 >= 1
@constraint m f4 >= 1
@constraint m f5 >= 1
@constraint m f6 >= 1
@constraint m f7 >= 1
@constraint m f8 >= 1
@constraint m f9 >= 1
@constraint m f10 >= 1
@constraint m f11 >= 1
@constraint m f12 >= 1
@constraint m f13 >= 1
@constraint m f14 >= 1
@constraint m f15 >= 1
@constraint m f16 >= 1
@constraint m f17 >= 1


status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))