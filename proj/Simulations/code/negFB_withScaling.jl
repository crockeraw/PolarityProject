using DifferentialEquations
using ModelingToolkit
using LinearAlgebra
using Plots
using Colors
using Images
using Statistics
using Sundials
using Random
using JLD2

function setup(r, seed)
    Random.seed!(seed)
    # Generate constants
    N = 100
    SA = 4*pi*r^2
    V = (4/3)*pi*r^3
    mem_thickness = 0.01
    n = (mem_thickness * SA) / V #scaling factor: volume at membrane region divided by volume of cytosol

    Ax = Array(Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1]))
    Ax[1,end] = 1.0
    Ax[end,1] = 1.0
    dx = (r*sqrt(pi))/N
    Ax = Ax/(dx^2) # adjust for 1/microns
    Ay = copy(Ax)

    r0 = zeros(100,100,3)
    r0[:,:,1] .= 10 .*(rand.())   # Cdc42-GTPm
    r0[:,:,2] .= .2 - mean(r0[:,:,1])*n   # Cdc42-GDPm
    r0[:,:,3] .= 0
    
    # Dummy parameters used only locally in fxn but passed to specify scope, or something..
    Ayt = zeros(N,N)
    tAx = zeros(N,N)
    D42t = zeros(N,N)
    D42d = zeros(N,N)
    Dpak = zeros(N,N)
    R = zeros(N,N)
    dummy = (Ayt, tAx, D42t, D42d, Dpak, R)
    # Actual parameters
    a = 1
    b = 0.25
    c = .5
    d = 0.001
    e = 0.1
    Dm = 0.01
    Dc = 10
    Dm2 = 0.02
    n = n

    p = (a, b, c, d, e, Dm, Dc, Dm2, n, Ax, Ay, dummy)
    return p, r0
end

function negfb!(dr,r,p,t)
    a, b, c, d, e, Dm, Dc, Dm2, n, Ax, Ay, dummy = p
    Ayt, tAx, D42t, D42d, Dpak, R = dummy
    # Window variables
    rhoT = @view r[:,:,1]
    rhoD = @view r[:,:,2]
    pak = @view r[:,:,3]
    # Calculate diffusion
    mul!(Ayt,Ay,rhoT)
    mul!(tAx,rhoT,Ax)
    @. D42t = Dm*(Ayt + tAx)
    mul!(Ayt,Ay,rhoD)
    mul!(tAx,rhoD,Ax)
    @. D42d = Dc*(Ayt + tAx)
    mul!(Ayt,Ay,pak)
    mul!(tAx,pak,Ax)
    @. Dpak = Dm2*(Ayt + tAx)
    # Calculate reactions, add diffusion
    @. R = (a*rhoT^2*rhoD) - b*rhoT - c*rhoT*pak^2
    @. dr[:,:,1] = R + D42t
    @. dr[:,:,2] = -R*n + D42d
    @. dr[:,:,3] = d*rhoT - e*pak + Dpak
end

function runner(radius, seed)
    p, r0 = setup(radius, seed)
    neg_prob = ODEProblem(negfb!,r0,(0.0,600),p)
    sol_neg = solve(neg_prob,CVODE_BDF(linear_solver = :GMRES), saveat=(500:600))
    @save "../sims/negFB_10min_seed_$(seed)_radius_$(radius).jld2" sol_neg
end

for arg in ARGS
    seed = parse(Int,arg)
    println("running seed $seed...")
    for r in (10,8,7,6,5,4,3,2)
        runner(r, seed)
        println("radius $(r) complete")
    end
    println("seed $seed complete!")
end