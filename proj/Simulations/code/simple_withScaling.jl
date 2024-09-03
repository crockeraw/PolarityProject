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

function setup(V, seed)
    Random.seed!(seed)
    # Generate constants
    N = 100
    # V = (4/3)*pi*r^3
    r = cbrt(V/(4/3 * pi))
    SA = 4*pi*r^2
    mem_thickness = 0.01
    n = (mem_thickness * SA) / V

    Ax = Array(Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1]))
    Ax[1,end] = 1.0
    Ax[end,1] = 1.0
    dx = (r*sqrt(pi))/N
    Ax = Ax/(dx^2) # adjust for 1/microns
    Ay = copy(Ax)

    r0 = zeros(100,100,2)
    r0[:,:,1] .= 20 .*(rand.())   # Cdc42-GTPm
    r0[:,:,2] .= .5 - mean(r0[:,:,1])*n   # Cdc42-GDPm 
    # VALUE FOR TOTAL RHO SHIFTS RELATIONSHIP BETWEEN NUMBER OF SITES AND RADIUS
    
    # Dummy parameters used only locally in fxn but passed to specify scope, or something..
    Ayt = zeros(N,N)
    tAx = zeros(N,N)
    D42t = zeros(N,N)
    D42d = zeros(N,N)
    R = zeros(N,N)
    dummy = (Ayt, tAx, D42t, D42d, R)
    # Actual parameters
    a = 2
    b = 0.5
    Dm = 0.01
    Dc = 10
    n = n

    p = (a, b, Dm, Dc, n, Ax, Ay, dummy)
    return p, r0
end

function simple!(dr,r,p,t)
    a, b, Dm, Dc, n, Ax, Ay, dummy = p
    Ayt, tAx, D42t, D42d, R = dummy
    # Window variables
    rhoT = @view r[:,:,1]
    rhoD = @view r[:,:,2]
    # Calculate diffusion
    mul!(Ayt,Ay,rhoT)
    mul!(tAx,rhoT,Ax)
    @. D42t = Dm*(Ayt + tAx)
    mul!(Ayt,Ay,rhoD)
    mul!(tAx,rhoD,Ax)
    @. D42d = Dc*(Ayt + tAx)
    # Calculate reactions, add diffusion
    @. R = (a*rhoT^2*rhoD) - b*rhoT
    @. dr[:,:,1] = R + D42t
    @. dr[:,:,2] = -R*n + D42d
end

function runner(volume, seed)
    p, r0 = setup(volume, seed)
    min_prob = ODEProblem(simple!,r0,(0.0,1800),p)
    sol_simp = solve(min_prob,CVODE_BDF(linear_solver = :GMRES), saveat=(1799,1800))
    @save "../sims/simple_30min_seed_$(seed)_volume_$(volume).jld2" sol_simp
end

for arg in ARGS
    seed = parse(Int,arg)
    println("running seed $seed...")
    for v in (900,800,700,600,500,400,300,200,100,50)
        runner(v, seed)
        println("volume $(v) complete")
    end
    println("seed $seed complete!")
end