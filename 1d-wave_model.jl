print("Hello World")
using NeuralPDE, Lux, Optimization, OptimizationOptimJL
using ModelingToolkit: Interval

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#PDE
C = 1
eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))

# Initial and boundary conditions
bcs = [u(t, 0) ~ 0.0,# for all t > 0
    u(t, 1) ~ 0.0,# for all t > 0
    u(0, x) ~ x * (1.0 - x), #for all 0 < x < 1
    Dt(u(0, x)) ~ 0.0] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]
# Discretization
dx = 0.1

# Neural network
chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
discretization = PhysicsInformedNN(chain, GridTraining(dx))

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = OptimizationOptimJL.BFGS()
res = Optimization.solve(prob, opt; callback, maxiters = 250)
phi = discretization.phi

#PLOTS OF THE RESULTS
using Plots
ts, xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
function analytic_sol_func(t, x)
    sum([(8 / (k^3 * pi^3)) * sin(k * pi * x) * cos(C * k * pi * t) for k in 1:2:50000])
end

u_predict = reshape([first(phi([t, x], res.u)) for t in ts for x in xs],
    (length(ts), length(xs)))
u_real = reshape([analytic_sol_func(t, x) for t in ts for x in xs],
    (length(ts), length(xs)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(ts, xs, u_real, linetype = :contourf, title = "analytic", xlabel="Space domain", ylabel="Time domain");
p2 = plot(ts, xs, u_predict, linetype = :contourf, title = "predict",  xlabel="Space domain", ylabel="Time domain");
p3 = plot(ts, xs, diff_u, linetype = :contourf, title = "error",  xlabel="Space domain", ylabel="Time domain");
plot(p1, p2, p3)
savefig(p1, "analytic.png")
savefig(p2, "predict.png")
savefig(p3, "error.png")
savefig("all.png")
#save the results in a csv file
using CSV
using DataFrames

#convert the analytic solution into list and save the analytic solution in csv file
df_analytic= DataFrame(t = repeat(ts, inner = length(xs)), x = repeat(xs, outer = length(ts)), u_real = vec(u_real))
CSV.write("analytic.csv", df_analytic)
#convert the prediction into list and save the predicted results in csv file
df_predict= DataFrame(t = repeat(ts, inner = length(xs)), x = repeat(xs, outer = length(ts)), u_predict = vec(u_predict))
CSV.write("predict.csv", df_predict)

###################################################################################################################################################################################


#Hyperparameter tuning

using OptimizationOptimisers

opt1 = OptimizationOptimisers.Adam(0.001)
res1 = Optimization.solve(prob, opt1; callback, maxiters = 250)