# Run setting
- 10 seeds
- 200 evals (appox. 2 sec)

# Targets
- HPOBench (MLP 8)
- HPOLib (4)
- JAHSBench201 (2)
- Benchmark functions (11 x 3 = 33/{5, 10, 30} dims)
    1. Sphere (simple convex)
    2. Styblinski (multimodal)
    3. Rastrigin (multimodal)
    4. Schwefel (separable, multimodal)
    5. Ackley (multimodal)
    6. Griewank (multimodal)
    7. Perm (simple convex)
    8. KTablet (ill-conditioned convex)
    9. WeightedSphere (ill-conditioned convex)
    10. Rosenbrock
    11. Levy

# Control parameters
- gamma (8)
    1. linear (0.05, 0.1, 0.15, 0.2)
    2. sqrt (0.25, 0.5, 0.75, 1.0)
- weighting strategy (3 / {uniform, recency decay, expected improvement})
- kernel (2 / {multivariate, univariate})
- magic clipping (3 x 4 = 12)
    1. continuous (1/100, 1/50, 1/10, 1/5 / not for HPOBench and HPOLib) 
    2. discrete (0.5/grids, 1.0/grids, 1.5/grids / not for Benchmark functions)

# Estimated budget
- Targets
    1. Continuous: 11 x 3 = 33
    2. Discrete: 12
    3. Mix: 2
- Control parameter settings
    1. Continuous: 8 x 3 x 2 x 4 = 192
    2. Discrete: 8 x 3 x 2 x 3 = 144
    3. Mix: 8 x 3 x 2 x 3 x 4 = 576
- Total settings (considering seeds)
    1. Continuous: 33 x 192 x 10 = 63360
    2. Discrete: 12 x 144 x 10 = 17280
    3. Mix: 2 x 576 x 10 = 11520

1: 1 / 20 = 0.05
2: 2 / 25 = 0.08
3: 3 / 30 = 0.1
4: 4 / 35 = 0.114
5: 5 / 60 = 0.08

Optuna TPE is not recency decay, but decay by performance.
