import Pkg

# List of required packages
packages = [
    "Flux",
    "CSV",
    "DataFrames",
    "RDatasets",
    "MLUtils",
    "Plots",
    "Statistics",
    "Random"
]

# Install all packages
for pkg in packages
    Pkg.add(pkg)
end

