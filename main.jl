using CSV
using DataFrames
using Flux
using MLUtils
using Statistics
using Random
using Plots
import Dates

# -----------------------------
# 1. Load and Prepare Data
# -----------------------------
df = CSV.read("iris.csv", DataFrame)

# Extract features and labels
X = Matrix(df[:, 1:4])'  # shape (4, 150)
labels = df.species

# Label encoding
label_map = Dict("Iris-setosa" => 1, "Iris-versicolor" => 2, "Iris-virginica" => 3)
y_int = [label_map[string(lbl)] for lbl in labels]
Y = Flux.onehotbatch(y_int, 1:3)

# Normalize features (column-wise for transposed data)
X = (X .- mean(X, dims=2)) ./ std(X, dims=2)

# Convert all to Float32
X = Float32.(X)
Y = Float32.(Y)

# Shuffle and split
Random.seed!(69)
dataset = [(X[:, i], Y[:, i]) for i in 1:size(X, 2)]
train_data, test_data = splitobs(shuffleobs(dataset), at=0.8)

# -----------------------------
# 2. Define Model
# -----------------------------
model = Chain(
    Dense(4, 16, relu),
    Dense(16, 8, relu),
    Dense(8, 3),
    softmax
)

loss(m, x, y) = Flux.crossentropy(m(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
opt_state = Flux.setup(Adam(0.01), model)  # custom learning rate

# -----------------------------
# 3. Training Loop
# -----------------------------
epochs = 50
train_loss = Float64[]
train_acc = Float64[]

for epoch in 1:epochs
    for (x, y) in train_data
        gs = Flux.gradient(model) do m
            loss(m, x, y)
        end
        Flux.update!(opt_state, model, gs[1])
    end

    # Evaluate on train set
    train_X = Float32.(hcat(first.(train_data)...))
    train_Y = Float32.(hcat(last.(train_data)...))
    l = loss(model, train_X, train_Y)
    a = accuracy(train_X, train_Y)

    push!(train_loss, l)
    push!(train_acc, a)
    println("Epoch $epoch - Loss: $(round(l, digits=4)) | Accuracy: $(round(a * 100, digits=2))%")
end

# -----------------------------
# 4. Evaluate
# -----------------------------
test_X = Float32.(hcat(first.(test_data)...))
test_Y = Float32.(hcat(last.(test_data)...))
test_acc = accuracy(test_X, test_Y)
println("Final Test Accuracy: ", round(test_acc * 100, digits=2), "%")
println("Final Test Loss: ", round(loss(model, test_X, test_Y), digits=4))

# -----------------------------
# 5. Save Results & Plots
# -----------------------------
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")
mkpath("results")

# Predict on test set
ŷ = Flux.onecold(model(test_X))
y_true = Flux.onecold(test_Y)

# Create confusion matrix
conf_matrix = zeros(Int, 3, 3)
for (true_label, pred_label) in zip(y_true, ŷ)
    conf_matrix[true_label, pred_label] += 1
end

# Plot confusion matrix
heatmap(conf_matrix, xticks=(1:3, ["Setosa", "Versicolor", "Virginica"]),
                    yticks=(1:3, ["Setosa", "Versicolor", "Virginica"]),
                    xlabel="Predicted", ylabel="True", cbar_title="Count",
                    title="Confusion Matrix", size=(500, 400), color=:blues)
savefig("results/confusion_matrix_$timestamp.png")


plot(train_loss, xlabel="Epoch", ylabel="Loss", title="Training Loss", legend=false)
savefig("results/loss_$timestamp.png")

plot(train_acc, xlabel="Epoch", ylabel="Accuracy", title="Training Accuracy", legend=false)
savefig("results/accuracy_$timestamp.png")
