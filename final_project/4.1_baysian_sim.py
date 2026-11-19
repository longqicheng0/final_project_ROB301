import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
office = np.arange(2,13)


top_map_colors = [
    "yellow", "green", "blue",
    "orange", "orange", "green",
    "blue", "orange", "yellow",
    "green", "blue"
]

motion_model = {
    -1: (0.85, 0.10, 0.05),
    0:  (0.05, 0.90, 0.05),
    1:  (0.05, 0.10, 0.85)
}

sensor_model = {
    "blue":     np.array([0.60, 0.20, 0.05, 0.05]),
    "green":    np.array([0.20, 0.60, 0.05, 0.05]),
    "yellow":   np.array([0.05, 0.05, 0.65, 0.20]),
    "orange":   np.array([0.05, 0.05, 0.15, 0.60]),
    "nothing":  np.array([0.10, 0.10, 0.10, 0.10])
}

commands = [1] * 8 + [0] + [1] * 3

true_readings = [
    "orange", "yellow", "green", "blue",
    "nothing", "green", "blue", "green",
    "orange", "yellow", "green", "blue"
]

beliefs = np.array([[1/11] * 11])  # uniform distribution vector indicating the initla belief of the robot's position
history = [beliefs.sum(axis=0)]  #record of every probablity summed at position j

color_index = {
    "blue": 0,
    "green": 1,
    "yellow": 2,
    "orange": 3,
}

print("\nStep 0: initial uniform belief")
print("Belief:", np.round(history[0], 3))
print("Estimated office: unknown (uniform prior)")

for step, (command, true_reading) in enumerate(
    zip(commands, true_readings), start=1
):
    reverse, stay, forward = motion_model[command] 

    previous = history[-1]
    n = len(office)
    prediction = np.zeros(n)

    #prior update: update the prediction on each col given possible previous positions
    for j in range (n):
        prediction[j] = forward * previous[(j - 1) % n] + stay * previous[j] + reverse * previous[(j + 1) % n]

    #measurement update: given the true reading, update the likelihood of each color at each position
    likelihood = np.zeros(n)
    
    for k in range(n):
        color = top_map_colors[k]

        likelihood[k] = sensor_model[true_reading][color_index[color]]

    #posterior update: combine the prediction and likelihood to get the posterior distribution
    posterior = prediction * likelihood
    normalized_posterior = posterior / np.sum(posterior)

    #add this back into history so we use x-1 for next iteration
    history.append(normalized_posterior)

    best_index = np.argmax(normalized_posterior)

    print(f"\nStep {step}: u={command}, z={true_reading}")
    print("Belief:", np.round(normalized_posterior, 3))
    print("Estimated office:", office[best_index])
    print("Confidence:", round(normalized_posterior[best_index], 3))

    
    




    
 


# Convert history into a matrix.
# Rows = 11 offices, columns = 13 time steps.
heatmap = np.array(history).T

fig, ax = plt.subplots(figsize=(10, 5))
image = ax.imshow(
    heatmap,
    cmap="viridis",
    aspect="auto",
    origin="lower",
    vmin=0,
    vmax=1
)

for office_index in range(len(office)):
    for step_index in range(len(history)):
        probability = heatmap[office_index, step_index]
        text_color = "black" if probability >= 0.5 else "white"
        ax.text(
            step_index,
            office_index,
            f"{probability:.3f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=6,
        )

fig.colorbar(image, ax=ax, label="Probability")
ax.set_xticks(range(len(history)))
ax.set_yticks(range(len(office)), office)
ax.set_xlabel("Time Step (k)")
ax.set_ylabel("Office Number")
ax.set_title("Bayesian Localization - Belief Evolution")

fig.tight_layout()
output_path = "belief_heatmap.png"
fig.savefig(output_path, dpi=300)
plt.close(fig)

final_belief = history[-1]
final_index = np.argmax(final_belief)
print("\nFinal estimate")
print("Office:", office[final_index])
print("Confidence:", round(final_belief[final_index], 3))
print("Saved heatmap:", output_path)
