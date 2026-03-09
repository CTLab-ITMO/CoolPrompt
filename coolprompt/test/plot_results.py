import json
import matplotlib.pyplot as plt
from prompts import META_PREFIX

with open("logs_hype/" + META_PREFIX + "results.json", "r") as f:
    data = json.load(f)

prompt_ids = [prompt["id"] for prompt in data["prompts"]]
compute_times = [prompt["compute_time"] for prompt in data["prompts"]]

plt.figure(figsize=(12, 6))
bars = plt.bar(prompt_ids, compute_times, color="skyblue", edgecolor="navy")

plt.xlabel("Prompt ID")
plt.ylabel("Computation Time (seconds)")
plt.title("Computation Time by Prompt ID")
plt.xticks(prompt_ids)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height:.2f}s",
        ha="center",
        va="bottom",
        rotation=0,
    )

plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.savefig(f"logs_hype/{META_PREFIX}compute_time_histogram.png")
print(f"Histogram saved as {META_PREFIX}compute_time_histogram.png")

plt.show()
