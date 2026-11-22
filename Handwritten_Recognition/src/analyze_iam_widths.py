
from datasets import load_from_disk
import numpy as np

print("========== IAM IMAGE WIDTH ANALYSIS ==========")

dataset = load_from_disk("data/iam/huggingface_dataset")

TARGET_HEIGHT = 64

for split_name in ["train", "validation", "test"]:
    widths = []

    for sample in dataset[split_name]:
        image = sample["image"].convert("L")

        original_width, original_height = image.size

        # Resize proportionally to height 64
        new_width = round(
            original_width * TARGET_HEIGHT / original_height
        )

        widths.append(new_width)

    widths = np.array(widths)

    print(f"\n{split_name.upper()} SPLIT")
    print("Number of images:", len(widths))
    print("Minimum width:", widths.min())
    print("Maximum width:", widths.max())
    print("Average width:", round(widths.mean(), 2))

    print("Width percentiles:")
    for percentile in [90, 95, 99, 99.5, 100]:
        print(
            f"{percentile}%:",
            round(np.percentile(widths, percentile))
        )

    print("\nImages exceeding possible width limits:")
    for limit in [1024, 1536, 2048, 2560]:
        count = np.sum(widths > limit)
        percentage = count / len(widths) * 100

        print(
            f"{limit}px: {count} images "
            f"({percentage:.2f}%)"
        )

print("\nWidth analysis completed!")
