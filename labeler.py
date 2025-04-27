import csv
import cv2
from os.path import isfile
import argparse


action_lookup = {
    48: 0,  # No one
    49: 1,  # Cooking
    50: 2,  # Cleaning
    51: 3,  # Just in kitchen
    52: 4,  # Both in kitchen
}

action_verbose = {
    48: "No one in kitchen",  # No one
    49: "Cooking in the kitchen",  # Cooking
    50: "Cleaning in the kitchen",  # Cleaning
    51: "Hanging in the kitchen",  # Just in kitchen
    52: "Both in the kitchen",  # Both in kitchen
}

person_lookup = {
    48: 0,  # Oscar
    49: 1,  # Pontus
}

person_verbose = {
    48: "Oscar",  # Oscar
    49: "Pontus",  # Pontus
}


def main(action):
    filename = "uploads/labels.csv"
    print(action)

    img_id = 0

    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            img_id += 1

    with open(filename, "a") as csv_file:
        writer = csv.writer(csv_file)

        while True:
            if isfile(f"uploads/image-{img_id}.jpg"):
                img = cv2.imread(f"uploads/image-{img_id}.jpg")
                cv2.imshow("image", img)
                key = cv2.waitKey(0)
                writer.writerow([img_id, action_lookup[key] if action else person_lookup[key]])
                print(f"Wrote {action_verbose[key] if action else person_verbose[key]}")
                img_id += 1
            else:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label images")
    parser.add_argument(
        "-a", "--action", action="store_true"
    )
    args = parser.parse_args()
    main(args.action)
