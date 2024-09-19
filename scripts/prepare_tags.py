"Script for preparing and splitting tags data"

import os
import random

ALL_TAGS_PATH = "../data/tags.txt"
UNIVERSAL_TAGS_PATH = "../data/universal_tags.txt"
EVAL_TAGS_PATH = "../data/eval_tags.txt"
TEST_TAGS_PATH = "../data/test_tags.txt"

if (
    os.path.exists(UNIVERSAL_TAGS_PATH) and
    os.path.exists(EVAL_TAGS_PATH) and
    os.path.exists(TEST_TAGS_PATH)
):
    print("Tags already split.")
    with open(UNIVERSAL_TAGS_PATH) as file:
        tags_universal = [line.rstrip() for line in file]
        print(f"Loaded {len(tags_universal)} universal tags.")
    with open(EVAL_TAGS_PATH) as file:
        tags_eval = [line.rstrip() for line in file]
        print(f"Loaded {len(tags_eval)} eval tags.")
    with open(TEST_TAGS_PATH) as file:
        tags_test = [line.rstrip() for line in file]
        print(f"Loaded {len(tags_test)} test tags.")

else:
    if os.path.exists(ALL_TAGS_PATH):
        print("Splitting tags. Please relaoad the cell after splits have been created.")
        with open(ALL_TAGS_PATH) as file:
            tags = [line.rstrip() for line in file]
        print(f"Loaded {len(tags)} tags.")

        random_tagging = random.Random(42)
        tags_universal = tags.copy()
        tags_eval = random_tagging.sample(tags_universal, k=10)
        tags_universal = [t for t in tags_universal if t not in tags_eval]
        tags_test = random_tagging.sample(tags_universal, k=10)
        tags_universal = [t for t in tags_universal if t not in tags_test]

        assert set(tags) == set(tags_universal) | set(tags_test) | set(tags_eval)

        with open(UNIVERSAL_TAGS_PATH, "w") as file:
            for tag in tags_universal:
                file.write(f"{tag}\n")
            print("Created universal tags split.")
        with open(EVAL_TAGS_PATH, "w") as file:
            for tag in tags_eval:
                file.write(f"{tag}\n")
            print("Created eval tags split.")
        with open(TEST_TAGS_PATH, "w") as file:
            for tag in tags_test:
                file.write(f"{tag}\n")
            print("Created test tags split.")
    else:
        raise Exception("No tags file found.")
