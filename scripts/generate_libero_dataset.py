import argparse

from src.data.dataset_generator import generate_dataset


def main():

    parser = argparse.ArgumentParser(
        description="Generate the LIBERO S1 decision dataset."
    )

    parser.add_argument(
        "--task-ids",
        nargs="+",
        type=int,
        default=[0],
    )

    parser.add_argument(
        "--num-initial-states",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--magnitude",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/libero_s1_pilot",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    generate_dataset(
        task_ids=args.task_ids,
        num_initial_states=args.num_initial_states,
        translation_magnitude=args.magnitude,
        horizon_steps=args.horizon,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
