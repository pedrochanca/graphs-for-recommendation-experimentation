import pandas as pd


def main():

    df = pd.read_csv("../datasets/ml-latest-small/ratings.csv")
    df.rename(
        columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"},
        inplace=True,
    )


if __name__ == "__main__":
    main()
