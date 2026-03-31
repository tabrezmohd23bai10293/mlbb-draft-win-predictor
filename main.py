from src.predict import predict_draft


def main():
    blue_team = ["Karrie", "Fredrinn", "Lunox", "Badang", "Arlott"]
    red_team = ["Ling", "Esmeralda", "Atlas", "Beatrix", "Paquito"]

    result = predict_draft(blue_team, red_team)

    print("=" * 60)
    print("MLBB Draft Win Prediction")
    print("=" * 60)
    print(f"Blue Team : {', '.join(blue_team)}")
    print(f"Red Team  : {', '.join(red_team)}")
    print("-" * 60)
    print(f"Predicted Winner     : {result['predicted_winner'].upper()}")
    print(f"Blue Win Probability : {result['blue_win_probability'] * 100:.2f}%")
    print(f"Red Win Probability  : {result['red_win_probability'] * 100:.2f}%")


if __name__ == "__main__":
    main()
