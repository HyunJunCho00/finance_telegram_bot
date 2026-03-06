from collectors.coinmetrics_collector import coinmetrics_collector


def main():
    results = coinmetrics_collector.run()
    for symbol, snapshot in results.items():
        print(
            f"{symbol}: risk_bias={snapshot.get('risk_bias')} "
            f"bias_score={snapshot.get('bias_score')} "
            f"as_of_date={snapshot.get('as_of_date')}"
        )


if __name__ == "__main__":
    main()
