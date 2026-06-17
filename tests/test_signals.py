from src.signals import compute_market_signal


def test_market_signal_score_range():
    result = compute_market_signal(
        last_price=100,
        rsi=55,
        ma20=95,
        ma50=90,
        volatility=0.02,
        model_probability=0.60,
        sentiment_score=50
    )

    assert 0 <= result["market_signal_score"] <= 100


def test_market_signal_output_keys():
    result = compute_market_signal(
        last_price=100,
        rsi=55,
        ma20=95,
        ma50=90,
        volatility=0.02,
        model_probability=0.60,
        sentiment_score=50
    )

    expected_keys = {
        "market_signal_score",
        "signal",
        "interpretation",
        "model_probability_score",
        "technical_score",
        "sentiment_score",
        "volatility_score",
    }

    assert expected_keys.issubset(result.keys())
