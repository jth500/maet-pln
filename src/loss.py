def rouge(y_true: str, y_pred: str) -> float:
    """Rouge loss function to be applied element wise.

    Args:
        y_true (str): True value.
        y_pred (str): Predicted value.

    Returns:
        float: Loss value.
    """
    # TODO: Implement rouge loss
    # Not sure if inputs should be strings or tokenised lists?? Haven't looked at this at all
    return None


def eval_loss(y_true: list, y_pred: list, loss_metric="rouge") -> list:
    """Apply chosen loss metric to each element in the batch

    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.

    Returns:
        list: List of losses for each element in the batch.
    """
    # TODO: Vectorize this (if possible)
    loss_funcs = {"rouge": rouge, "other": "TODO"}
    return [loss_funcs[loss_metric](y_true[i], y_pred[i]) for i in range(len(y_true))]


def test_rouge():
    """
    Test the rouge function.

    This function tests the rouge function by comparing the output of the function
    with expected values. It asserts that the calculated rouge score matches the
    expected score for different input strings.

    Returns:
        None
    """
    y_true = "This is a test"
    y_pred = "This is a test"
    assert rouge(y_true, y_pred) == 1.0  # guessing this is the expected value?
    y_pred = "This is not a test"
    assert rouge(y_true, y_pred) == 0.0
    y_pred = ""


if __name__ == "__main__":
    test_rouge()
