from model_builder import T5ModelBuilder


def test_model_builder_starts():
    model_id = "t5-small"
    model_builder = T5ModelBuilder(model_id)
    base_model = model_builder.base_model
    assert base_model is not None
