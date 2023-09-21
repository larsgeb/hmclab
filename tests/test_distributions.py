from hmclab.Distributions.base import AbstractTargetDistribution
import pytest, numpy as np

distributions = AbstractTargetDistribution.__subclasses__()


def test_list_all_distributions():
    print()
    print(f"All distributions that will be tested:")
    print(distributions)


# Parametrize the distribution classes for log_prob test
@pytest.mark.parametrize("distribution_class", distributions)
def test_log_prob_exist_and_run(distribution_class):
    dimensionality = 10
    distribution_instance = distribution_class.create_default(dimensionality)

    assert hasattr(distribution_instance, "log_prob")

    # Check if log_prob method is callable
    assert callable(getattr(distribution_instance, "log_prob"))

    # Run the log_prob method and check if it returns without errors
    try:
        x = np.random.randn(dimensionality)
        log_prob_result = distribution_instance.log_prob(x)
    except Exception as e:
        # If an exception is raised during method execution, fail the test
        assert (
            False
        ), f"Error running log_prob for {distribution_class}: {str(e)}"


# Parametrize the distribution classes for log_prob_grad test
@pytest.mark.parametrize("distribution_class", distributions)
def test_log_prob_grad_exist_and_run(distribution_class):
    dimensionality = 10
    distribution_instance = distribution_class.create_default(dimensionality)

    assert hasattr(distribution_instance, "log_prob_grad")

    # Check if log_prob_grad method is callable
    assert callable(getattr(distribution_instance, "log_prob_grad"))

    # Run the log_prob_grad method and check if it returns without errors
    try:
        x = np.random.randn(dimensionality)
        log_prob_grad_result = distribution_instance.log_prob_grad(x)
    except Exception as e:
        # If an exception is raised during method execution, fail the test
        assert (
            False
        ), f"Error running log_prob_grad for {distribution_class}: {str(e)}"
