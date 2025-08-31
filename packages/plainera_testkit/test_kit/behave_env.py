"""
This is used to hold behave automated funcs such as 'after_scenario' for example.
"""

from behave.runner import Context



def before_all(context, *, service_name: str | None = None):
    """
    This function is executed once before any feature files are run.

    It initializes the resource loader and loads resources if required by any feature.
    """
    pass

def before_feature(context, feature, *, service_name: str | None = None):
    """
    This function is executed before each feature file is run.

    It ensures resources are available for features tagged with specific tags.
    """
    pass


def before_scenario(context: Context, scenario, *, service_name: str | None = None):
    """
    This function is executed before each scenario is run.

    It ensures that everything is fresh for each scenario, so that it is ran
    independently.

    Args:
        context (Context): The Behave context object containing scenario-specific data.
        scenario: The scenario but before the execution.
    """
    pass


def after_scenario(context: Context, scenario, *, service_name: str | None = None):
    """Executed after each scenario.

    This function tears down specified elements if they exist0 in the context
    after each scenario.

    Args:
        context (Context): The Behave context object containing scenario-specific data.
        scenario: The scenario that just finished execution.
    """
    pass

def teardown_database(context: Context, *, service_name: str | None = None):
    """Teardown the temporary database.

    This function disconnects from the current database manager, connects to the default
    'postgres' database, and drops the temporary database.

    Args:
        context (Context): The Behave context object containing database connection details.
    """
    pass

def after_all(context: Context, *, service_name: str | None = None):
    """
    Perform cleanup activities after all tests have been executed.

    This function is designed to execute cleanup or teardown procedures
    after the entire suite of tests has been run. It ensures that the
    necessary final actions are completed before ending the test session.

    Parameters:
    context: Context
        The test context carrying information shared across all test steps
        and scenarios. It provides access to various resources used during
        the testing process.
    """
    pass
