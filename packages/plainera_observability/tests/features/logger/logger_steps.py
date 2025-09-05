from behave import given, then, when
from behave.runner import Context

from observability.observability.decorator import logger
from tests.features.logger_io import log_exists, save_to_db


# --- Given/When state ---
@given("a function named {func_name}")
def _given_func(context: Context, func_name: str):
    context.func_name = func_name

@given("a log message '{msg}' is prepared for logging")
def _given_msg(context: Context, msg: str):
    context.log_msg = msg

@when("I call save_to_db with the function name, a log message and the log level, {level} level")
def _when_save(context: Context, level: str):
    save_to_db(context.db_manager, table="logging.logger",
               level=level, function=context.func_name, message=context.log_msg)

@then("a log entry should be saved in the database")
def _then_row_saved(context: Context):
    assert log_exists(context.db_manager, function=getattr(context, "func_name", None)), "No log entry found"

@then("the log level should be {level} and function name is {func_name}")
def _then_level_and_func(context: Context, level: str, func_name: str):
    assert log_exists(context.db_manager, level=level, function=func_name), "Expected level/func not found"

# --- Decorated function flows ---
@given("a decorated function '{func_label}' with a log message 'Executing function'")
def _given_decorated_before(context: Context, func_label: str):
    def _save(level: str, fn: str, msg: str, details: dict):
        save_to_db(context.db_manager, table="logging.logger", level=level, function=fn, message=msg, details=details)
    @logger(save=_save, log_before=True)
    def decoratedFunction():
        return None
    context.function = decoratedFunction

@given("a decorated function '{func_label}' with arguments 'user, password'")
def _given_decorated_with_args(context: Context, func_label: str):
    def _save(level: str, fn: str, msg: str, details: dict):
        save_to_db(context.db_manager, table="logging.logger", level=level, function=fn, message=msg, details=details)
    @logger(save=_save, log_before=True, redact=("password",))
    def secure_function(user: str, password: str):
        return "ok"
    context.function = secure_function

@when("I call the decorated function")
def _when_call_decorated(context: Context):
    context.function()  # type: ignore[misc]

@when("I call the decorated function with user='admin' and password='supersecret'")
def _when_call_decorated_args(context: Context):
    context.function(user="admin", password="supersecret")  # type: ignore[misc]

@then("a log entry with 'Executing function' should be saved in the database before the function execution")
def _then_before_log(context: Context):
    assert log_exists(context.db_manager, substr="Executing function"), "Missing pre-execution log"

@then("a log entry with 'admin' should be saved in the database")
def _then_includes_admin(context: Context):
    assert log_exists(context.db_manager, substr="admin"), "Missing username in log"

@then("the log entry should include 'password': '[REDACTED]'")
def _then_redacted(context: Context):
    # Check JSON details stored contains '[REDACTED]' text in message/details
    assert log_exists(context.db_manager, substr="[REDACTED]"), "Password was not redacted"

# --- Result and duration ---
@given('a decorated function \'timed_function\' that returns "Hello"')
def _given_timed(context: Context):
    def _save(level: str, fn: str, msg: str, details: dict):
        save_to_db(context.db_manager, table="logging.logger", level=level, function=fn, message=msg, details=details)
    @logger(save=_save, log_result=True, log_duration=True)
    def timed_function():
        return "Hello"
    context.function = timed_function

@then('the log entry should contain "Result: \'Hello\'"')
def _then_has_result(context: Context):
    assert log_exists(context.db_manager, substr="Hello"), "Result not logged"

@then('the log entry should contain "Duration"')
def _then_has_duration(context: Context):
    assert log_exists(context.db_manager, substr="duration"), "Duration not logged"

# --- Exception path ---
@given("a decorated function 'failing_function' that raises an exception")
def _given_failing(context: Context):
    def _save(level: str, fn: str, msg: str, details: dict):
        save_to_db(context.db_manager, table="logging.logger", level=level, function=fn, message=msg, details=details)
    @logger(save=_save)
    def failing_function():
        raise RuntimeError("boom")
    context.function = failing_function

@when("I call the function")
def _when_call_plain(context: Context):
    try:
        context.function()  # type: ignore[misc]
    except Exception:
        context._raised = True

@then("an error log entry should be saved in the database")
def _then_error_log(context: Context):
    assert log_exists(context.db_manager, level="error"), "Error log not found"

@then('the log message should contain "Exception in failing_function"')
def _then_contains_exc(context: Context):
    assert log_exists(context.db_manager, substr="Exception in failing_function"), "Exception message not logged"

# --- Invalid table path ---
@given("an invalid table name '{fqn}'")
def _given_invalid_table(context: Context, fqn: str):
    context.invalid_table = fqn

@when("I call save_to_db with this table")
def _when_save_invalid(context: Context):
    context._saved_ok = False
    try:
        save_to_db(context.db_manager, table=context.invalid_table, level="INFO", function="x", message="y")
        context._saved_ok = True
    except Exception:
        pass

@then("a ValueError should be raised")
def _then_value_error(context: Context):
    assert not context._saved_ok, "Expected failure when writing to invalid table"

@then("no entry should be saved")
def _then_no_entry(context: Context):
    assert not log_exists(context.db_manager, function="x", level="info"), "Unexpected row inserted"
