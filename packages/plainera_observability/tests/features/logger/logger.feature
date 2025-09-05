Feature: Database Logging System
  As a user I want to check that a message is logged when using the logger.

  Background:
    Given I setup a temporary database called logger
    And I setup a table called logging.logger

  Scenario: Successfully log a message with default info level

    Given a function named successful_function
    And a log message 'Test log message' is prepared for logging
    When I call save_to_db with the function name, a log message and the log level, INFO level
    Then a log entry should be saved in the database
    And the log level should be info and function name is successful_function

  Scenario: Successfully log a message with a specified error level
    Given a function named error_function
    And a log message 'Error occurred' is prepared for logging
    When I call save_to_db with the function name, a log message and the log level, ERROR level
    Then a log entry should be saved in the database
    And the log level should be error and function name is error_function

  Scenario: Successfully log a message with a specified error level
    Given a function named debug_function
    And a log message 'Debugging..' is prepared for logging
    When I call save_to_db with the function name, a log message and the log level, DEBUG level
    Then a log entry should be saved in the database
    And the log level should be debug and function name is debug_function

  Scenario: Log a message before function execution
    Given a decorated function 'decoratedFunction' with a log message 'Executing function'
    When I call the decorated function
    Then a log entry with 'Executing function' should be saved in the database before the function execution

  Scenario: Redacted arguments should not be logged
    Given a decorated function 'secure_function' with arguments 'user, password'
    And the 'password' argument should be redacted
    When I call the decorated function with user='admin' and password='supersecret'
    And I print the contents of the logger table
    Then a log entry with 'admin' should be saved in the database
    And the log entry should include 'password': '[REDACTED]'

  Scenario: Duration and result should be logged when enabled
    Given a decorated function 'timed_function' that returns "Hello"
    And the decorator is configured to log result and duration
    When I call the decorated function
    And I print the contents of the logger table
    Then the log entry should contain "Result: 'Hello'"
    And the log entry should contain "Duration"

  Scenario: Logging should capture an exception
    Given a decorated function 'failing_function' that raises an exception
    When I call the function
    Then an error log entry should be saved in the database
    And the log message should contain "Exception in failing_function"

  Scenario: Invalid table name triggers ValueError
    Given an invalid table name 'fake_schema.fake_table'
    When I call save_to_db with this table
    Then a ValueError should be raised
    And no entry should be saved
