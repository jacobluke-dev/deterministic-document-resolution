Feature: Message Logging System
  As a user I want to verify that messages are correctly logged using message_logger

  Background:
    Given I setup a temporary database called message_logger
    And I setup a table called logging.logger

  Scenario: Log a message from the calling function using message_logger
    Given a function 'message_call' that calls message_logger with 'Special message'
    When I call the message function
    Then a log entry with 'MESSAGE LOGGER: called by message_call message is Special message' should be saved in the database
    And the log level should be message and function name is message_call

  Scenario: Log a message from a calling function that raises an Exception and should log that exception
    Given a function that raises an exception
    When I call the exception function
    Then a log entry with 'MESSAGE LOGGER: Exception:division by zero' should be saved in the database
    And the log level should be error and function name is div_zero_function

  Scenario: Inline message_logger captures calling arguments
    Given a function 'caller_function' that uses inline message_logger
    When I call the function with arguments 'x=5' and 'y=10'
    And I print the contents of the logger table
    Then the log entry should include 'x=5'
    And the log entry should include 'y=10'

  Scenario: Unknown log level uses fallback
    Given a function 'funky_level_function'
    When I log a message with an undefined log level
    And I print the contents of the logger table
    Then the level name should be 'UNKNOWN LEVEL' in the database

  Scenario: Log a message where args are not inspectable
    Given a function 'opaque_logger' that logs from a nested lambda
    When I call the message function
    And I print the contents of the logger table
    Then a log entry should be saved
    And the arguments field should contain an error message


  Scenario: Log a message with a custom log level
    Given a function 'custom_logger' that logs with level WARNING
    When I call the message function
    Then a log entry should be saved
    And the log level should be warning and function name is custom_logger
