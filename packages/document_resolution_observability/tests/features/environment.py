from test_kit.behave_env import (
    after_all as _after_all,
)
from test_kit.behave_env import (
    after_feature as _after_feature,
)
from test_kit.behave_env import (
    after_scenario as _after_scenario,
)
from test_kit.behave_env import (
    before_all as _before_all,
)
from test_kit.behave_env import (
    before_feature as _before_feature,
)
from test_kit.behave_env import (
    before_scenario as _before_scenario,
)
from test_kit.behave_env import (
    teardown_database as _teardown_database,
)

SERVICE_NAME = "document_resolution_observability"

def before_all(context):
    _before_all(context, service_name=SERVICE_NAME)

def before_feature(context, feature):
    _before_feature(context, feature, service_name=SERVICE_NAME)

def before_scenario(context, scenario):
    _before_scenario(context, scenario, service_name=SERVICE_NAME)

def after_scenario(context, scenario):
    _after_scenario(context, scenario, service_name=SERVICE_NAME)

def after_feature(context, feature):
    _after_feature(context, feature, service_name=SERVICE_NAME)

def after_all(context):
    _after_all(context, service_name=SERVICE_NAME)

def teardown_database(context):
    _teardown_database(context, service_name=SERVICE_NAME)
