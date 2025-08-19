from plainera_core.domain.models import Acronym, DefinitionCandidate
from plainera_core.services.resolver import AcronymResolver


def test_resolver_picks_best():
    def mem(_):
        return [
            DefinitionCandidate("National Health Service", 0.9),
            DefinitionCandidate("Non-Hodgkin's Something", 0.1),
        ]
    print("test_resolver_picks_best")
    r = AcronymResolver(mem)
    top = r.resolve(Acronym("NHS"))
    assert top and top[0].text == "National Health Service"
