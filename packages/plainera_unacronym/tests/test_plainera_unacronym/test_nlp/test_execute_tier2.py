from plainera_unacronym.nlp.execute import detect_and_extract


class TestDetectAndExtractE2ETier2MixedCaseAcronyms:

    def test_stylised_ios_parenthetical(self):
        pass
        # det, extr = detect_and_extract("iOS (iPhone Operating System) is supported.")
        # assert picked_def(extr, "iOS") in {"iPhone Operating System"}, extr.picks.get("iOS")

    def test_stylised_ebay_parenthetical(self):
        pass
        # det, extr = detect_and_extract("eBay (electronic Bay) is a marketplace.")
        # assert picked_def(extr, "eBay") in {"electronic Bay"}, extr.picks.get("eBay")

    def test_stylised_latex_parenthetical(self):
        pass
        # det, extr = detect_and_extract("LaTeX (Lamport TeX) is used for typesetting.")
        # assert picked_def(extr, "LaTeX") in {"Lamport TeX"}, extr.picks.get("LaTeX")
