.PHONY: core api
core:
	$(MAKE) -C packages/plainera_core $(t)

api:
	$(MAKE) -C services/public_api $(t)
