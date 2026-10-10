"""Production ASGI entrypoint for FraudShield."""

from fraudshield.production.bootstrap import create_production_app

app = create_production_app()
