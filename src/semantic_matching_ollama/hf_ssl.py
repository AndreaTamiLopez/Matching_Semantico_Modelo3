def enable_hf_ssl_fix() -> str:
    """
    Si te da SSLCertVerificationError al bajar modelos de HuggingFace,
    ejecuta esto ANTES de crear SentenceTransformer.
    """
    import os
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    return certifi.where()
