import secrets
secret = secrets.token_hex(16)  # 32 characters (128 bits)
print(secret)