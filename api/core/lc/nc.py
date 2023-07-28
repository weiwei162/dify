from sqlalchemy import create_engine, text
import os
import json
import base64
import hashlib
from Crypto.Cipher import AES


def get_db_uri(id: str) -> str:
    engine = create_engine(
        os.environ.get("NC_DB")
    )
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT * FROM nc_bases_v2 where id = :id'), {"id": id}
        ).first()

    config = json.loads(decrypt_aes_ciphertext(result.config))['connection']
    db_type = {
        'oracle': 'oracle+cx_oracle',
        'pg': 'postgresql',
        'mysql2': 'mysql+pymysql'
    }.get(result.type)

    uri = f"{db_type}://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    return uri


def decrypt_aes_ciphertext(ciphertext_base64):
    password = os.environ.get("NC_AUTH_JWT_SECRET").encode('ASCII')
    ciphertext = base64.b64decode(ciphertext_base64)
    ciphertext_hex = ciphertext.hex()
    salt = bytes.fromhex(ciphertext_hex[16:32])

    derived_key = b''
    data = b''
    while len(derived_key) < 48:
        md5 = hashlib.md5()
        md5.update(data + password + salt)
        data = md5.digest()
        derived_key += data

    key = derived_key[:32]
    iv = derived_key[32:]

    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(bytes.fromhex(
        ciphertext_hex[32:])).decode('ASCII')
    return plaintext[:-ord(plaintext[-1])]
