"""Fetch versioned NLTK data during image build, never service startup."""
import hashlib
from io import BytesIO
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

REVISION = '550b6625bcef1f2abff2ff770a5a0d272c9c6b2a'
ASSETS = {
    'punkt': '51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec',
    'punkt_tab': 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106',
}

def main(destination):
    target = Path(destination) / 'tokenizers'
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        url = f'https://raw.githubusercontent.com/nltk/nltk_data/{REVISION}/packages/tokenizers/{name}.zip'
        with urlopen(url, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f'NLTK {name} checksum mismatch')
        with ZipFile(BytesIO(payload)) as archive:
            archive.extractall(target)
        print(f'Verified and installed NLTK {name}; upstream README included.')

if __name__ == '__main__':
    main(sys.argv[1])
