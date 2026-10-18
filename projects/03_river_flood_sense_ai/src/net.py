"""Small retrying HTTP helper. The Open-Meteo / GCS endpoints occasionally drop TLS
connections from this network, so every request gets bounded retries with backoff."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "river-flood-sense-ai/1.0 (educational demo)"}


def get_json(url, params=None, tries=6, timeout=120, pause=2.0):
    if params:
        url = url + "?" + urllib.parse.urlencode(params, safe=",")
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code == 429:
                # Open-Meteo enforces a per-minute quota; wait it out rather than fail.
                last = f"HTTP 429: {body}"
                print(f"    rate-limited, sleeping 65s (attempt {attempt + 1}/{tries})")
                time.sleep(65)
                continue
            # Other 4xx means a malformed request - retrying will not help.
            if 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code} for {url[:140]}\n{body}") from None
            last = f"HTTP {e.code}: {body}"
        except Exception as e:  # noqa: BLE001 - transient TLS / socket errors
            last = f"{type(e).__name__}: {e}"
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {url[:140]}\nlast error: {last}")


def download_file(url, dest, tries=5, timeout=180, pause=2.0):
    """Download to `dest` (pathlib.Path). Returns True if written, False if it existed."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            tmp.unlink(missing_ok=True)
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed to download {url[:140]}\nlast error: {last}")


def gcs_list(bucket, prefix, delimiter=None):
    """List a public GCS bucket over plain HTTPS (no gsutil / credentials needed)."""
    base = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    items, prefixes, token = [], [], None
    while True:
        p = {"prefix": prefix, "maxResults": 1000}
        if delimiter:
            p["delimiter"] = delimiter
        if token:
            p["pageToken"] = token
        d = get_json(base, p)
        items += [(i["name"], int(i["size"])) for i in d.get("items", [])]
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            break
    return items, prefixes
