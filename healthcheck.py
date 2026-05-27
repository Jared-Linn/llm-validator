"""Docker healthcheck: verify the API is responding."""
import http.client
import sys

try:
    c = http.client.HTTPConnection("localhost", 8900, timeout=5)
    c.request("GET", "/api/status")
    r = c.getresponse()
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
