import os
import base64
import requests
import datetime
import webbrowser
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
assert CLIENT_ID is not None
assert CLIENT_SECRET is not None
PORT = 7390
REDIRECT_URL = f"http://localhost:{PORT}"


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urlparse(self.path).query
        query_components = parse_qs(query)
        code = query_components.get('code', [''])[0]  # Get 'code' parameter, default to '' if not present
        self.server.code = code
        response = f'Received code: {code}'
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(response.encode())


class Auth:

    def __init__(self, refresh_time=2*60) -> None:
        self.refresh_time = refresh_time
        self._token = None
        self._refresh_token = None
        self._expires = None
        self._timestamp = None

    @property
    def _b64_auth_string(self):
        return base64.b64encode((CLIENT_ID + ':' + CLIENT_SECRET).encode("ascii")).decode("ascii")

    def login(self):
        # step 1
        auth_url = f"https://accounts.spotify.com/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URL}"
        webbrowser.open(auth_url, new=1, autoraise=True)
        
        with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
            httpd.handle_request()
            code = httpd.code
            print("Code received", code)
        # step 2
        r = requests.post(
            url="https://accounts.spotify.com/api/token",
            data={
                "code": code,
                "redirect_uri": REDIRECT_URL,
                "grant_type": "authorization_code"
            },
            headers={
                "Authorization": f"Basic {self._b64_auth_string}"
            }
        )
        data = r.json()
        self._token = data["access_token"]
        self._refresh_token = data["refresh_token"]
        self._expires = int(data["expires_in"])
        self._timestamp = datetime.datetime.now()

    def refresh(self):
        r = requests.post(
            url="https://accounts.spotify.com/api/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token
            },
            headers={
                "Authorization": f"Basic {self._b64_auth_string}"
            }
        )
        if r.status_code != 200:
            self.login()
        data = r.json()
        self._token = data["access_token"]
        self._expires = int(data["expires_in"])
        self._timestamp = datetime.datetime.now()

    def _expires_soon(self):
        dt = (datetime.datetime.now() - self._timestamp).total_seconds()
        remaining = self._expires - dt
        return remaining <= self.refresh_time

    @property
    def token(self):
        if self._token is None:
            self.login()
        if self._expires_soon():
            self.refresh()
        return self._token