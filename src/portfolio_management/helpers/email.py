"""Email notification helpers using Microsoft Graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from msal import ConfidentialClientApplication

from .config import EMAIL_CONFIG
from .http import get_requests_verify

logger = logging.getLogger(__name__)

class _RequestsHttpClient:
    """MSAL-compatible HTTP client that uses our CA bundle resolver."""

    def __init__(self, *, verify: bool | str):
        self._verify = verify
        self._session = requests.Session()

    def get(self, url: str, **kwargs):
        kwargs.setdefault("verify", self._verify)
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs):
        kwargs.setdefault("verify", self._verify)
        return self._session.post(url, **kwargs)


@dataclass
class EmailClient:
    tenant_id: str | None = EMAIL_CONFIG.tenant_id
    client_id: str | None = EMAIL_CONFIG.client_id
    client_secret: str | None = EMAIL_CONFIG.client_secret
    sender: str | None = EMAIL_CONFIG.sender
    recipient: str | None = EMAIL_CONFIG.recipient

    def enabled(self) -> bool:
        fields = [
            self.tenant_id,
            self.client_id,
            self.client_secret,
            self.sender,
            self.recipient,
        ]
        return all(fields)

    def send(self, subject: str, body: str, *, content_type: str = "Text") -> None:
        if not self.enabled():
            logger.info("Microsoft Graph configuration incomplete; skipping email.")
            return

        try:
            verify = get_requests_verify()
            app = ConfidentialClientApplication(
                self.client_id,
                authority=f"https://login.microsoftonline.com/{self.tenant_id}",
                client_credential=self.client_secret,
                http_client=_RequestsHttpClient(verify=verify),
            )
            token_result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            access_token = token_result.get("access_token")
            if not access_token:
                logger.error("Failed to acquire Graph access token: %s", token_result)
                return

            payload = {
                "message": {
                    "subject": subject,
                    "body": {"contentType": content_type, "content": body},
                    "toRecipients": [{"emailAddress": {"address": self.recipient}}],
                    "from": {"emailAddress": {"address": self.sender}},
                },
                "saveToSentItems": "false",
            }
            resp = requests.post(
                f"https://graph.microsoft.com/v1.0/users/{self.sender}/sendMail",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
                verify=verify,
            )
            if resp.status_code != 202:
                logger.error(
                    "Graph send failed (%s): %s", resp.status_code, resp.text[:2000]
                )
            else:
                logger.info("Graph email sent successfully.")
        except Exception as exc:  # pragma: no cover - safety net
            logger.exception("Graph email send error: %s", exc)

    def send_html(self, subject: str, html_body: str) -> None:
        self.send(subject, html_body, content_type="HTML")
