"""
agent/core/deactivation.py
--------------------------
Shared, empathetic, reason-aware account-deactivation messaging.

Every post-login flow (statement, brokerage, order status, account details,
IPO, login query) hits the same situation: the customer's account is disabled
or purged, so the requested action can't proceed. Instead of each flow showing
its own generic one-liner, they all call `deactivation_message()` here — which
returns a warm, multi-line message that states the SPECIFIC reason and the
reactivation steps, per the flowchart ("show de-active message with reason
along with process for reactivation", "empathetic and not a one liner").

Codes mirror thor's activation_agent.py (entStatusLov deactivation codes):
  - AUTOMATED codes  → specific reason + self-serve reactivation step/link.
  - ESCALATE codes   → no self-serve path; apologise + point to a support ticket.
  - unknown / purged → empathetic generic reason + support contact.
"""

from __future__ import annotations

# ── Links / contacts (aligned with thor's activation drafts) ─────────────────
_LOGIN_PORTAL     = "https://login.axisdirect.in/?redirectPage=My%20Page"
_DORMANT_ACTIVATE = "https://simplehai.axisdirect.in/"
_FATCA_FORM       = "https://simplehai.axisdirect.in/images/NSDL_Form/Individual---FATCA.pdf"
_TICKET_LINK      = "https://simplehai.axisdirect.in/portal/index.php/supportPortal/raise-query"
_CARE_NUMBERS     = "📞 022-40508080 / 022-61480808 / 022-68515400 · ✉️ helpdesk@axisdirect.in"
# Thor's support hours — shown with the support-contact branches.
_SUPPORT_HOURS    = ("🕓 Support hours: Mon–Fri 8:45 AM – 5:30 PM IST; "
                     "Sat (1st, 3rd, 5th) 10:00 AM – 1:00 PM IST")

# ── Escalate codes: no automated self-serve path (thor-aligned) ──────────────
ESCALATE_CODES: frozenset[str] = frozenset({
    "D004",  # Client Request
    "D006",  # Other
    "D008",  # Physical AOF not received
    "D012",  # NC Welcome Calling
    "D013",  # Mis-sell Welcome Calling
    "D014",  # Incorrect/Incomplete documents
    "D016",  # Demat Account Suspended
    "D017",  # Demat account not activated
    "D020",  # Fund Payout Failed
    "D021",  # Karvy Overlapping
    "D022",  # To be purged
    "D025",  # PEP Confirmation not received
})

# ── Automated codes → (reason, reactivation step incl. link) ─────────────────
# One place to maintain the reason + primary reactivation process per code.
# Reasons/links mirror thor's activation drafts (agent/prompts/email_templates.py),
# condensed to the primary self-serve action suitable for a chat message.
CODE_MESSAGES: dict[str, tuple[str, str]] = {
    "D005": ("your KYC attributes are not yet updated",
             f"Log in and complete the Digilocker + Aadhaar e-sign KYC update here: {_LOGIN_PORTAL}"),
    "D007": ("the 6 mandatory KYC attributes are not updated on your account",
             f"Log in and complete the Digilocker + Aadhaar e-sign KYC update here: {_LOGIN_PORTAL}"),
    "D019": ("a FATCA declaration is pending on your account",
             f"Log in and accept the FATCA declaration pop-up to activate within 1 working day: {_LOGIN_PORTAL}\n"
             f"Offline FATCA form: {_FATCA_FORM}"),
    "D024": ("your account has been dormant for 24+ months (NSE guidelines)",
             f"Reactivate your dormant account online here: {_DORMANT_ACTIVATE}"),
    "D002": ("your linked demat account is closed",
             f"Reactivate by opening a new demat account or linking an existing one. "
             f"Start here: {_LOGIN_PORTAL}"),
    "D010": ("your email ID or mobile number is used on more than one account "
             "(NSDL/POLICY/2018/0031)",
             f"Log in to 'My Profile' and update your contact details via Aadhaar e-sign: {_LOGIN_PORTAL}"),
    "D003": ("your linked bank account is closed",
             f"Log in → My Profile → Account Details → Modify Bank to link an active account: {_LOGIN_PORTAL}"),
    "D018": ("an NRI bank account is linked to your Resident Indian account",
             f"Log in → My Profile → Account Details → Modify Bank to link an RI account: {_LOGIN_PORTAL}"),
    "D009": ("your Account Opening Charges (AOC) are pending / Re-KYC is required",
             f"Please complete the Re-KYC/activation by logging in here: {_LOGIN_PORTAL}"),
}


def _wrap(name: str, body: str, closing: str) -> str:
    """Compose the empathetic multi-line envelope: warm opener → body → closing."""
    greeting = f"Hi {name}," if name else "Hi,"
    return (
        f"{greeting}\n\n"
        f"We're sorry for the inconvenience — {body}\n\n"
        f"{closing}\n\n"
        "Once your account is active again, you'll be able to continue right here. "
        "We appreciate your patience and are happy to help if you need anything."
    )


def deactivation_message(code: str | None, name: str = "", *, purged: bool = False) -> str:
    """
    Build an empathetic, reason-aware deactivation message.

    Args:
        code:   the deactivation reason code (entStatusLov), e.g. "D002". May be
                empty/None if the API didn't return one.
        name:   customer name for a personal greeting (optional).
        purged: True when the account status is "purged" (terminal).

    Returns a warm, multi-line message stating the reason + reactivation steps.
    """
    c = (code or "").strip().upper()

    # Purged / terminal — no self-serve reactivation.
    if purged:
        return _wrap(
            name,
            "your account appears to be permanently closed, so we're unable to "
            "process this request.",
            f"Our team can help you with the next steps. Please reach out:\n{_CARE_NUMBERS}\n"
            f"Or raise a request here: {_TICKET_LINK}\n{_SUPPORT_HOURS}",
        )

    # Automated code — specific reason + self-serve reactivation step.
    if c in CODE_MESSAGES:
        reason, step = CODE_MESSAGES[c]
        return _wrap(
            name,
            f"your account is currently deactivated because {reason}, so we can't "
            "complete this request just yet.",
            step,
        )

    # Escalate code — reason not self-serviceable; point to a support ticket.
    if c in ESCALATE_CODES:
        return _wrap(
            name,
            "your account is currently deactivated and this needs a quick review "
            "by our team before it can be reactivated.",
            f"Please raise a request and our team will assist you:\n{_TICKET_LINK}\n{_CARE_NUMBERS}\n{_SUPPORT_HOURS}",
        )

    # Unknown / missing code — empathetic generic fallback with support contact.
    return _wrap(
        name,
        "your account is currently deactivated, so we're unable to complete this "
        "request right now.",
        f"Our support team can help you reactivate it:\n{_CARE_NUMBERS}\n"
        f"Or raise a request here: {_TICKET_LINK}\n{_SUPPORT_HOURS}",
    )
