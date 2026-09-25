"""
chatbot_web/src/flows/how_to_trade/instructions.py
----------------------------------------------------
Static step-by-step trading instructions, keyed by trade_type + app.
Ported from chatbot/src/flows/how_to_trade/handler.py — kept as a
separate module so the handler stays lean.
"""

from __future__ import annotations

# ── Option lists ──────────────────────────────────────────────────────────────

TRADE_TYPES = ["Cash & ETF", "E-Margin & Intraday", "Stop Loss", "Derivatives", "Other"]
BUY_SELL_TYPES = ["BUY", "SELL"]
DERIVATIVE_TYPES = ["Futures", "Options", "FNO Sell"]
OTHER_TYPES = ["Encash", "GTDT", "Intersettlement", "Cover"]
APP_TYPES = ["Traders App", "Investors App", "Swift Trade"]
APP_TYPES_NO_INVESTORS = ["Traders App", "Swift Trade"]   # Cover not on Investors App

# Trade types that need a BUY/SELL sub-selection
BUY_SELL_TRADES: frozenset[str] = frozenset({"Cash & ETF", "E-Margin & Intraday"})

# ── Instructions dict ─────────────────────────────────────────────────────────
# Key format: "<trade_type>" or "<trade_type> BUY/SELL" for multi-direction types

INSTRUCTIONS: dict[str, dict[str, str]] = {
    "Cash & ETF BUY": {
        "Traders App": (
            "1) Search for the desired Stock or ETF in the search tab\n"
            "2) Then click on BUY and enter the quantity\n"
            "3) Choose the product type as delivery and the price as on Market or Limit price — "
            "once desired options are updated click on BUY\n"
            "4) Confirm the order and swipe to buy at the bottom"
        ),
        "Investors App": (
            "1) Search for the desired Stock or ETF in the search tab\n"
            "2) Then click on BUY and enter the quantity\n"
            "3) Choose the order type as delivery and the price as on Market or Limit price — "
            "once desired options are updated click on BUY\n"
            "4) Confirm the order and swipe to buy at the bottom"
        ),
        "Swift Trade": (
            "1) Select CASH in Equity under the BUY column\n"
            "2) Enter the name of the scrip in Company name and fill the other details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "Cash & ETF SELL": {
        "Traders App": (
            "1) Select the desired Stock or ETF in Portfolio or Demat Holdings\n"
            "2) Then click on SELL and enter the quantity\n"
            "3) Choose the product type as delivery and the price as on Market or Limit price — "
            "once desired options are updated click on SELL\n"
            "4) Confirm the order and swipe to sell at the bottom"
        ),
        "Investors App": (
            "1) Select the desired Stock or ETF in Portfolio or Demat Holdings\n"
            "2) Then click on SELL and enter the quantity\n"
            "3) Choose the product type as delivery and the price as on Market or Limit price — "
            "once desired options are updated click on SELL\n"
            "4) Confirm the order and swipe to sell at the bottom"
        ),
        "Swift Trade": (
            "1) Select the desired Stock or ETF in the Demat balance under Reports\n"
            "2) Then click on SELL and enter the quantity\n"
            "3) Choose the product type as delivery and the price as on Market or Limit price — "
            "once desired options are updated click on SELL\n"
            "4) Confirm the order and swipe to sell at the bottom"
        ),
    },
    "E-Margin & Intraday BUY": {
        "Traders App": (
            "1) Search for your desired stock in the search tab\n"
            "2) Click on BUY and select the product as 'Intraday' or 'E-Margin'\n"
            "3) Enter the quantity and click on BUY at the bottom\n"
            "4) Confirm the order and swipe to buy"
        ),
        "Investors App": (
            "1) Search for your desired stock in the search tab\n"
            "2) Click on BUY and select the order type as 'Intraday' or 'E-Margin'\n"
            "3) Enter the quantity and click on BUY at the bottom\n"
            "4) Confirm the order and swipe to buy"
        ),
        "Swift Trade": (
            "1) Select Intraday or E-Margin in Equity under the BUY column\n"
            "2) Enter the scrip name and fill the other order details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "E-Margin & Intraday SELL": {
        "Traders App": (
            "1) Search for your stock in Open Positions in Portfolio\n"
            "2) Click on SELL and select 'Intraday' or 'E-Margin'\n"
            "3) Enter the quantity and click on SELL at the bottom\n"
            "4) Confirm the order and swipe to sell"
        ),
        "Investors App": (
            "1) Search for your stock in Positions in Portfolio\n"
            "2) Click on BUY and select 'Intraday' or 'E-Margin'\n"
            "3) Enter the quantity and click on BUY at the bottom\n"
            "4) Confirm the order and swipe to buy"
        ),
        "Swift Trade": (
            "1) Select your stock in Open Positions under the Reports column\n"
            "2) Enter the quantity and the order details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "Stop Loss": {
        "Traders App": (
            "1) Login to Axis Direct Traders app\n"
            "2) Click on your initials (top left) → Portfolio Statements → Demat Holdings\n"
            "3) Select the scrip you wish to square off\n"
            "4) Choose the exchange and click on 'Square off'\n"
            "5) Enter quantity and click on Stop Loss to set the Trigger Price\n"
            "6) Click the red 'Square off' tab → confirm → swipe to square off"
        ),
        "Investors App": (
            "1) Login to Axis Direct Investors app\n"
            "2) Click on your initials (top left) → Reports → Demat Holdings\n"
            "3) Select the scrip you wish to square off\n"
            "4) Choose the exchange and click 'Square off'\n"
            "5) Enter quantity → Advanced options → add Trigger Price\n"
            "6) Confirm the order and swipe to square off"
        ),
        "Swift Trade": (
            "1) Login → SWIFT TRADE → Equity → Demat Balance (Reports)\n"
            "2) Click the red '-' next to your holding\n"
            "3) Choose Square off, set Exchange, Scrip, Quantity, Price, and Trigger Price\n"
            "4) Click PLACE ORDER → Confirm Order"
        ),
    },
    "Futures": {
        "Traders App": (
            "1) Search for the desired contract in the search tab\n"
            "2) Click on BUY → enter quantity → choose Margin, Intraday or Cover\n"
            "   Set Market or Limit price → click BUY\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        "Investors App": (
            "1) Search for the desired contract in the search tab\n"
            "2) Click on BUY → enter quantity → choose Margin, Intraday or Cover\n"
            "   Set Market or Limit price → click BUY\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        "Swift Trade": (
            "1) Select Future Index or Future Stock under the F&O column\n"
            "2) Enter the scrip name and fill the order details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "Options": {
        "Traders App": (
            "1) Search for the desired options contract in the search tab\n"
            "2) Click on BUY → enter quantity → choose Margin, Intraday or Cover\n"
            "   Set Market or Limit price → click BUY\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        "Investors App": (
            "1) Search for the desired options contract in the search tab\n"
            "2) Click on BUY → enter quantity → choose Margin, Intraday or Cover\n"
            "   Set Market or Limit price → click BUY\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        "Swift Trade": (
            "1) Select Future Index or Future Stock under the F&O column\n"
            "2) Enter the scrip name and fill the order details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "FNO Sell": {
        "Traders App": (
            "1) Find your contract in Open Positions in Portfolio\n"
            "2) Click on SELL and enter the order details\n"
            "3) Enter the quantity and click on SELL at the bottom\n"
            "4) Confirm the order and swipe to sell"
        ),
        "Investors App": (
            "1) Find your contract in Positions in Portfolio\n"
            "2) Click on SELL and enter the order details\n"
            "3) Enter the quantity and click on BUY at the bottom\n"
            "4) Confirm the order and swipe to buy"
        ),
        "Swift Trade": (
            "1) Find your contract in Open Positions under Reports\n"
            "2) Enter the quantity and order details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "Encash": {
        "Traders App": (
            "1) Select the desired Stock or ETF in Portfolio or Demat Holdings\n"
            "2) Click on SELL → enter quantity → choose product type as Encash\n"
            "   Set Market or Limit price → click on SELL\n"
            "3) Confirm the order and swipe to sell at the bottom"
        ),
        "Investors App": (
            "1) Select the desired Stock or ETF in Portfolio or Demat Holdings\n"
            "2) Click on SELL → enter quantity → choose product type as Encash\n"
            "   Set Market or Limit price → click on SELL\n"
            "3) Confirm the order and swipe to sell at the bottom"
        ),
        "Swift Trade": (
            "1) Select the desired Stock or ETF in the Demat balance under Reports\n"
            "2) Click on SELL → enter quantity → choose product type as Encash\n"
            "   Set Market or Limit price → click on SELL\n"
            "3) Confirm the order and swipe to sell at the bottom"
        ),
    },
    "GTDT": {
        "Traders App": (
            "1) Search for the desired Stock or ETF in the search tab\n"
            "2) Click on BUY → enter price and quantity → Advanced options → select GTD\n"
            "   Set GTD till date (max 90 calendar days) → click BUY\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        "Investors App": (
            "1) Search for the desired Stock or ETF in the search tab\n"
            "2) Click on BUY → select GTD (Till Date) → enter quantity, price, GTD till date\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        "Swift Trade": (
            "1) Select GTDt order in Equity under the BUY column\n"
            "2) Enter scrip name, stop loss trigger price, selling price, and other details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
    "Intersettlement": {
        "Traders App": (
            "1) Click on your initials (top left) → Demat Holdings → T1 holdings\n"
            "2) Select the desired stock → click SELL → enter quantity\n"
            "   Choose product type as CASH, set price → click SELL\n"
            "3) Confirm the order and swipe to sell at the bottom"
        ),
        "Investors App": (
            "1) Select the desired Stock or ETF in Portfolio or Demat Holdings\n"
            "2) Click on SELL → enter quantity → choose delivery → set price → click SELL\n"
            "3) Confirm the order and swipe to sell at the bottom"
        ),
        "Swift Trade": (
            "1) Select the desired Stock or ETF in Demat balance under Reports\n"
            "2) Click on SELL → enter quantity → choose delivery → set price → click SELL\n"
            "3) Confirm the order and swipe to sell at the bottom"
        ),
    },
    "Cover": {
        "Traders App": (
            "1) Search for the desired Stock or ETF in the search tab\n"
            "2) Click on BUY → enter quantity → choose product type as Cover\n"
            "   Set stop loss trigger price and sell price → click BUY\n"
            "3) Confirm the order and swipe to buy at the bottom"
        ),
        # Investors App does NOT support Cover — not included
        "Swift Trade": (
            "1) Select Cover in Equity under the BUY column\n"
            "2) Enter scrip name, stop loss trigger price, sell price, and other details\n"
            "3) Click on the Place order tab to proceed\n"
            "4) Confirm the order on the next screen"
        ),
    },
}


def get_instructions(trade_type: str, app: str) -> str | None:
    """
    Look up step-by-step instructions for a trade_type + app combination.
    Returns None if the combination is not supported (e.g. Cover + Investors App).
    """
    entry = INSTRUCTIONS.get(trade_type)
    if entry is None:
        return None
    return entry.get(app)


def get_available_apps(trade_type: str) -> list[str]:
    """Return the list of apps that support the given trade type."""
    entry = INSTRUCTIONS.get(trade_type, {})
    return list(entry.keys())
