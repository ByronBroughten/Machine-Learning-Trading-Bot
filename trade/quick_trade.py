import tda, time, json
import utils, config

from tda.orders.common import Duration, Session
from tda.utils import Utils
from tda.orders.equities import equity_buy_market, equity_sell_market

def quick_trade():
    cfg = config.Client()

    client = tda.auth.easy_client(
        cfg.api_key, cfg.redirect_url, cfg.token_path,
        webdriver_func=utils.misc.Get_Webdriver(cfg.webdriver_path))

    order = client.place_order(
            cfg.account_id,
            equity_buy_market('IBM', 1).build())
            # equity_buy_market('IBM', 1).set_session(Session.SEAMLESS).build())

    assert order.ok, order.raise_for_status()
    order_id = Utils(client, cfg.account_id).extract_order_id(order)
    assert order_id is not None

    time.sleep(1)

    order_info = client.get_order(order_id, cfg.account_id)
    orders = client.get_orders_by_query()
    accounts = client.get_accounts(fields=client.Account.Fields.ORDERS)

    print(order_info.json())
    print(orders.json())
    print(accounts.json())

    order_stuff = {
        'order_info': order_info, 
        'orders': orders,
        'accounts': accounts,
    },

    return order_stuff


