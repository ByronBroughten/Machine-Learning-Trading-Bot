#%%
import tda, selenium, time
import utils
import config
cfg = config.Tda()

client = tda.auth.easy_client(
            cfg.api_key, cfg.redirect_url, cfg.token_path, webdriver_func=utils.misc.Get_Webdriver(cfg.webdriver_path))

#%%
quote = client.get_quote('IBM')
accounts = client.get_accounts(fields=None)
accounts = client.get_accounts(fields=client.Account.Fields.ORDERS)
# client.Account.Fields.ORDERS

# ('orders', 'positions') can be passed into fields

# quote.json()['IBM']['lastPrice']
# quote.json()['IBM']['bidPrice']

# ['assetType', 'assetMainType', 'cusip', 'symbol', 'description', 'bidPrice', 'bidSize', 'bidId', 'askPrice', 'askSize',
#  'askId', 'lastPrice','lastSize','lastId','openPrice','highPrice','lowPrice','bidTick','closePrice','netChange',
#  'totalVolume','quoteTimeInLong','tradeTimeInLong','mark','exchange','exchangeName','marginable','shortable','volatility',
#  'digits','52WkHigh','52WkLow','nAV','peRatio','divAmount','divYield','divDate','securityStatus','regularMarketLastPrice',
#  'regularMarketLastSize','regularMarketNetChange','regularMarketTradeTimeInLong','netPercentChangeInDouble',
#  'markChangeInDouble','markPercentChangeInDouble','regularMarketPercentChangeInDouble','delayed']

# 'availableFunds': 2438.65,
# 'longMarketValue': 123.4,
# 'cashBalance': 2377.23,
# 'buyingPower': 4877.3,
# 'buyingPowerNonMarginableTrade': 2438.65,
#  'equity': 2500.63,

# quantity = get_quantity(amount_in_account, quote, fraction)


#%%
from tda.orders.common import Duration, Session
from tda.utils import Utils
from tda.orders.equities import equity_buy_market, equity_sell_market
order = client.place_order(
            cfg.account_id,
            equity_buy_market('IBM', 1).build())
            # equity_buy_market('IBM', 1).set_session(Session.SEAMLESS).build())

assert order.ok, order.raise_for_status()
order_id = Utils(client, cfg.account_id).extract_order_id(order)
assert order_id is not None

time.sleep(5)

order_info = client.get_order(order_id, cfg.account_id)
orders = client.get_orders_by_query()

print(order_info.json())
print(orders.json())
#%%
# sell long
order = client.place_order(cfg.account_id, equity_sell_market('IBM', 1).build())

#%%
from tda.orders.equities import equity_sell_short_market, equity_buy_to_cover_market

quantity = 1

order = client.place_order(equity_buy_market('IBM', quantity).build())
order = client.place_order(equity_sell_market('IBM', quantity).build())

# short
order = client.place_order(
    equity_sell_short_market('IBM', quantity).build())
order = order = client.place_order(
    equity_buy_to_cover_market('IBM', quantity).build())

#  https://tda-api.readthedocs.io/en/stable/client.html#placing-new-orders
# Optimizations
# client.cancel_order(order_id, account_id)
# client.replace_order(account_id, order_id, order_spec)



# import json
# order_info = result[0]['order_info'].json()
# orders = result[0]['orders'].json()
# accounts = result[0]['accounts'].json()

# print(json.dumps(order_info, indent=4))
# print(json.dumps(orders, indent=4))
# print(json.dumps(accounts, indent=4))