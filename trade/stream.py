# makes basic out of raw and appends to a dataframe.
import torch, tda, asyncio, os, json, time, pandas as pd, datetime as dt
from tda.streaming import StreamClient
from pathlib import Path
import utils, config, munge, train

class Store_Checker(object):    
    def __init__(self, key, seq_len):
        self.key = key
        self.seq_len = seq_len
    
    def __call__(self, store):
        # modify this to assess for gaps in the sequence length
        if len(store[self.key] + 1) >= self.seq_len:
        # + 1 because store len doesn't update while open
            self.data_sufficient = True
            self.basic_seq = store[self.key][-seq_len:]
            return self
    
def make_experience_datum(df, data_of):

    # models should have their scalers stored with them
    # I need dv, for that matter
    done_data = {}

    for dk, dv in data_of.items():
        
        if dk == 'ys': continue
        elif dk == 'context': data = df[-1:]
        elif dk in ('seqs', 'pics'): data = df

        data = munge.process.glob_maker_funcs[dk](df, **dv['dataglobs']['make'])
        # maybe need to add an axis

    if 'funcs' in dv['processing']:
        data = munge.process.process_funcs(dv['processing']['funcs'])
        

        scalers = dv['processing']['scale']['scalers']
        if 'scale_func' not in scalers:

            # I don't think I want to pass the scaler as a json...
            scalers = dv['processing']['scale']['scalers']
            for i, scaler_info in enumerate(scalers):
                try:
                    scalers[i]['scaler'] = joblib.load(scaler_info['path'])
                except FileNotFoundError:
                    scalers[i]['scaler'] = munge.scaler.load_scaler_params(scaler_info['package'],
                                                                            scaler_info['type'],
                                                                            scaler_info['params'])

            scalers['scale_func'] = scale.Transform(dv['processing']['scale'], dk)
        
        done_data[dk] = scalers['scale_func'](dataset)
    
    # live_set = {'live': done_data}
    return done_data
    
def standardize_raw(msg):
    raw_chart_data = msg['content'][0]
    standard_data = {}

    conversions = [['datetime', 'CHART_TIME'], ['open', 'OPEN_PRICE'], ['high', 'HIGH_PRICE'],
                ['low', 'LOW_PRICE'], ['close', 'CLOSE_PRICE'], ['volume', 'VOLUME']]
    for cv in conversions:
        standard_data[cv[0]] = raw_chart_data[cv[1]]

    mins_since_epoch = int(standard_data['datetime']/60000)
    df = pd.DataFrame(standard_data, index=[mins_since_epoch])
    return df
    
def prep_to_basic(df):
    df['datetime'] = utils.misc.stamp_mils_to_secs(df['datetime'])
    df = munge.basic.get_seasonality(df) # uses utc time
    df.datetime = df.datetime.astype('int32')
    return df

def append_and_assess_store(basic_df, path, key, assess_func=None):
    with pd.HDFStore(path) as store:
        try: assert basic_df.index[0] in store[key].index
        except (AssertionError, KeyError):

            basic_df.to_hdf(store, key=key, mode='a', append=True, format='table')
            if assess_func:
                return assess_func(store)

def init_trading_paths(cfg):
        paths = {
            'outer': '',
            'beliefs_json': os.path.join(cfg.trial_dir, 'params.json'),
            'data_info_json': os.path.join(cfg.trial_dir, 'data_info.json'),
            'beliefs_model': os.path.join(cfg.trial_dir, cfg.beliefs_dir),
        }
        paths['outer_data']: os.join(paths['outer'], 'data', 'live')
        return paths

def load_trading_params(paths):
    params = {
        'beliefs': utils.misc.json_to_dict(paths['beliefs_json']),
        'data_info': utils.misc.json_to_dict(paths['data_info_json']),
    }
    return params

class Trader(object):
    def __init__(self, cfg):

        self.client = cfg.client
        self.paths = init_paths(cfg)
        self.params = load_params(self.paths)
        self.get_dirs_paths_and_keys(self.paths, self.params)

        # for making features
        self.data_of = self.params['data_info']['data_of']
        
        # for deciding to trade
        self.acceptable_lag_secs = cfg.acceptable_lag_secs
        self.seq_len = self.data_of['seqs']['dataglobs']['make']['seq_len']
        self.store_checker = Store_Checker(self.key, self.seq_len)

        # for making trades
        device = utils.pt.check_device()
        self.beliefs, self.beliefs_package = train.init.read_beliefs(self.paths['beliefs_model'], device)

    def get_dirs_paths_and_keys(paths, params):
    # get dir, paths, and key
        price_data_info = params['data_info']['price_data_info']['training_and_trading']
        self.tda_datatype = price_data_info['tda_datatype']
        self.step_mins = price_data_info['step_mins']
        self.tda_datatype_dir = munge.paths.get_tda_datatype_dir(self.tda_datatype, self.step_mins)
        self.key = munge.paths.get_symbol_key(cfg.symbol)
        self.paths['basic'] = munge.paths.get_data_path(paths['outer_data'], self.tda_dataype_dir, mode='glob', head='basic.h5')

    def __call__(self, msg):
        
        # make it only work within trading hours
        # set this up so the function only and always runs during trading hours

        standard_raw = standardize_raw(msg)
        single_basic = prep_to_basic(standard_raw)
        chart_time = single_basic['datetime'][0]

        self.store_checker = append_and_assess_store(single_basic, self.paths['basic'], self.key,
                                                                    assess_func=self.store_checker)


        if self.decide_if_trade(chart_time, self.store_checker.data_sufficient):
            experience_datum = make_experience_datum(self.store_checker.basic_seq, self.data_of)
            # experience = self.make_experience(experience_datum)
            experience = utils.misc.Objdict(experience_datum)
            open_action = self.decide_open_action(experience)
            
            # this can work for either hsl or sl
            if open_action != 0: # 0 is hold
            
                quantity = self.get_quantity()
                order = self.open_position(open_action)
                # I'll need the $amount that was traded (for making ys)
                # what and the quantity of what was traded (for closing)
                # I could close the specific order, too, but I guess that doesn't
                # really matter too much

                orders.update(order)
                # I'll need to get my active orders somehow

        self.close_positions(experience)

    def decide_if_trade(self, chart_time_secs, data_sufficient):
    # chart time secs is from the begining of the last one-minute candlestick
    # so 70 means 10 seconds after the candlestick was generated
    
        if data_sufficient:
            time_elapsed = int(time.time()) - chart_time_secs
            soon_enough = time_elapsed < self.acceptable_lag_secs + 60
        
            print('time_elapsed', time_elapsed)
            print('soon_enough', soon_enough)
            print('self.data_sufficient', data_sufficient)

        return soon_enough and data_sufficient
    
    def make_experience(experience_data):
        pass

    def decide_open_action(self, experience):
        pred = self.beliefs(experience)
        action = (torch.argmax(q_pred, 1) + 1).tolist()
        return action[0]
        
        #figure out how many trades I'm going to make at a time
        # if seq_len > 100, then maybe 100 at a time to follow the 1% rule
        # otherwise, seq_len many
        # maybe make it so that no matter what each trade is only 1% of my money at a time
        # Im not sure, though.
    
    def get_quantity():
        # placeholder
        return 1

    def open_position(action, symbol, quantity, client, candle_time):
    # put this in a file. maybe the file can be something like, "action file"
    # or "order file"
        
        if action == 1:
            order_type = equity_sell_short_market(symbol, quantity)
        elif action == 2:
            order_type = equity_buy_market(symbol, quantity)    

        order = client.place_order(cfg.account_id, equity_buy_market(symbol, quantity).build())

        assert order.ok, order.raise_for_status()
        order_id = Utils(client, cfg.account_id).extract_order_id(order)
        assert order_id is not None

        #!!!!!!!!!!!!!!!!
        order_stats = client.get_order(order_id, cfg.account_id)
        orders = client.get_orders_by_query()
        accounts = client.get_accounts(fields=client.Account.Fields.ORDERS)
        # one of these should give me the entry price that I need

#        "orderLegCollection": [
#         {
#             "orderLegType": "EQUITY",
#             "legId": 1,
#             "instrument": {
#                 "assetType": "EQUITY",
#                 "cusip": "459200101",
#                 "symbol": "IBM"
#             },
#             "instruction": "BUY",
#             "positionEffect": "OPENING",
#             "quantity": 1.0
#         }
#     ],
        order_stats = order_stats['orderLegCollection'][0]
        order_info = {
            'order_id': order_id,
            'symbol': order_stats['symbol'],
            'positon': order_stats['instruction'], # probably not quite right
            'quantity': order_stats['quantity'], # probably not quite right
            'candle_time': candle_time,
            # 'price': None, # not exactly sure how to get this. not exactly sure if it matters.
        }
        return order_info

    def close_position(self, df):
        # make it so that the position closes if enough candle_time has passed
        # but you'll have to use a way to translate candletime to a range
        pass


    
    def make_glob_paths(self):
        # save to zarr hierarchy with symbol as key
        self.paths['seqs'] = os.path.join(folder_path, 'seqs', 'symbol_label.zarr')
        self.paths['context'] = os.path.join(folder_path, 'context', 'symbol_label.zarr')
        self.paths['ys'] = os.path.join(folder_path, 'ys', 'symbol_label.zarr')

def print_handler(msg):
    print(json.dumps(msg, indent=4))

async def read_stream(stream_client, cfg):
    await stream_client.login()
    await stream_client.quality_of_service(StreamClient.QOSLevel.EXPRESS)
    
    glob_hanlder = Trader(cfg) # tda_data_type is gotten from the datatype that was trained on
    await getattr(stream_client, glob_hanlder.tda_data_type)([cfg.symbol])
    # await stream_client.nasdaq_book_subs(['IBM'])
    #await stream_client.chart_equity_subs([cfg.symbol])

    
    stream_client.add_chart_equity_handler(print_handler)
    # stream_client.add_timesale_options_handler(
    #         lambda msg: print(json.dumps(msg, indent=4)))

    while True:
        await stream_client.handle_message()

def get_live_clients(cfg):
    client = tda.auth.easy_client(
            cfg.api_key, cfg.redirect_url, cfg.token_path, webdriver_func=utils.misc.Get_Webdriver(cfg.webdriver_path))

    stream_client = StreamClient(client, account_id=cfg.account_id)

    return client, stream_client

def stream():
    cfg = config.Tda()
    client, stream_client = get_live_clients(cfg.clients)
    cfg.handler.client = client
    asyncio.get_event_loop().run_until_complete(read_stream(stream_client, cfg.handler))

# sample stream candle
# {
#     "service": "CHART_EQUITY",
#     "timestamp": 1599258716699,
#     "command": "SUBS",
#     "content": [
#         {
#             "seq": 165,
#             "key": "IBM",
#             "OPEN_PRICE": 122.3,
#             "HIGH_PRICE": 122.3,
#             "LOW_PRICE": 122.3,
#             "CLOSE_PRICE": 122.3,
#             "VOLUME": 597572.0,
#             "SEQUENCE": 690,
#             "CHART_TIME": 1599258600000,
#             "CHART_DAY": 18509
#         }
#     ]
# }


# all kinds of order stuff from quick_trade

# order stats
# {
#     "session": "NORMAL",
#     "duration": "DAY",
#     "orderType": "MARKET",
#     "complexOrderStrategyType": "NONE",
#     "quantity": 1.0,
#     "filledQuantity": 0.0,
#     "remainingQuantity": 1.0,
#     "requestedDestination": "AUTO",
#     "destinationLinkName": "AutoRoute",
#     "orderLegCollection": [
#         {
#             "orderLegType": "EQUITY",
#             "legId": 1,
#             "instrument": {
#                 "assetType": "EQUITY",
#                 "cusip": "459200101",
#                 "symbol": "IBM"
#             },
#             "instruction": "BUY",
#             "positionEffect": "OPENING",
#             "quantity": 1.0
#         }
#     ],
#     "orderStrategyType": "SINGLE",
#     "orderId": 3285821115,
#     "cancelable": true,
#     "editable": false,
#     "status": "QUEUED",
#     "enteredTime": "2020-09-04T23:19:24+0000",
#     "accountId": 497141542
# }

# orders -- maybe this will be what I want once I have orders out there
# []

# accounts
# [
#     {
#         "securitiesAccount": {
#             "type": "MARGIN",
#             "accountId": "497141542",
#             "roundTrips": 0,
#             "isDayTrader": false,
#             "isClosingOnlyRestricted": false,
#             "orderStrategies": [
#                 {
#                     "session": "NORMAL",
#                     "duration": "DAY",
#                     "orderType": "MARKET",
#                     "complexOrderStrategyType": "NONE",
#                     "quantity": 1.0,
#                     "filledQuantity": 0.0,
#                     "remainingQuantity": 1.0,
#                     "requestedDestination": "AUTO",
#                     "destinationLinkName": "AutoRoute",
#                     "orderLegCollection": [
#                         {
#                             "orderLegType": "EQUITY",
#                             "legId": 1,
#                             "instrument": {
#                                 "assetType": "EQUITY",
#                                 "cusip": "459200101",
#                                 "symbol": "IBM"
#                             },
#                             "instruction": "BUY",
#                             "positionEffect": "OPENING",
#                             "quantity": 1.0
#                         }
#                     ],
#                     "orderStrategyType": "SINGLE",
#                     "orderId": 3285821115,
#                     "cancelable": true,
#                     "editable": false,
#                     "status": "QUEUED",
#                     "enteredTime": "2020-09-04T23:19:24+0000",
#                     "accountId": 497141542
#                 },
#                 {
#                     "session": "NORMAL",
#                     "duration": "DAY",
#                     "orderType": "MARKET",
#                     "complexOrderStrategyType": "NONE",
#                     "quantity": 1.0,
#                     "filledQuantity": 0.0,
#                     "remainingQuantity": 1.0,
#                     "requestedDestination": "AUTO",
#                     "destinationLinkName": "AutoRoute",
#                     "orderLegCollection": [
#                         {
#                             "orderLegType": "EQUITY",
#                             "legId": 1,
#                             "instrument": {
#                                 "assetType": "EQUITY",
#                                 "cusip": "459200101",
#                                 "symbol": "IBM"
#                             },
#                             "instruction": "BUY",
#                             "positionEffect": "OPENING",
#                             "quantity": 1.0
#                         }
#                     ],
#                     "orderStrategyType": "SINGLE",
#                     "orderId": 3285798336,
#                     "cancelable": true,
#                     "editable": false,
#                     "status": "QUEUED",
#                     "enteredTime": "2020-09-04T22:36:54+0000",
#                     "accountId": 497141542
#                 }
#             ],
#             "initialBalances": {
#                 "accruedInterest": 0.0,
#                 "availableFundsNonMarginableTrade": 2500.95,
#                 "bondValue": 0.0,
#                 "buyingPower": 5001.9,
#                 "cashBalance": 2500.95,
#                 "cashAvailableForTrading": 0.0,
#                 "cashReceipts": 0.0,
#                 "dayTradingBuyingPower": 0.0,
#                 "dayTradingBuyingPowerCall": 0.0,
#                 "dayTradingEquityCall": 0.0,
#                 "equity": 2500.95,
#                 "equityPercentage": 2500.95,
#                 "liquidationValue": 2500.95,
#                 "longMarginValue": 0.0,
#                 "longOptionMarketValue": 0.0,
#                 "longStockValue": 0.0,
#                 "maintenanceCall": 0.0,
#                 "maintenanceRequirement": 0.0,
#                 "margin": 2500.95,
#                 "marginEquity": 2500.95,
#                 "moneyMarketFund": 0.0,
#                 "mutualFundValue": 0.0,
#                 "regTCall": 0.0,
#                 "shortMarginValue": 0.0,
#                 "shortOptionMarketValue": 0.0,
#                 "shortStockValue": 0.0,
#                 "totalCash": 2500.95,
#                 "isInCall": false,
#                 "pendingDeposits": 0.0,
#                 "marginBalance": 0.0,
#                 "shortBalance": 0.0,
#                 "accountValue": 2500.95
#             },
#             "currentBalances": {
#                 "accruedInterest": 0.0,
#                 "cashBalance": 2500.95,
#                 "cashReceipts": 0.0,
#                 "longOptionMarketValue": 0.0,
#                 "liquidationValue": 2500.95,
#                 "longMarketValue": 0.0,
#                 "moneyMarketFund": 0.0,
#                 "savings": 0.0,
#                 "shortMarketValue": -0.0,
#                 "pendingDeposits": 0.0,
#                 "availableFunds": 0.0,
#                 "availableFundsNonMarginableTrade": 2500.95,
#                 "buyingPower": 5001.9,
#                 "buyingPowerNonMarginableTrade": 2378.63,
#                 "dayTradingBuyingPower": 0.0,
#                 "equity": 2745.59,
#                 "equityPercentage": 100.0,
#                 "longMarginValue": 244.64,
#                 "maintenanceCall": 0.0,
#                 "maintenanceRequirement": 0.0,
#                 "marginBalance": 0.0,
#                 "regTCall": 0.0,
#                 "shortBalance": 0.0,
#                 "shortMarginValue": 0.0,
#                 "shortOptionMarketValue": -0.0,
#                 "sma": 2500.95,
#                 "bondValue": 0.0
#             },
#             "projectedBalances": {
#                 "availableFunds": 2378.63,
#                 "availableFundsNonMarginableTrade": 2378.63,
#                 "buyingPower": 4757.26,
#                 "dayTradingBuyingPower": 0.0,
#                 "dayTradingBuyingPowerCall": 0.0,
#                 "maintenanceCall": 0.0,
#                 "regTCall": 0.0,
#                 "isInCall": false,
#                 "stockBuyingPower": 4757.26
#             }
#         }
#     }
# ]