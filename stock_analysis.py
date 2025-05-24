import yfinance as yf
import math
import json
import pandas as pd
import numpy as np

class Stock:
    def __init__(self, ticker, ticksym):
        self.__ticker = ticker
        self.__info = ticker.info
        self.name = self.__info.get('shortName')
        self.ticksym = ticksym
        self.__ebit = self.__calcEBIT(self.__ticker)
        self.__ey = self.__calcEY(self.__info, self.__ebit)
        self.__bs = ticker.balance_sheet
        self.__roc = self.__calcROC(self.__bs, self.__ebit)
        self.__cashmindebt = self.__calcCashMinDebt(self.__info)
        self.__freecashflow = self.__info.get("freeCashflow")
        self.__ann_earnings_growth = self.__calcAnnEG(self.__ticker)
        self.__cash_debt_ratio = self.__calcCDR(self.__ticker)
        self.__inst_holding = self.__ticker.info.get("heldPercentInstitutions")*100

    def __calcCDR(self, ticker):
        return ticker.balancesheet.loc['Cash And Cash Equivalents'].iloc[0] / ticker.balancesheet.loc['Total Debt'].iloc[0]        
       

    def __calcAnnEG(self, ticker):
        incomestmt = ticker.incomestmt
        earnings = list()
        for date, ebit in incomestmt.loc['EBIT'].items():
            if not math.isnan(ebit):
                earnings.append((date, ebit))
        sorted_earnings = sorted(earnings, key = lambda x: x[0])
        cumsum = 0
        avdenom = 0
        for i in range(len(sorted_earnings)-1):
            initearn = sorted_earnings[i][1]
            finearn = sorted_earnings[i+1][1]
            percinc = ((finearn - initearn)/abs(initearn))*100
            cumsum += percinc
            avdenom += 1
        return cumsum / avdenom
            

    def __calcEBIT(self, ticker):
        # Get EBITDA from .info
        ebitda = ticker.info.get("ebitda")
        
        # Get depreciation from cash flow statement
        cf = ticker.cashflow
        #depreciation = None
        
        # Try both common labels
        depreciation = cf.loc["Depreciation And Amortization"].iloc[0]
        
        # Calculate EBIT
        if ebitda is not None and depreciation is not None:
            ebit = ebitda - depreciation
            return ebit
        else:
            print(f"Missing data to compute EBIT for {self.name}.")
            return ebitda

    def __calcEY(self, info, ebit):
        # gathering items
        market_cap = info.get("marketCap")               # Market Capitalization
        total_debt = info.get("totalDebt")               # Total debt
        cash = info.get("totalCash")                     # Cash & equivalents

        # checking for NoneType 
        assert(market_cap is not None)
        assert(total_debt is not None)
        assert(cash is not None)
        
        # Calculate Enterprise Value
        enterprise_value = market_cap + total_debt - cash
        
        # Earnings Yield (as Greenblatt defines it)
        earnings_yield = ebit / enterprise_value

        # ran into a special case
        if ebit < 0 and earnings_yield > 0:
            earnings_yield = -earnings_yield
        return earnings_yield

    def __calcROC(self, bs, ebit):
        # Balance sheet values
        total_current_assets = bs.loc["Current Assets"].iloc[0]
        cash = bs.loc["Cash"][0] if "Cash" in bs.index else 0
        current_assets_adj = total_current_assets - cash
        
        total_current_liabilities = bs.loc["Current Liabilities"].iloc[0]
        
        # Net Working Capital (excluding cash)
        nwc = current_assets_adj - total_current_liabilities
        
        # Net Fixed Assets = Total Assets – Total Current Assets – Intangibles
        total_assets = bs.loc["Total Assets"].iloc[0]
        intangibles = bs.loc["Intangible Assets"][0] if "Intangible Assets" in bs.index else 0
        net_fixed_assets = total_assets - total_current_assets - intangibles
        
        # Invested Capital = NWC + Net Fixed Assets
        invested_capital = nwc + net_fixed_assets
        
        # Greenblatt ROIC
        roc = ebit / invested_capital
        return roc

    def __calcCashMinDebt(self, info):
        total_debt = info.get("totalDebt")               # Total debt
        cash = info.get("totalCash")                     # Cash & equivalents
        assert(total_debt is not None)
        assert(cash is not None)
        return cash - total_debt

    def getEY(self):
        return self.__ey

    def getROC(self):
        return self.__roc

    def getCMD(self):
        return self.__cashmindebt
 
    def getFCF(self):
        return self.__freecashflow

    def getAvAnnEarnGrowth(self):
        return self.__ann_earnings_growth

    def getCDR(self):
        return self.__cash_debt_ratio

    def getInstHoldPer(self):
        return self.__inst_holding

    def forJSON(self):
        return (self.name, dict(ey=self.__ey*100, 
                                roc=self.__roc*100, 
                                cashdebtratio=self.__cash_debt_ratio, 
                                fcf=self.__freecashflow, 
                                avganearngrowth=self.__ann_earnings_growth, 
                                instholdper=self.__inst_holding))

    def forPandasDF(self):
        return  dict(tick=self.ticksym,
                     ey=self.__ey*100, 
                     roc=self.__roc*100, 
                     cashdebtratio=self.__cash_debt_ratio, 
                     fcfebit=self.__freecashflow/self.__ebit, 
                     avganearngrowth=self.__ann_earnings_growth, 
                     instholdper=self.__inst_holding)

class Stocks:
    def __init__(self, stocks):
        self.stocks = stocks
        self.__stocks_df = self.__toDataFrame(stocks)
        self.__norm_stocks = self.__normalizeStocks(self.__stocks_df, self.stocks)
        self.__agg_vals = self.__getAggVals(self.__norm_stocks)

    def __updateInternals(self):
        self.__stocks_df = self.__toDataFrame(stocks)
        self.__norm_stocks = self.__normalizeStocks(self.__stocks_df)

    def __toDataFrame(self, stocks):
        stock_list = [x.forPandasDF() for x in stocks]
        return pd.DataFrame(stock_list)

    def __getMedianDict(self, df):
        # get the median of each column
        col_len = len(df.columns)
        med_dict = dict()
        for col in range(col_len):
            series = df.iloc[:, col]
            ser_name = series.name
            if ser_name != 'tick':
                med_dict[ser_name] = series.median()
        return(med_dict)

    def __getMADDict(self, df, med_dict):
        # get the median of each column
        col_len = len(df.columns)
        mad_dict = dict()
        for col in range(col_len):
            series = df.iloc[:, col]
            ser_name = series.name
            if ser_name != 'tick':
                arr = series.to_numpy()
                median = med_dict[ser_name]
                mad = np.median(np.abs(arr - median))
                mad_dict[ser_name] = mad
        return(mad_dict)

    def __normalizeStocks(self, df, stocks):
        # create a median dictionary
        med_dict = self.__getMedianDict(df)
        # create a MAD dictionary
        mad_dict = self.__getMADDict(df, med_dict)
        # create a new list of stocks that are normalized
        norm_list = list()
        for stock in stocks:
            norm_stock = stock.forPandasDF()
            for key in med_dict.keys():
                x = norm_stock[key]
                median = med_dict[key]
                mad = mad_dict[key]
                norm_stock[key] = (x - median) / mad
            norm_list.append(norm_stock)
        return pd.DataFrame(norm_list)

    def __getAggVals(self, norm_stocks):
        agg_vals = dict()
        num_rows = len(norm_stocks)
        for rnum in range(num_rows):
            row = norm_stocks.iloc[rnum]
            arr = row.to_numpy()
            tick = arr[0]
            inst_hold = arr[-1]
            vals = np.delete(np.delete(arr, -1), 0)
            sums = np.sum(vals) - inst_hold # sub inst hold because the lower the better
            agg_vals[tick] = sums
        return agg_vals


    def addStock(self, stock):
        self.stocks.append(stock)
        self.__updateInternals()

    def getJSON(self):
        json_dict = dict()
        for stock in self.stocks:
            json_data = stock.forJSON()
            json_dict[json_data[0]] = json_data[1]
        return json_dict

    def getDF(self):
        return self.__stocks_df

    def getNormStocks(self):
        return self.__norm_stocks       

    def getAggVals(self):
        return self.__agg_vals

tickers = () # add stock tickers here
stock_list = [Stock(yf.Ticker(tick), tick) for tick in tickers]
stocks = Stocks(stock_list)
json_print = stocks.getJSON()

with open("stock_data.json", "w") as outfile:
    json.dump(json_print, outfile, indent=4)
print("Data written to JSON")

print(stocks.getDF())
print(stocks.getNormStocks())

for key, value in sorted(stocks.getAggVals().items(), key=lambda item: item[1], reverse=True):
    print(f"{key} -> {value}")

