# Webull-Python-API-Stock-Market-Data-Candlestick-Plot
Plotting candlestick data in Python using Webull's API


To `import` and `login` Webull (plugin your email and account password):

    !pip install webull
    from webull import webull
    wb = webull()
    wb.login('my@email.com', 'password')

To obtain stock data, we'll use the `get_bars` function and specify the stock's symbol, timeframe, and number of samples (i.e. `stock`, `interval`, and `count`) and save it as a `DataFrame` using `pandas`'s `to_csv` function:

    stock_symbol = 'SPY'
    stock_data = wb.get_bars(stock=stock_symbol, interval='m1', count=390, extendTrading=0)

    _date = '2023-09-26'
    file_name = f'/content/drive/My Drive/Colab Notebooks/DATA_FOLDERS/DATA_FRAMES/{stock_symbol}_{_date}.csv'
    import pandas as pd
    stock_data.to_csv(file_name)

where `'m1'` refers to a one-minute timeframe.

We'll then retrieve our saved stock data using `pandas`'s `read_csv` function and parse the `timestamp`s into time and date:

    df = pd.read_csv(file_name)
    
    def parse_date_from_timestamp(timestamp):
        return timestamp[:timestamp.find(' ')]
    
    def parse_time_from_timestamp(timestamp):
        def correct_timestamps(timestamp):
            '''convert timestamp from 24h to 12h'''
            return timestamp if int(timestamp[:2]) <= 12 else '0' + str(int(timestamp[:2]) - 12) + timestamp[2:]
        return correct_timestamps(timestamp[timestamp.find(' ') + 1:timestamp.find('-4:') - 5])
    
    df['date'] = df.timestamp.map(parse_date_from_timestamp)
    df['timestamp'] = df.timestamp.map(parse_time_from_timestamp)

We'll `def`ine a `candlestick_plot_function` to group the data into `numpy` `arrays` and then plot the data as candlesticks using a combination of `BoxStyle`, `FancyBboxPatch`, and `Line2D` `from` `matplotlib`:

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import BoxStyle
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D

    def candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=30, wick_linewidth=2.0, fancy_box_padding=0.0005):
        def calc_num_candlesticks(df, candlestick_size_in_minutes):
            return int(df.index.stop / candlestick_size_in_minutes) if df.index.stop / candlestick_size_in_minutes == int(df.index.stop / candlestick_size_in_minutes) else int(df.index.stop / candlestick_size_in_minutes) + 1
    
        def calc_box_width(num_candlesticks):
            return 0.98 - (num_candlesticks / 125)
    
        def remove_prefixed_zero(timestamp):
            return timestamp[:5] if timestamp[0] != '0' else timestamp[1:5]
    
        y = np.zeros((calc_num_candlesticks(df, candlestick_size_in_minutes), 4)).astype(float)
        box_width = calc_box_width(y.shape[0])
    
        ax.set_facecolor([0, 0, 0.35])
        ax.grid(which='major', axis='both', color=[1, 1, 1], linewidth=0.5, zorder=0)
        for candlestick in range(y.shape[0]):
            indexes_per_candlestick = range(candlestick*candlestick_size_in_minutes, (candlestick + 1)*candlestick_size_in_minutes, 1)
            if indexes_per_candlestick.stop > df.index.stop:
                indexes_per_candlestick = range(candlestick*candlestick_size_in_minutes, df.index.stop, 1)
            data = df.iloc[indexes_per_candlestick]
            y[candlestick, :] = np.array([data.open.iloc[0], np.max(data.high), np.min(data.low), data.close.iloc[-1]])
    
            top_of_box = np.max([y[candlestick, 0], y[candlestick, 3]])
            bottom_of_box = np.min([y[candlestick, 0], y[candlestick, 3]])
            box_color = np.array([0.0, 0.8, 0.6941]) if y[candlestick, 0] < y[candlestick, 3] else np.array([1.0, 0.0, 0.0])
    
            ax.add_line(Line2D(xdata=(candlestick, candlestick), ydata=(y[candlestick, 2], bottom_of_box), color=box_color, linewidth=wick_linewidth, antialiased=True, zorder=2))
            ax.add_line(Line2D(xdata=(candlestick, candlestick), ydata=(y[candlestick, 1], bottom_of_box), color=box_color, linewidth=wick_linewidth, antialiased=True, zorder=2))
            ax.add_patch(FancyBboxPatch(xy=(candlestick - box_width*0.5, bottom_of_box), width=box_width, height=top_of_box - bottom_of_box, facecolor=box_color, edgecolor=box_color, boxstyle=BoxStyle('round', pad=fancy_box_padding), zorder=2))
    
        ax.set_ylim(np.min(y) - 0.1*(np.max(y) - np.min(y)), np.max(y) + 0.1*(np.max(y) - np.min(y)))
        ax.set_xlim(-0.1*y.shape[0], y.shape[0] + 0.1*y.shape[0])
        ax.set_title(f'''
            {stock_symbol}
            {candlestick_size_in_minutes} Minute Candlesticks''', fontsize=15, fontweight='bold')
        ax.set_yticks([_y for _y in ax.get_yticks()][1:-1])
        ax.set_yticklabels(['{:.2f}'.format(_y) for _y in ax.get_yticks()])
        ax.set_xticks([0, int(y.shape[0]*0.5), y.shape[0] - 1])
        ax.set_xticklabels([remove_prefixed_zero(df.timestamp.values[0]), remove_prefixed_zero(df.timestamp.values[int(df.timestamp.values.shape[0]*0.5)]), remove_prefixed_zero(df.timestamp.values[-1])])
        for axis in ['left', 'right', 'top', 'bottom']:
            ax.spines[axis].set_visible(False) if axis in ['top', 'right'] else ax.spines[axis].set_linewidth(5)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
            label.set_fontweight('bold')
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol)
    plt.show()


Which creates a 30-minute candlesticks plot:

![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/8de61b45-b5fd-43ad-810b-de800d044085)


We can also adjust the `candlestick_size_in_minutes` parameter to plot different timeframes; for example 10-minute candlesticks:

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=10, wick_linewidth=1.0)
    plt.show()

![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/6e5ceabe-94d9-420d-8cff-31043268ff43)


or 60-minute candlesticks:

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=60)
    plt.show()


![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/d859c8b2-5db5-48aa-b13f-1b21e99ce86e)



# Grouping Candlesticks by Timestamp in a Vectorized Process:


First we need a `list` of start and stop times as hour and minute `int`egers that can span across one or more days:

    def get_candlesticks(df, candlestick_size_in_minutes=30):
        '''returns a list of pairs of time stamps as integers [hour (start), minute (start), next hour (stop), next minute (stop)] from 9:30am up to (but not including) 4:00pm grouped by the candlestick_size_in_minutes parameter'''
        __hour = df.hour.iloc[0]
        def round_minute_down(__minute, candlestick_size_in_minutes):
            candlesticks = np.arange(0, 60 + candlestick_size_in_minutes, candlestick_size_in_minutes).astype(int)
            first_candlestick = np.where(np.abs(candlesticks - __minute) == np.min(np.abs(candlesticks - __minute)))
            if candlesticks[first_candlestick] > __minute:
                first_candlestick[0][0] -= 1
            return candlesticks[first_candlestick[0][0]]

        __minute = round_minute_down(df.minute.iloc[0], candlestick_size_in_minutes)
        __date = 0
        candlesticks = [[__hour, __minute, __hour + 1 if __minute + candlestick_size_in_minutes > 60 else __hour, min(__minute + candlestick_size_in_minutes, 60), np.unique(np.array(df.date))[__date]]]
        for _ in range(int((390 - candlestick_size_in_minutes) / candlestick_size_in_minutes)*np.unique(np.array(df.date)).shape[0]):
            __minute += candlestick_size_in_minutes
            if __minute >= 60:
                __minute -= 60
                __hour += 1
            if __hour == 13:
                __hour = 1
            if __hour == 4:
                __hour = 9
                __minute = 30
                __date += 1
            if __date >= np.unique(np.array(df.date)).shape[0] - 1 and candlesticks[-1][1] <= df.minute.iloc[-1] < candlesticks[-1][3] and candlesticks[-1][0] == df.hour.iloc[-1] and candlesticks[-1][4] == df.date.iloc[-1]:
                return candlesticks
            candlesticks.append([__hour, __minute, __hour + 1 if __minute + candlestick_size_in_minutes > 60 else __hour, min(__minute + candlestick_size_in_minutes, 60), df.date.iloc[0] if np.unique(np.array(df.date)).shape[0] == 1 else np.unique(np.array(df.date))[min(__date, np.unique(np.array(df.date)).shape[0] - 1)]])
        return candlesticks


We'll then `def`ine a function that `return`s `True` for indexes that fall between the start and stop hours and minutes `list` and match the correct `'date'`:

    def is_a_candlestick(df_hours, df_minutes, _hour, _minute, _next_hour, _next_minute, df_date, _date):
        '''returns True for timestamps ([hour, minute]) that match the correct candlestick; conditions vectorized for numpy.where() function'''
        if all([df_date == _date, df_hours == _hour, _hour == _next_hour, df_minutes >= _minute, df_minutes < _next_minute]) or all([df_date == _date, df_hours == _hour, _hour != _next_hour, df_minutes >= _minute, df_minutes < _next_minute if _minute < _next_minute else True]) or all([df_date == _date, df_hours == _next_hour, _next_minute != 60, df_minutes < _next_minute, df_minutes >= _minute if _minute < _next_minute else True]):
            return True

    is_a_candlestick = np.vectorize(is_a_candlestick)


What `numpy`'s `vectorize` function does is it allows us to `def`ine a function hook into which another `numpy` function can get as a vectorized input. In the simplest case, which we did, is use an `if-statement` that `return`s `True`, which when passed to `numpy`'s `where()` function, will provide us the indexes that meet our condition(s), without having to iterate through each one:

    for candlestick, (_hour, _minute, _next_hour, _next_minute, _date) in enumerate(candlesticks):
        indexes = np.where(is_a_candlestick(np.array(df.hour), np.array(df.minute), _hour, _minute, _next_hour, _next_minute, np.array(df.date), _date))
        y[candlestick, :] = np.array([df.open.iloc[indexes].iloc[0], np.max(df.high.iloc[indexes]), np.min(df.low.iloc[indexes]), df.close.iloc[indexes].iloc[-1]])


Altogether we have:

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import BoxStyle
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    
    def candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=30, wick_linewidth=2.0, fancy_box_padding=0.0005):
        def get_candlesticks(df, candlestick_size_in_minutes=30):
            '''returns a list of pairs of time stamps as integers [hour (start), minute (start), next hour (stop), next minute (stop)] from 9:30am up to (but not including) 4:00pm grouped by the candlestick_size_in_minutes parameter'''
            __hour = df.hour.iloc[0]
            def round_minute_down(__minute, candlestick_size_in_minutes):
                candlesticks = np.arange(0, 60 + candlestick_size_in_minutes, candlestick_size_in_minutes).astype(int)
                first_candlestick = np.where(np.abs(candlesticks - __minute) == np.min(np.abs(candlesticks - __minute)))
                if candlesticks[first_candlestick] > __minute:
                    first_candlestick[0][0] -= 1
                return candlesticks[first_candlestick[0][0]]
    
            __minute = round_minute_down(df.minute.iloc[0], candlestick_size_in_minutes)
            __date = 0
            candlesticks = [[__hour, __minute, __hour + 1 if __minute + candlestick_size_in_minutes > 60 else __hour, min(__minute + candlestick_size_in_minutes, 60), np.unique(np.array(df.date))[__date]]]
            for _ in range(int((390 - candlestick_size_in_minutes) / candlestick_size_in_minutes)*np.unique(np.array(df.date)).shape[0]):
                __minute += candlestick_size_in_minutes
                if __minute >= 60:
                    __minute -= 60
                    __hour += 1
                if __hour == 13:
                    __hour = 1
                if __hour == 4:
                    __hour = 9
                    __minute = 30
                    __date += 1
                if __date >= np.unique(np.array(df.date)).shape[0] - 1 and candlesticks[-1][1] <= df.minute.iloc[-1] < candlesticks[-1][3] and candlesticks[-1][0] == df.hour.iloc[-1] and candlesticks[-1][4] == df.date.iloc[-1]:
                    return candlesticks
                candlesticks.append([__hour, __minute, __hour + 1 if __minute + candlestick_size_in_minutes > 60 else __hour, min(__minute + candlestick_size_in_minutes, 60), df.date.iloc[0] if np.unique(np.array(df.date)).shape[0] == 1 else np.unique(np.array(df.date))[min(__date, np.unique(np.array(df.date)).shape[0] - 1)]])
            return candlesticks
    
        def calc_box_width(num_candlesticks):
            '''returns a box_width value such that candlesticks are close together but not touching or overlapping'''
            return 0.98 - (num_candlesticks / 125)
    
        def remove_prefixed_zero(timestamp):
            '''returns for example "9:30" in place of "09:30"'''
            return timestamp[:5] if timestamp[0] != '0' else timestamp[1:5]
    
        def is_a_candlestick(df_hours, df_minutes, _hour, _minute, _next_hour, _next_minute, df_date, _date):
            '''returns True for timestamps ([hour, minute]) that match the correct candlestick; conditions vectorized for numpy.where() function'''
            if all([df_date == _date, df_hours == _hour, _hour == _next_hour, df_minutes >= _minute, df_minutes < _next_minute]) or all([df_date == _date, df_hours == _hour, _hour != _next_hour, df_minutes >= _minute, df_minutes < _next_minute if _minute < _next_minute else True]) or all([df_date == _date, df_hours == _next_hour, _next_minute != 60, df_minutes < _next_minute, df_minutes >= _minute if _minute < _next_minute else True]):
                return True
    
        is_a_candlestick = np.vectorize(is_a_candlestick)
    
        candlesticks = get_candlesticks(df, candlestick_size_in_minutes)
        y = np.zeros((len(candlesticks), 4)).astype(float)
        box_width = calc_box_width(y.shape[0])
    
        ax.set_facecolor([0, 0, 0.35])
        ax.grid(which='major', axis='both', color=[1, 1, 1], linewidth=0.5, zorder=0)
        for candlestick, (_hour, _minute, _next_hour, _next_minute, _date) in enumerate(candlesticks):
            indexes = np.where(is_a_candlestick(np.array(df.hour), np.array(df.minute), _hour, _minute, _next_hour, _next_minute, np.array(df.date), _date))
            y[candlestick, :] = np.array([df.open.iloc[indexes].iloc[0], np.max(df.high.iloc[indexes]), np.min(df.low.iloc[indexes]), df.close.iloc[indexes].iloc[-1]])
    
            top_of_box = np.max([y[candlestick, 0], y[candlestick, 3]])
            bottom_of_box = np.min([y[candlestick, 0], y[candlestick, 3]])
            box_color = np.array([0.0, 0.8, 0.6941]) if y[candlestick, 0] < y[candlestick, 3] else np.array([1.0, 0.0, 0.0])
    
            ax.add_line(Line2D(xdata=(candlestick, candlestick), ydata=(y[candlestick, 2], bottom_of_box), color=box_color, linewidth=wick_linewidth, antialiased=True, zorder=2))
            ax.add_line(Line2D(xdata=(candlestick, candlestick), ydata=(y[candlestick, 1], bottom_of_box), color=box_color, linewidth=wick_linewidth, antialiased=True, zorder=2))
            ax.add_patch(FancyBboxPatch(xy=(candlestick - box_width*0.5, bottom_of_box), width=box_width, height=top_of_box - bottom_of_box, facecolor=box_color, edgecolor=box_color, boxstyle=BoxStyle('round', pad=fancy_box_padding), zorder=2))
    
        ax.set_ylim(np.min(y) - 0.1*(np.max(y) - np.min(y)), np.max(y) + 0.1*(np.max(y) - np.min(y)))
        ax.set_xlim(-0.1*y.shape[0], y.shape[0] + 0.1*y.shape[0])
        ax.set_title(f'''
            {stock_symbol}
            {candlestick_size_in_minutes} Minute Candlesticks''', fontsize=15, fontweight='bold')
        ax.set_yticks([_y for _y in ax.get_yticks()][1:-1])
        ax.set_yticklabels(['{:.2f}'.format(_y) for _y in ax.get_yticks()])
        ax.set_xticks([0, int(y.shape[0]*0.5), y.shape[0] - 1])
        ax.set_xticklabels([remove_prefixed_zero(df.timestamp.values[0]), remove_prefixed_zero(df.timestamp.values[int(df.timestamp.values.shape[0]*0.5)]), remove_prefixed_zero(df.timestamp.values[-1])])
        for axis in ['left', 'right', 'top', 'bottom']:
            ax.spines[axis].set_visible(False) if axis in ['top', 'right'] else ax.spines[axis].set_linewidth(5)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
            label.set_fontweight('bold')
    
    stock_symbol = 'SPY'
    _date = '2023-09-27'
    
    file_name = f'/content/drive/My Drive/Colab Notebooks/DATA_FOLDERS/DATA_FRAMES/{stock_symbol}_{_date}.csv'
    
    import pandas as pd
    df = pd.read_csv(file_name)
    
    def parse_date_from_timestamp(timestamp):
        return timestamp[:timestamp.find(' ')]
    
    def parse_time_from_timestamp(timestamp):
        def correct_timestamps(timestamp):
            '''convert timestamp from 24h to 12h'''
            return timestamp if int(timestamp[:2]) <= 12 else '0' + str(int(timestamp[:2]) - 12) + timestamp[2:]
        return correct_timestamps(timestamp[timestamp.find(' ') + 1:timestamp.find('-4:') - 5])
    
    df['date'] = df.timestamp.map(parse_date_from_timestamp)
    df['timestamp'] = df.timestamp.map(parse_time_from_timestamp)
    
    def get_hour(timestamp):
        return int(timestamp[:2])
    
    def get_minute(timestamp):
        return int(timestamp[3:5])
    
    df['hour'] = df.timestamp.map(get_hour)
    df['minute'] = df.timestamp.map(get_minute)
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=30)
    plt.show()


![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/487f8960-ca21-4876-ae15-1846ed82ecfd)


    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=60)
    plt.show()


![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/179392a6-573e-4688-99aa-97e075a5961c)





